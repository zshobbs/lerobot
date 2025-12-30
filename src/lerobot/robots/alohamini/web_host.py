#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import base64
from contextlib import asynccontextmanager
import os

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from lerobot.robots.alohamini.config_lekiwi_uni import LeKiwiUniConfig
from lerobot.robots.alohamini.lekiwi_uni import LeKiwiUni

robot: LeKiwiUni | None = None
latest_action: dict | None = None
latest_observation: dict = {}
server_action_state: dict = {}


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


async def stream_robot_state():
    """Periodically fetches robot state, encodes images, and broadcasts everything."""
    global latest_observation
    while True:
        if robot:
            # Update the lift controller to execute its control loop
            robot.lift.update()

            obs = robot.get_observation()
            latest_observation = obs.copy()  # Store the latest observation
            
            state_payload = {}
            image_payload = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    # Convert BGR (OpenCV default) to RGB for browser display
                    rgb_frame = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                    ret, buffer = cv2.imencode(".jpg", rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        image_payload[k] = base64.b64encode(buffer).decode("utf-8")
                else:
                    # Convert numpy floats to python floats for JSON serialization
                    if isinstance(v, (np.float32, np.float64)):
                        state_payload[k] = float(v)
                    else:
                        state_payload[k] = v
            
            await manager.broadcast(json.dumps({
                "type": "robot_state",
                "data": state_payload,
                "images": image_payload,
            }))
        
        # Adjust sleep time for desired frequency (e.g., 20Hz)
        await asyncio.sleep(1 / 20)


async def send_merged_actions():
    """Periodically sends the merged action state to the robot."""
    while True:
        if robot and server_action_state:
            # Create a copy to avoid race conditions if the state is updated during send
            action_to_send = server_action_state.copy()
            logging.debug(f"Sending merged action: {action_to_send}")
            robot.send_action(action_to_send)
        # Match the rate of the leader arm bridge
        await asyncio.sleep(1 / 50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global robot
    logging.info("Configuring LeKiwi Uni-arm Robot")
    robot_config = LeKiwiUniConfig()
    robot_config.id = "AlohaMiniUniRobot"
    robot = LeKiwiUni(robot_config)

    logging.info("Connecting to AlohaMini")
    robot.connect()

    if not robot.is_connected:
        logging.error("Failed to connect to the robot. Server will not start.")
        raise RuntimeError("Failed to connect to the robot.")

    # Start background tasks
    stream_task = asyncio.create_task(stream_robot_state())
    action_task = asyncio.create_task(send_merged_actions())

    yield

    # Shutdown
    stream_task.cancel()
    action_task.cancel()
    logging.info("Shutting down AlohaMini Host.")
    if robot and robot.is_connected:
        robot.disconnect()


app = FastAPI(lifespan=lifespan)

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
web_dir = os.path.join(script_dir, "web")

# Mount static files
app.mount("/web", StaticFiles(directory=web_dir), name="web")


@app.get("/")
async def root():
    return FileResponse(os.path.join(web_dir, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global latest_action, server_action_state
    LIFT_STEP_MM = 10.0
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "action":
                action_data = message["data"]
                
                # Handle lift control logic
                lift_velocity_command = action_data.pop("lift.vel", None)
                if lift_velocity_command is not None:
                    if lift_velocity_command != 0.0 and latest_observation:
                        # Key is PRESSED: Set a new position target
                        current_height = latest_observation.get("lift_axis.height_mm", 0.0)
                        target_height = current_height + (lift_velocity_command * LIFT_STEP_MM)
                        action_data["lift_axis.height_mm"] = target_height
                        # Ensure we don't also send a velocity command from a previous state
                        if "lift_axis.vel" in server_action_state:
                            del server_action_state["lift_axis.vel"]
                    else:
                        # Key is RELEASED: Send a stop command (velocity 0)
                        action_data["lift_axis.vel"] = 0
                        # Ensure we don't also send a position command
                        if "lift_axis.height_mm" in server_action_state:
                            del server_action_state["lift_axis.height_mm"]

                # Update the shared state with the new data from this client
                logging.debug(f"Updating server_action_state with: {action_data}")
                server_action_state.update(action_data)
                
                # Store for recording purposes
                latest_action = server_action_state.copy()
            else:
                logging.warning(f"Received unknown message type from websocket: {message.get('type')}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except json.JSONDecodeError:
        logging.error("Received invalid JSON from websocket")


def main():
    parser = argparse.ArgumentParser(description="LeRobot Web Host")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Host port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s][%(name)s] %(message)s")
    logging.info(f"Starting LeRobot Web Host on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
