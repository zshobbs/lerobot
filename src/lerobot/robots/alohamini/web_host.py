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
import base64
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import os
from functools import partial

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from lerobot.robots.alohamini.config_lekiwi import LeKiwiConfig
from lerobot.robots.alohamini.lekiwi import LeKiwi

# Note: LeRobotDataset is NOT imported here to save resources on the Pi.
# Recording is offloaded to a remote client.

robot: LeKiwi | None = None
latest_action: dict | None = None
latest_observation: dict = {}
server_action_state: dict = {}

class RecordingState:
    """
    Manages the recording state (flag and metadata) but DOES NOT write to disk.
    It serves as a signal source for the remote recorder.
    """
    def __init__(self):
        self.is_recording = False
        self.repo_id = None
        self.frame_count = 0

    def start(self, repo_id: str):
        if self.is_recording:
            return
        logging.info(f"Broadcasting START recording signal: {repo_id}")
        self.repo_id = repo_id
        self.is_recording = True
        self.frame_count = 0

    def stop(self):
        if not self.is_recording:
            return
        logging.info("Broadcasting STOP recording signal.")
        self.is_recording = False

    def increment_frame(self):
        if self.is_recording:
            self.frame_count += 1

recording_state = RecordingState()


class ConnectionManager:
    def __init__(self):
        self.lq_connections: list[WebSocket] = []
        self.hq_connections: list[WebSocket] = []
        
        # LQ: Track active send tasks to drop frames if busy (Low Latency)
        self.lq_active_tasks: dict[WebSocket, asyncio.Task] = {}
        
        # HQ: Use a queue to buffer frames (Reliability)
        self.hq_queues: dict[WebSocket, asyncio.Queue] = {}
        self.hq_sender_tasks: dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, quality: str = "lq"):
        await websocket.accept()
        if quality == "hq":
            self.hq_connections.append(websocket)
            # Create a safety buffer for network jitter (approx 6 seconds @ 30fps)
            # This protects the Robot's RAM while allowing smooth sending.
            queue = asyncio.Queue(maxsize=200)
            self.hq_queues[websocket] = queue
            # Start a dedicated worker to drain this queue
            self.hq_sender_tasks[websocket] = asyncio.create_task(self._hq_sender_loop(websocket, queue))
        elif quality == "lq":
            self.lq_connections.append(websocket)
        # "silent" or others are ignored for broadcasting

    def disconnect(self, websocket: WebSocket):
        if websocket in self.lq_connections:
            self.lq_connections.remove(websocket)
            if websocket in self.lq_active_tasks:
                self.lq_active_tasks[websocket].cancel()
                del self.lq_active_tasks[websocket]
                
        if websocket in self.hq_connections:
            self.hq_connections.remove(websocket)
            if websocket in self.hq_sender_tasks:
                self.hq_sender_tasks[websocket].cancel()
                del self.hq_sender_tasks[websocket]
            if websocket in self.hq_queues:
                del self.hq_queues[websocket]

    async def _safe_send_lq(self, websocket: WebSocket, message: str):
        """Helper to send LQ message and cleanup task tracker."""
        try:
            await websocket.send_text(message)
        except Exception:
            pass
        finally:
            self.lq_active_tasks.pop(websocket, None)

    async def _hq_sender_loop(self, websocket: WebSocket, queue: asyncio.Queue):
        """Dedicated worker to send HQ frames reliably."""
        try:
            while True:
                message = await queue.get()
                await websocket.send_text(message)
                queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"HQ Sender Error: {e}")

    async def broadcast(self, message_lq: str, message_hq: str):
        # 1. LQ Clients: Drop frame if previous send is still active (Latency priority)
        for connection in self.lq_connections:
            if connection not in self.lq_active_tasks:
                task = asyncio.create_task(self._safe_send_lq(connection, message_lq))
                self.lq_active_tasks[connection] = task
                
        # 2. HQ Clients: Buffer frame in queue (Completeness priority)
        for connection in self.hq_connections:
            queue = self.hq_queues.get(connection)
            if queue:
                try:
                    queue.put_nowait(message_hq)
                except asyncio.QueueFull:
                    logging.warning(f"HQ Client buffer full! Dropping frame for {connection.client}")

manager = ConnectionManager()


def encode_frame(frame: np.ndarray, quality: int) -> str | None:
    """Converts, encodes, and base64-encodes a single image frame."""
    try:
        # Convert BGR (OpenCV default) to RGB for browser display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # JPEG encoding
        ret, buffer = cv2.imencode(".jpg", rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ret:
            return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to encode frame: {e}")
    return None


async def stream_robot_state():
    """Periodically fetches robot state, encodes images in parallel, and broadcasts everything."""
    global latest_observation
    while True:
        try:
            if not robot:
                await asyncio.sleep(0.1)
                continue
            
            obs = await asyncio.to_thread(robot.get_observation)
            latest_observation = obs.copy()

            if recording_state.is_recording:
                recording_state.increment_frame()

            state_payload = {}
            image_frames = {}

            for k, v in obs.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    image_frames[k] = v
                elif isinstance(v, (np.float32, np.float64)):
                    state_payload[k] = float(v)
                else:
                    state_payload[k] = v
            
            # Concurrently encode all images in a thread pool
            encoding_tasks = []
            for key, frame in image_frames.items():
                # partial is used to pass arguments to the function called by to_thread
                lq_task = asyncio.to_thread(partial(encode_frame, frame=frame, quality=50))
                hq_task = asyncio.to_thread(partial(encode_frame, frame=frame, quality=90))
                encoding_tasks.append((key, lq_task, hq_task))
            
            encoded_results = await asyncio.gather(*[t for _, lq_t, hq_t in encoding_tasks for t in (lq_t, hq_t)])
            
            image_payload_lq = {}
            image_payload_hq = {}
            
            result_idx = 0
            for key, _, _ in encoding_tasks:
                lq_b64 = encoded_results[result_idx]
                hq_b64 = encoded_results[result_idx + 1]
                if lq_b64:
                    image_payload_lq[key] = lq_b64
                if hq_b64:
                    image_payload_hq[key] = hq_b64
                result_idx += 2

            base_message = {
                "type": "robot_state",
                "data": state_payload,
                "action": server_action_state,
                "is_recording": recording_state.is_recording,
                "frame_count": recording_state.frame_count,
            }

            message_lq = json.dumps({**base_message, "images": image_payload_lq})
            message_hq = json.dumps({**base_message, "images": image_payload_hq})

            await manager.broadcast(message_lq, message_hq)
            
        except Exception as e:
            logging.error(f"Error in stream_robot_state: {e}")
            
        # Adjust sleep time for desired frequency (e.g., 30Hz)
        await asyncio.sleep(1 / 30)


async def send_merged_actions():
    """Periodically sends the merged action state to the robot."""
    while True:
        try:
            if robot:
                # Execute lift control loop (P-controller) safely
                await asyncio.to_thread(robot.update_lift)

            if robot and server_action_state:
                action_to_send = server_action_state.copy()
                await asyncio.to_thread(robot.send_action, action_to_send)
        except Exception as e:
            logging.error(f"Error in send_merged_actions: {e}")
        # Match the rate of the leader arm bridge (e.g., 50Hz)
        await asyncio.sleep(1 / 50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global robot
    logging.info("Configuring LeKiwi Robot")
    robot_config = LeKiwiConfig()
    robot_config.id = "AlohaMiniRobot"
    robot = LeKiwi(robot_config)

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
    recording_state.stop() # Ensure recording is stopped
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
async def websocket_endpoint(websocket: WebSocket, quality: str = "lq"):
    await manager.connect(websocket, quality)
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
                server_action_state.update(action_data)
                
                # Store for recording purposes
                latest_action = server_action_state.copy()
            
            elif message.get("type") == "start_recording":
                repo_id = message.get("repo_id")
                if not repo_id:
                     repo_id = f"lerobot/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                recording_state.start(repo_id)
                await manager.broadcast(
                    json.dumps({
                        "type": "recording_status",
                        "status": "started",
                        "repo_id": repo_id
                    }),
                    json.dumps({
                        "type": "recording_status",
                        "status": "started",
                        "repo_id": repo_id
                    })
                )
            
            elif message.get("type") == "stop_recording":
                recording_state.stop()
                await manager.broadcast(
                    json.dumps({
                        "type": "recording_status",
                        "status": "stopped"
                    }),
                    json.dumps({
                        "type": "recording_status",
                        "status": "stopped"
                    })
                )

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