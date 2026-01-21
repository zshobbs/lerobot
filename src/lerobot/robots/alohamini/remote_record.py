import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import sys
from pathlib import Path

import cv2
import numpy as np
import websockets
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

# Define the exact state ordering used by LeKiwiUni
STATE_KEYS = [
    "arm_left_shoulder_pan.pos",
    "arm_left_shoulder_lift.pos",
    "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos",
    "arm_left_wrist_roll.pos",
    "arm_left_gripper.pos",
    "x.vel",
    "y.vel",
    "theta.vel",
    "lift_axis.height_mm",
]

# Camera configuration matching LeKiwiUniConfig
CAMERA_CONFIG = {
    "head_top": (480, 640),
    "wrist_right": (480, 640),
    "head_front": (480, 640),
    "wrist_left": (480, 640),
}

def decode_image(b64_string: str) -> np.ndarray:
    if not b64_string:
        logging.debug("decode_image: Empty base64 string provided.")
        return None
    try:
        logging.debug(f"decode_image: Decoding base64 string of length {len(b64_string)}")
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logging.error("decode_image: cv2.imdecode returned None.")
            return None
        logging.debug(f"decode_image: Decoded image shape: {frame.shape} (BGR)")
        # Convert BGR (OpenCV) to RGB (LeRobotDataset expectation)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return None

def build_features(fps: int):
    """Constructs the feature dictionary for LeRobotDataset."""
    features = {
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_KEYS),),
            "names": STATE_KEYS,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_KEYS),),
            "names": STATE_KEYS,
        },
    }
    
    for cam_name, (h, w) in CAMERA_CONFIG.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    
    logging.debug(f"build_features: Constructed features: {list(features.keys())}")
    return features

class RemoteRecorder:
    def __init__(self, host: str, port: int, repo_id: str, root: Path, fps: int):
        self.uri = f"ws://{host}:{port}/ws?quality=hq"
        self.repo_id = repo_id
        self.root = root
        self.fps = fps
        self.dataset = None
        self.current_episode_recording = False
        self.stop_event = asyncio.Event()
        self.loop = None
        logging.debug(f"RemoteRecorder initialized: host={host}, port={port}, repo_id={repo_id}, root={root}, fps={fps}")

    async def connect_and_record(self):
        logging.info(f"Connecting to {self.uri}...")
        
        # Initialize dataset immediately to verify path/permissions
        self.init_dataset()
        
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(self.uri, max_size=None) as websocket:
                    logging.info("Connected to Robot Web Server. Waiting for recording signal...")
                    
                    while not self.stop_event.is_set():
                        message = await websocket.recv()
                        msg_size = len(message)
                        logging.debug(f"Received websocket message. Size: {msg_size} bytes")
                        
                        data = json.loads(message)
                        msg_type = data.get("type")
                        logging.debug(f"Parsed message type: {msg_type}")
                        
                        if msg_type == "robot_state":
                            await self.process_frame(data)
                        else:
                            logging.debug(f"Ignored message type: {msg_type}")
                            
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                logging.warning(f"Connection lost: {e}. Reconnecting in 3s...")
                if self.current_episode_recording:
                    logging.warning("Episode interrupted by disconnect! Saving partial episode...")
                    self.dataset.save_episode()
                    self.current_episode_recording = False
                await asyncio.sleep(3)
            except Exception as e:
                logging.error(f"Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def init_dataset(self):
        if self.dataset is None:
            logging.debug("init_dataset: initializing LeRobotDataset...")
            features = build_features(self.fps)
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.root,
                features=features,
                use_videos=True, 
                image_writer_processes=1, # Use subprocess for image writing
                image_writer_threads=4 
            )
            logging.info(f"Dataset initialized at {self.root}")

    async def process_frame(self, data: dict):
        is_recording = data.get("is_recording", False)
        frame_count = data.get("frame_count", -1)
        logging.debug(f"process_frame: is_recording={is_recording}, frame_count={frame_count}")
        
        # State Machine Transition: Start Recording
        if is_recording and not self.current_episode_recording:
            logging.info("Recording STARTED by remote signal.")
            self.current_episode_recording = True
            
        # State Machine Transition: Stop Recording
        if not is_recording and self.current_episode_recording:
            logging.info("Recording STOPPED by remote signal. Saving episode...")
            self.dataset.save_episode()
            self.current_episode_recording = False
            logging.info(f"Episode saved. Total episodes: {self.dataset.num_episodes}")

        # Process Data if Recording
        if self.current_episode_recording:
            try:
                logging.debug("process_frame: Parsing frame data...")
                frame_data = self.parse_frame_data(data)
                if frame_data:
                    logging.debug("process_frame: Adding frame to dataset.")
                    self.dataset.add_frame(frame_data)
                    # print(".", end="", flush=True) # Progress indicator
                else:
                    logging.warning("process_frame: Frame data parse failed (returned None).")
            except Exception as e:
                logging.error(f"Error processing frame: {e}", exc_info=True)

    def parse_frame_data(self, data: dict) -> dict | None:
        """Parses the WebSocket message into a LeRobotDataset compatible frame."""
        obs_data = data.get("data", {})
        act_data = data.get("action", {})
        images_b64 = data.get("images", {})
        
        logging.debug(f"parse_frame_data: obs_keys={list(obs_data.keys())}, act_keys={list(act_data.keys())}, image_keys={list(images_b64.keys())}")
        
        # 1. Vectorize Observation State
        state_vec = []
        for key in STATE_KEYS:
            val = obs_data.get(key, 0.0)
            state_vec.append(val)
        
        # 2. Vectorize Action
        # Note: server_action_state might not have all keys initially, fill with 0.0 or matching state?
        # Ideally, actions align with state keys for position-controlled motors.
        action_vec = []
        for key in STATE_KEYS:
            # If action is missing, use current state (no-op) or 0.0
            # For velocity keys (x.vel), 0.0 is safe.
            # For position keys, using current state is safer than 0.0 if missing.
            default_val = obs_data.get(key, 0.0)
            val = act_data.get(key, default_val) 
            action_vec.append(val)

        # 3. Decode Images
        decoded_images = {}
        for cam_name in CAMERA_CONFIG:
            b64 = images_b64.get(cam_name)
            if b64:
                img = decode_image(b64)
                if img is not None:
                    decoded_images[f"observation.images.{cam_name}"] = img
                else:
                    logging.warning(f"Failed to decode image: {cam_name}")
                    return None # Drop frame if image corrupted
            else:
                # If image missing (e.g. different fps or packet loss), skipping frame is usually best
                # or we could use previous frame if implemented. For now, skip.
                logging.debug(f"Missing image for {cam_name}. Skipping frame.")
                return None

        # Assemble Frame
        frame = {
            "observation.state": torch.tensor(state_vec, dtype=torch.float32),
            "action": torch.tensor(action_vec, dtype=torch.float32),
            **decoded_images
        }
        
        # TODO: Get task from message if available?
        # For now, use a default task or allow it to be set via args/message
        frame["task"] = "teleoperation" 
        
        logging.debug(f"parse_frame_data: Frame assembled. State shape: {frame['observation.state'].shape}")
        
        return frame

    def run(self):
        # Handle graceful shutdown
        def handler(signum, frame):
            logging.info("Shutting down...")
            self.stop_event.set()
            
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        try:
            asyncio.run(self.connect_and_record())
        finally:
            if self.dataset:
                if self.current_episode_recording:
                    logging.info("Saving final in-progress episode before exit...")
                    self.dataset.save_episode()
                self.dataset.finalize()
                logging.info("Dataset finalized.")

def main():
    parser = argparse.ArgumentParser(description="LeRobot Remote Recorder")
    parser.add_argument("--host", type=str, required=True, help="Robot IP address")
    parser.add_argument("--port", type=int, default=8000, help="Robot Web Server Port")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset Repo ID (e.g., lerobot/my_dataset)")
    parser.add_argument("--root", type=Path, default="data/remote_recordings", help="Local root directory for dataset")
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Always use DEBUG level if requested, otherwise INFO
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info(f"Starting RemoteRecorder with log_level={'DEBUG' if args.debug else 'INFO'}")

    recorder = RemoteRecorder(
        host=args.host,
        port=args.port,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps
    )
    recorder.run()

if __name__ == "__main__":
    main()
