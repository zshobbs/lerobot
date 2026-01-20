#!/usr/bin/env python

import asyncio
import argparse
import json
import base64
import logging
import cv2
import numpy as np
import websockets
from datetime import datetime
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class RemoteRecorder:
    def __init__(self, host_url, dataset_root=None, fps=30):
        self.host_url = host_url
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.fps = fps
        self.dataset = None
        self.is_recording = False
        self.frame_count = 0
        self.current_repo_id = None
        self.features = None
        
        # Hardcoded feature definition for LeKiwiUni
        self.state_names = [
            "arm_left_shoulder_pan.pos",
            "arm_left_shoulder_lift.pos",
            "arm_left_elbow_flex.pos",
            "arm_left_wrist_flex.pos",
            "arm_left_wrist_roll.pos",
            "arm_left_gripper.pos",
            "x.vel",
            "y.vel",
            "theta.vel",
            "lift_axis.height_mm"
        ]
        
        self.action_names = self.state_names
        
        # Cameras will be inferred from the first message
        self.camera_names = []

    def _init_dataset(self, repo_id, first_observation):
        """Initialize or load the LeRobotDataset."""
        # Determine the full path
        if self.dataset_root:
            dataset_path = self.dataset_root / repo_id
        else:
            from lerobot.utils.constants import HF_LEROBOT_HOME
            dataset_path = Path(HF_LEROBOT_HOME) / repo_id

        if dataset_path.exists():
            logging.info(f"Loading existing dataset for resume at {dataset_path}")
            self.dataset = LeRobotDataset(repo_id, root=dataset_path)
            # Infer camera names for image processing from existing metadata
            self.camera_names = []
            for key in self.dataset.meta.camera_keys:
                 # LeRobot features use 'observation.images.cam_name'
                 cam_name = key.split(".")[-1]
                 self.camera_names.append(cam_name)
            
            # Start image writer for recording
            self.dataset.start_image_writer(num_threads=4 * len(self.camera_names))
            self.features = self.dataset.features
            return

        # Define features for new dataset
        features = {}
        features[f"{OBS_STR}.state"] = {
            "dtype": "float32",
            "shape": (len(self.state_names),),
            "names": self.state_names,
        }
        features[ACTION] = {
            "dtype": "float32",
            "shape": (len(self.action_names),),
            "names": self.action_names,
        }
        
        self.camera_names = []
        for cam_name, b64_str in first_observation.get("images", {}).items():
            try:
                img_bytes = base64.b64decode(b64_str)
                img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_array is not None:
                    h, w, c = img_array.shape
                    features[f"{OBS_STR}.images.{cam_name}"] = {
                        "dtype": "video", 
                        "shape": (h, w, c),
                        "names": ["height", "width", "channels"],
                    }
                    self.camera_names.append(cam_name)
            except Exception as e:
                logging.warning(f"Failed to decode image for {cam_name}: {e}")

        logging.info(f"Creating new dataset {repo_id} at {dataset_path}")
        
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=self.fps,
            features=features,
            root=dataset_path,
            robot_type="lekiwi_uni",
            use_videos=True,
            image_writer_threads=4 * len(self.camera_names)
        )
        self.features = features

    async def run(self):
        logging.info(f"Connecting to {self.host_url}...")
        while True:
            try:
                async with websockets.connect(self.host_url) as websocket:
                    logging.info("Connected to Robot.")
                    
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "recording_status":
                            status = data.get("status")
                            if status == "started":
                                new_repo_id = data.get("repo_id")
                                if self.is_recording and self.current_repo_id == new_repo_id:
                                    logging.info(f"Ignoring duplicate START signal for {new_repo_id}")
                                    continue
                                
                                self.current_repo_id = new_repo_id
                                self.is_recording = True
                                self.frame_count = 0
                                logging.info(f"START RECORDING: {self.current_repo_id}")
                                self.dataset = None 
                                
                            elif status == "stopped":
                                self.is_recording = False
                                logging.info(f"STOP RECORDING. Total frames captured: {self.frame_count}")
                                self.save_current_episode()

                        elif msg_type == "robot_state":
                            if not self.is_recording:
                                continue
                                
                            obs_state = data.get("data", {})
                            images_b64 = data.get("images", {})
                            action_state = data.get("action", {})
                            
                            if self.dataset is None:
                                if not self.current_repo_id:
                                    logging.error("Recording active but no repo_id set.")
                                    self.is_recording = False
                                    continue
                                self._init_dataset(self.current_repo_id, {"images": images_b64})
                                logging.info("Recording initialized.")

                            frame = {}
                            state_vec = [obs_state.get(name, 0.0) for name in self.state_names]
                            frame[f"{OBS_STR}.state"] = np.array(state_vec, dtype=np.float32)
                            
                            action_vec = []
                            for name in self.action_names:
                                val = action_state.get(name)
                                if val is None:
                                    val = obs_state.get(name, 0.0)
                                action_vec.append(val)
                            frame[ACTION] = np.array(action_vec, dtype=np.float32)
                            
                            for cam_name in self.camera_names:
                                if cam_name in images_b64:
                                    try:
                                        img_bytes = base64.b64decode(images_b64[cam_name])
                                        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                                        if img_bgr is not None:
                                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                            frame[f"{OBS_STR}.images.{cam_name}"] = img_rgb
                                    except Exception as e:
                                        logging.warning(f"Image error {cam_name}: {e}")
                            
                            frame["task"] = "teleoperation"

                            try:
                                self.dataset.add_frame(frame)
                                self.frame_count += 1
                                if self.frame_count % 10 == 0:
                                    print(f"Recording... {self.frame_count} frames", end='\r')
                            except Exception as e:
                                logging.error(f"Error adding frame: {e}")
                                self.is_recording = False
                                
            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
                logging.warning("Connection lost. Retrying in 3 seconds...")
                self.is_recording = False
                self.dataset = None
                await asyncio.sleep(3)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                await asyncio.sleep(1)

    def save_current_episode(self):
        if self.dataset:
            logging.info("Saving episode...")
            try:
                self.dataset.save_episode()
                self.dataset.finalize()
                logging.info("Episode saved successfully.")
            except Exception as e:
                logging.error(f"Failed to save episode: {e}")
            finally:
                self.dataset = None

def main():
    parser = argparse.ArgumentParser(description="Remote Recorder for LeRobot Web Host")
    parser.add_argument("--host", type=str, required=True, help="Robot WebSocket URL (e.g., ws://192.168.1.100:8000/ws)")
    parser.add_argument("--root", type=str, default="data", help="Root directory for datasets")
    parser.add_argument("--fps", type=int, default=30, help="FPS assumption for dataset metadata")
    args = parser.parse_args()
    
    recorder = RemoteRecorder(args.host, args.root, args.fps)
    try:
        asyncio.run(recorder.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        recorder.save_current_episode()
    print("\nExiting...")

if __name__ == "__main__":
    main()