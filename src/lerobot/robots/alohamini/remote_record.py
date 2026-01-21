import argparse
import asyncio
import base64
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from queue import Full, Empty

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
        return None
    try:
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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
        
    return features

def parse_frame_data(data: dict) -> dict | None:
    """Parses the WebSocket message into a LeRobotDataset compatible frame."""
    obs_data = data.get("data", {})
    act_data = data.get("action", {})
    images_b64 = data.get("images", {})
    
    # 1. Vectorize Observation State
    state_vec = []
    for key in STATE_KEYS:
        val = obs_data.get(key, 0.0)
        state_vec.append(val)
    
    # 2. Vectorize Action
    action_vec = []
    for key in STATE_KEYS:
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
            return None

    # Assemble Frame
    frame = {
        "observation.state": torch.tensor(state_vec, dtype=torch.float32),
        "action": torch.tensor(action_vec, dtype=torch.float32),
        **decoded_images
    }
    
    frame["task"] = "teleoperation" 
    
    return frame

def dataset_worker(queue: multiprocessing.Queue, repo_id: str, root: Path, fps: int):
    """
    Worker process to handle dataset I/O and video encoding.
    Consumes raw JSON data from the queue.
    """
    # Ignore SIGINT so the main process handles the shutdown signal and this worker
    # can finish saving/finalizing cleanly when it receives the sentinel (None).
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Configure logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Worker] [%(levelname)s] %(message)s"
    )
    
    logging.info(f"Dataset Worker started. Root: {root}")
    
    features = build_features(fps)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        features=features,
        use_videos=True, 
        image_writer_processes=1, 
        image_writer_threads=4 
    )
    
    current_episode_recording = False
    frame_in_episode_count = 0
    
    try:
        while True:
            try:
                data = queue.get(timeout=1.0)
            except Empty:
                continue

            if data is None:
                logging.info("Received stop signal.")
                break
                
            is_recording = data.get("is_recording", False)
            
            # State Machine Transition: Start Recording
            if is_recording and not current_episode_recording:
                logging.info("Recording STARTED.")
                current_episode_recording = True
                frame_in_episode_count = 0
                
            # State Machine Transition: Stop Recording
            if not is_recording and current_episode_recording:
                logging.info(f"Recording STOPPED. Total frames in episode: {frame_in_episode_count}")
                logging.info("Saving episode (this may take a moment)...")
                st = time.time()
                dataset.save_episode()
                et = time.time()
                current_episode_recording = False
                logging.info(f"Episode saved in {et-st:.2f}s. Total episodes: {dataset.num_episodes}")

            # Process Data if Recording
            if current_episode_recording:
                try:
                    frame_data = parse_frame_data(data)
                    if frame_data:
                        dataset.add_frame(frame_data)
                        frame_in_episode_count += 1
                        if frame_in_episode_count % 30 == 0:
                            logging.info(f"Recording progress: {frame_in_episode_count} frames (~{frame_in_episode_count/fps:.1f}s)")
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
                    
    except Exception as e:
        logging.error(f"Worker crashed: {e}", exc_info=True)
    finally:
        if current_episode_recording:
            logging.info("Saving in-progress episode before exit...")
            dataset.save_episode()
        dataset.finalize()
        logging.info("Dataset finalized. Worker exiting.")

class RemoteRecorder:
    def __init__(self, host: str, port: int, repo_id: str, root: Path, fps: int):
        self.uri = f"ws://{host}:{port}/ws?quality=hq"
        self.queue_size = 2048
        self.queue = multiprocessing.Queue(maxsize=self.queue_size)
        
        self.worker = multiprocessing.Process(
            target=dataset_worker,
            args=(self.queue, repo_id, root, fps)
        )
        self.stop_event = asyncio.Event()
        self.frame_receive_count = 0

    def start(self):
        self.worker.start()

    async def connect_and_record(self):
        logging.info(f"Connecting to {self.uri}...")
        
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(self.uri, max_size=None) as websocket:
                    logging.info("Connected to Robot Web Server. Relaying data...")
                    
                    while not self.stop_event.is_set():
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data.get("type") == "robot_state":
                            self.frame_receive_count += 1
                            # Blocking put (waits if queue is full), wrapped in thread to keep WS alive
                            await asyncio.to_thread(self.queue.put, data)
                            
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                logging.warning(f"Connection lost: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)
            except Exception as e:
                logging.error(f"Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def run(self):
        self.start()
        
        # Handle graceful shutdown
        def handler(signum, frame):
            logging.info("Shutting down...")
            self.stop_event.set()
            
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        try:
            asyncio.run(self.connect_and_record())
        finally:
            logging.info("Stopping worker process...")
            self.queue.put(None)
            self.worker.join(timeout=5.0)
            
            if self.worker.is_alive():
                logging.warning("Worker did not exit in time. Terminating...")
                self.worker.terminate()
                self.worker.join()
            
            logging.info("Cleanup complete.")

def main():
    parser = argparse.ArgumentParser(description="LeRobot Remote Recorder")
    parser.add_argument("--host", type=str, required=True, help="Robot IP address")
    parser.add_argument("--port", type=int, default=8000, help="Robot Web Server Port")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset Repo ID (e.g., lerobot/my_dataset)")
    parser.add_argument("--root", type=Path, default="data/remote_recordings", help="Local root directory for dataset")
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

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