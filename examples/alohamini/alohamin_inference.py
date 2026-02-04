#!/usr/bin/env python3

import argparse
import asyncio
import base64
import json
import logging
import time
import os
from pprint import pformat
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import websockets

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini import LeKiwiClientConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.control_utils import predict_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.policies.utils import make_robot_action


class LeKiwiWebClient:
    """
    Async client for LeKiwiUni interacting via WebHost (WebSocket).
    Mimics the interface of LeKiwiClient but using websockets.
    """
    def __init__(self, config: LeKiwiClientConfig):
        self.config = config
        self.remote_ip = config.remote_ip
        # Use default port 8000 or allow override after init
        self.port = 8000 
        self.robot_type = "lekiwi_uni"
        self.websocket = None
        self.camera_map = {} # Map model_cam_name -> host_cam_name
        
        # Features definition matching LeKiwiUni (as served by web_host)
        self._state_ft_keys = [
            "arm_left_shoulder_pan.pos", "arm_left_shoulder_lift.pos",
            "arm_left_elbow_flex.pos", "arm_left_wrist_flex.pos",
            "arm_left_wrist_roll.pos", "arm_left_gripper.pos",
            "x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"
        ]

    @property
    def uri(self):
        return f"ws://{self.remote_ip}:{self.port}/ws"

    @property
    def observation_features(self):
         # The policy preprocessor expects keys with prefixes.
         # But hw_to_dataset_features will add prefixes later.
         # We keep the raw keys here to match the robot's physical features.
         ft = {k: float for k in self._state_ft_keys}
         if self.camera_map:
             for model_name in self.camera_map.keys():
                 ft[model_name] = (480, 640, 3)
         else:
             # Default cameras for LeKiwi
             for name in ["head_top", "head_front", "wrist_left", "wrist_right"]:
                 ft[name] = (480, 640, 3)
         return ft

    @property
    def action_features(self):
        return {k: float for k in self._state_ft_keys}

    async def connect(self):
        logging.info(f"Connecting to {self.uri}...")
        try:
            self.websocket = await websockets.connect(self.uri)
            logging.info("Connected to WebHost.")
        except Exception as e:
            logging.error(f"Failed to connect to {self.uri}: {e}")
            raise e

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            logging.info("Disconnected from WebHost.")

    async def get_observation(self):
        if not self.websocket:
            return {}
        
        try:
            msg_str = await self.websocket.recv()
            msg = json.loads(msg_str)
            
            if msg.get("type") == "robot_state":
                obs = {}
                data = msg.get("data", {})
                obs.update(data)
                
                images = msg.get("images", {})
                
                if self.camera_map:
                    for model_name, host_name in self.camera_map.items():
                        b64_str = images.get(host_name)
                        if b64_str:
                            obs[model_name] = self._decode_b64_image(b64_str, model_name)
                else:
                    for cam_name, b64_str in images.items():
                        if b64_str:
                            obs[cam_name] = self._decode_b64_image(b64_str, cam_name)
                return obs
        except websockets.exceptions.ConnectionClosed:
            logging.warning("WebSocket connection closed.")
        except Exception as e:
            logging.error(f"Error receiving observation: {e}")
            
        return {}

    def _decode_b64_image(self, b64_str, name):
        try:
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logging.warning(f"Failed to decode image {name}: {e}")
            return None

    async def send_action(self, action_dict):
        if not self.websocket:
            return
            
        clean_action = {}
        for k, v in action_dict.items():
            if hasattr(v, "item"):
                clean_action[k] = v.item()
            else:
                clean_action[k] = v
        
        payload = {
            "type": "action",
            "data": clean_action
        }
        
        try:
            await self.websocket.send(json.dumps(payload))
        except Exception as e:
            logging.error(f"Error sending action: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Run inference on Aloha Mini (LeKiwi) via WebHost.")
    parser.add_argument("--hf_model_id", type=str, required=True, help="HuggingFace model repo id (policy) or local path.")
    parser.add_argument("--hf_dataset_id", type=str, default=None, help="HuggingFace dataset repo id (optional, for stats).")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="LeKiwi host IP address.")
    parser.add_argument("--port", type=int, default=8000, help="Web host port (default: 8000).")
    parser.add_argument("--robot_id", type=str, default="lekiwi", help="Robot ID.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument("--task_description", type=str, default="Evaluation task", help="Description of the task.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu).")
    parser.add_argument("--map_cameras", type=str, default=None, help='JSON mapping of model_cam_name:host_cam_name, e.g. \'{"head_top": "cam_1"}\'')

    args = parser.parse_args()
    init_logging()

    model_path = args.hf_model_id
    if os.path.isdir(os.path.join(model_path, "pretrained_model")):
        model_path = os.path.join(model_path, "pretrained_model")
        logging.info(f"Auto-resolved model path to: {model_path}")

    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    robot = LeKiwiWebClient(robot_config)
    robot.port = args.port
    
    if args.map_cameras:
        robot.camera_map = json.loads(args.map_cameras)
        logging.info(f"Using camera mapping: {robot.camera_map}")
    
    await robot.connect()

    # The dataset features used by the policy usually have prefixes like 'observation.images.'
    # We use hw_to_dataset_features which converts raw robot feature keys into dataset feature keys.
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset_stats = None
    ds_meta = None

    if args.hf_dataset_id:
        logging.info(f"Loading dataset stats from {args.hf_dataset_id}...")
        dataset = LeRobotDataset(repo_id=args.hf_dataset_id)
        dataset_stats = dataset.meta.stats
        ds_meta = dataset.meta
    else:
        stats_path = os.path.join(model_path, "stats.json")
        if os.path.exists(stats_path):
            logging.info(f"Loading stats from local file: {stats_path}")
            with open(stats_path, "r") as f:
                dataset_stats = json.load(f)
        else:
             logging.warning(f"No stats.json found at {stats_path}. Proceeding without explicit stats.")

    logging.info(f"Loading policy from {model_path}...")
    try:
        from lerobot.configs.policies import PreTrainedConfig
        policy_cfg = PreTrainedConfig.from_pretrained(model_path)
        policy_cfg.pretrained_path = model_path
        policy_cfg.device = args.device
        policy = make_policy(policy_cfg, ds_meta=ds_meta)
    except Exception as e:
        logging.warning(f"Could not load generic policy config: {e}. Falling back to ACTPolicy specific loading.")
        policy = ACTPolicy.from_pretrained(model_path)
        policy.to(get_safe_torch_device(args.device))

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    policy.eval()
    if hasattr(preprocessor, "reset"): preprocessor.reset()
    if hasattr(postprocessor, "reset"): postprocessor.reset()
    policy.reset()

    logging.info("Starting inference loop. Press Ctrl+C to stop.")
    
    try:
        while True:
            start_loop_t = time.perf_counter()

            # 1. Get Observation from Robot
            obs = await robot.get_observation()
            if not obs:
                await asyncio.sleep(0.001)
                continue
            
            # 2. Build observation frame with correct prefixes (e.g. observation.images.head_top)
            # This is what build_dataset_frame does: it maps robot keys to dataset keys.
            observation_frame = build_dataset_frame(dataset_features, obs, prefix=OBS_STR)

            # 3. Predict Action
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=args.task_description,
                robot_type=robot.robot_type,
            )

            # 4. Process Action
            act_processed_policy = make_robot_action(action_values, dataset_features)
            # robot_action_processor expects (action, raw_observation)
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))

            # 5. Send Action
            await robot.send_action(robot_action_to_send)

            # 6. Sleep to maintain FPS
            dt_s = time.perf_counter() - start_loop_t
            sleep_time = 1 / args.fps - dt_s
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        if 'obs' in locals() and obs:
            img_keys = [k for k in obs.keys() if k not in robot._state_ft_keys]
            logging.error(f"Robot keys: {list(obs.keys())}")
    finally:
        await robot.disconnect()
        logging.info("Disconnected.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
