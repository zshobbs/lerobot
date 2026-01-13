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

import base64
import json
import logging
import time
import sys

import cv2
import zmq

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.zmq_observation_socket_hq = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket_hq.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket_hq.bind(f"tcp://*:{config.port_zmq_observations_hq}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_observation_socket_hq.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
 

def main():
    logging.info("Configuring LeKiwi")
    robot_config = LeKiwiConfig()
    robot_config.id = "AlohaMiniRobot"
    robot = LeKiwi(robot_config)


    logging.info("Connecting AlohaMini")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = LeKiwiHostConfig()
    host = LeKiwiHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                #print(f"Received action: {data}")   # debug 
                _action_sent = robot.send_action(data)
                
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.exception("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            
            robot.lift.update()
            last_observation = robot.get_observation()

            # Encode ndarrays to base64 strings with different qualities
            obs_hq = last_observation.copy()
            obs_lq = last_observation.copy()

            for cam_key, _ in robot.cameras.items():
                image = last_observation[cam_key]

                # HQ Encoding
                ret_hq, buffer_hq = cv2.imencode(
                    ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret_hq:
                    obs_hq[cam_key] = base64.b64encode(buffer_hq).decode("utf-8")
                else:
                    obs_hq[cam_key] = ""

                # LQ Encoding (Higher compression for remote)
                ret_lq, buffer_lq = cv2.imencode(
                    ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                )
                if ret_lq:
                    obs_lq[cam_key] = base64.b64encode(buffer_lq).decode("utf-8")
                else:
                    obs_lq[cam_key] = ""

            # Send the observation to the remote agent (LQ)
            try:
                host.zmq_observation_socket.send_string(json.dumps(obs_lq), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping LQ observation, no client connected")

            # Send the observation to the local agent (HQ)
            try:
                host.zmq_observation_socket_hq.send_string(json.dumps(obs_hq), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping HQ observation, no client connected")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down AlohaMini Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished AlohaMini cleanly")


if __name__ == "__main__":
    main()
