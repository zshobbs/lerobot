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

from dataclasses import dataclass
from .config import TeleoperatorConfig
from .so100_leader.so100_leader import SO100Leader, SO100LeaderConfig
from .teleoperator import Teleoperator


@TeleoperatorConfig.register_subclass("uni_so100_leader")
@dataclass
class UniSO100LeaderConfig(TeleoperatorConfig):
    port: str



class UniSO100Leader(Teleoperator):
    """
    A wrapper for the SO100Leader teleoperator that prefixes the action keys with "left_".
    This is for use with single-arm robots that expect this prefix, like the LeKiwiUni.
    """
    config_class = UniSO100LeaderConfig
    name = "uni_so100_leader"

    def __init__(self, config: UniSO100LeaderConfig):
        super().__init__(config)
        self.arm = SO100Leader(SO100LeaderConfig(port=config.port, id=config.id))

    def connect(self, calibrate: bool = True):
        self.arm.connect(calibrate)

    def disconnect(self):
        self.arm.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected

    def get_action(self) -> dict[str, float]:
        action = self.arm.get_action()
        # Prefix the keys with "left_"
        action = {f"left_{k}": v for k, v in action.items()}
        return action

    @property
    def action_features(self) -> dict:
        return self.arm.action_features

    @property
    def feedback_features(self) -> dict:
        return self.arm.feedback_features

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated

    def calibrate(self) -> None:
        self.arm.calibrate()

    def configure(self) -> None:
        self.arm.configure()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        self.arm.send_feedback(feedback)
