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

"""
Provides the OpenCVCamera class for capturing frames from cameras using OpenCV.
"""

import logging
import math
import os
import platform
import time
from threading import Event, Thread
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

from numpy.typing import NDArray

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

# ... (rest of the file until the worker function)

logger = logging.getLogger(__name__)


def _camera_thread_worker(
    config: OpenCVCameraConfig, frame_queue: Queue, stop_event: Event, connection_status_queue: Queue
):
    """
    Worker function that runs in a separate thread to handle camera operations.
    """
    camera = None
    try:
        # Instantiate and connect the camera inside the worker thread
        camera = OpenCVCamera(config)
        camera._connect_sync(warmup=True)
        connection_status_queue.put(True)
    except Exception as e:
        logger.error(f"Error connecting camera in worker thread: {e}", exc_info=True)
        connection_status_queue.put(False)
        return

    while not stop_event.is_set():
        try:
            frame = camera.read()
            try:
                # Clear the queue of any old frame.
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                # Put the new frame, but don't block.
                frame_queue.put(frame, block=False)
            except Full:
                # This can happen in a race condition if the consumer is slow.
                # It's safe to just ignore and drop the frame.
                pass
        except DeviceNotConnectedError:
            break
        except Exception:
            logger.warning(f"Error reading frame in background thread for {camera}", exc_info=True)
            time.sleep(0.1)

    if camera is not None:
        camera.disconnect()


class OpenCVCamera(Camera):
    """
    Manages camera interactions using OpenCV for efficient frame recording.
    """

    def __init__(self, config: OpenCVCameraConfig):
        super().__init__(config)
        self.config = config
        self.index_or_path = config.index_or_path
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.videocapture: cv2.VideoCapture | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_queue: Queue | None = None
        self._is_connected_async = False
        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = get_cv2_backend()
        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        if self.thread is not None:
            return self.thread.is_alive() and self._is_connected_async
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        self._connect_sync(warmup)

    def _connect_sync(self, warmup: bool):
        if self.videocapture is not None and self.videocapture.isOpened():
            return
        cv2.setNumThreads(1)
        self.videocapture = cv2.VideoCapture(self.index_or_path, self.backend)
        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(f"Failed to open {self}. Run `lerobot-find-cameras opencv` to find available cameras.")
        self._configure_capture_settings()
        if warmup and self.warmup_s > 0:
            logger.info(f"Warming up camera {self} by reading and discarding 6 frames...")
            for i in range(6):
                try:
                    self.read()
                    logger.debug(f"Read warmup frame {i + 1}/6.")
                except Exception as e:
                    logger.warning(f"Failed to read warmup frame {i + 1}/6: {e}")
                    break
        logger.info(f"{self} connected synchronously.")

    def _configure_capture_settings(self) -> None:
        if not (isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()):
            raise DeviceNotConnectedError(f"Cannot configure settings for {self} as it is not connected.")
        if self.config.fourcc is not None:
            self._validate_fourcc()
        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            self._validate_width_and_height()
        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            self._validate_fps()

    def _validate_fps(self) -> None:
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        if self.fps is None:
            raise ValueError(f"{self} FPS is not set")
        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _validate_fourcc(self) -> None:
        fourcc_code = cv2.VideoWriter_fourcc(*self.config.fourcc)
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        success = self.videocapture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        actual_fourcc_code = self.videocapture.get(cv2.CAP_PROP_FOURCC)
        actual_fourcc_code_int = int(actual_fourcc_code)
        actual_fourcc = "".join([chr((actual_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])
        if not success or actual_fourcc != self.config.fourcc:
            logger.warning(
                f"{self} failed to set fourcc={self.config.fourcc} (actual={actual_fourcc}, success={success}). "
                f"Continuing with default format."
            )

    def _validate_width_and_height(self) -> None:
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")
        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))
        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not width_success or self.capture_width != actual_width:
            raise RuntimeError(f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=}).")
        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not height_success or self.capture_height != actual_height:
            raise RuntimeError(f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=}).")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        found_cameras_info = []
        if platform.system() == "Linux":
            targets_to_scan = [str(p) for p in sorted(Path("/dev").glob("video*"))]
        else:
            targets_to_scan = list(range(MAX_OPENCV_INDEX))
        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_fourcc_code_int = int(camera.get(cv2.CAP_PROP_FOURCC))
                default_fourcc = "".join([chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])
                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV", "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "fourcc": default_fourcc, "width": default_width, "height": default_height, "fps": default_fps
                    },
                }
                found_cameras_info.append(camera_info)
                camera.release()
        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if self.thread is not None and self.thread.is_alive():
            raise NotImplementedError("Use `async_read` when background thread is running.")
        if not self.is_connected or self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} is not connected for synchronous reading.")
        start_time = time.perf_counter()
        ret, frame = self.videocapture.read()
        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")
        processed_frame = self._postprocess_image(frame, color_mode)
        logger.debug(f"{self} read took: {(time.perf_counter() - start_time) * 1e3:.1f}ms")
        return processed_frame

    def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:
        requested_color_mode = self.color_mode if color_mode is None else color_mode
        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}.")
        h, w, c = image.shape
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(f"{self} frame width={w} or height={h} does not match configured width={self.capture_width} or height={self.capture_height}.")
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")
        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)
        return processed_image

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None
        self.frame_queue = Queue(maxsize=1)
        self.stop_event = Event()
        connection_status_queue = Queue(maxsize=1)
        self.thread = Thread(
            target=_camera_thread_worker,
            args=(self.config, self.frame_queue, self.stop_event, connection_status_queue),
            name=f"{self}_read_thread",
        )
        self.thread.daemon = True
        self.thread.start()
        try:
            is_success = connection_status_queue.get(timeout=10.0)
            if not is_success:
                raise ConnectionError(f"Camera worker thread for {self} failed to connect.")
            self._is_connected_async = True
            logger.info(f"{self} background thread started and connected.")
        except Empty:
            self._stop_read_thread()
            raise TimeoutError(f"Timed out waiting for camera worker thread for {self} to connect.")

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"Thread for {self} did not terminate gracefully.")
        self.thread = None
        self.stop_event = None
        self.frame_queue = None
        self._is_connected_async = False

    def async_read(self, timeout_ms: float = 2000) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()
        if self.frame_queue is None:
            raise RuntimeError("Internal error: Frame queue not initialized for async reading.")
        try:
            frame = self.frame_queue.get(timeout=timeout_ms / 1000.0)
            return frame
        except Empty:
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. Read thread alive: {thread_alive}.")

    def disconnect(self) -> None:
        if self.thread is not None:
            self._stop_read_thread()
            logger.info(f"{self} background thread disconnected.")
        elif self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None
            logger.info(f"{self} disconnected.")
        else:
            raise DeviceNotConnectedError(f"{self} not connected.")

