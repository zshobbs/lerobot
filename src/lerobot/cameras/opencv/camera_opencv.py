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
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore  # TODO: add type stubs for OpenCV

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

# NOTE(Steven): The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


def _camera_process_worker(
    config: OpenCVCameraConfig, frame_queue: Queue, stop_event: Event, connection_status_queue: Queue
):
    """
    Worker function that runs in a separate process to handle camera operations.

    This function initializes the camera, continuously reads frames, and puts them
    into a queue. It's designed to run independently to not block the main process.

    Args:
        config: The camera configuration object.
        frame_queue: A multiprocessing.Queue to send captured frames to the main process.
        stop_event: A multiprocessing.Event to signal when the worker should stop.
        connection_status_queue: A queue to report connection success or failure.
    """
    camera = None
    try:
        # Instantiate and connect the camera inside the worker process
        camera = OpenCVCamera(config)
        camera.connect(warmup=True)
        connection_status_queue.put(True)  # Signal successful connection
    except Exception as e:
        logger.error(f"Error connecting camera in worker process: {e}")
        connection_status_queue.put(False)  # Signal connection failure
        return

    while not stop_event.is_set():
        try:
            frame = camera.read()
            # Non-blocking put: empty the queue first to always have the latest frame
            try:
                frame_queue.get_nowait()
            except Empty:
                pass  # Queue was empty, which is fine
            frame_queue.put(frame, block=False)
        except DeviceNotConnectedError:
            break
        except Exception as e:
            logger.warning(f"Error reading frame in background process for {camera}: {e}")
            # Avoid busy-looping on continuous errors
            time.sleep(0.1)

    if camera is not None:
        camera.disconnect()


class OpenCVCamera(Camera):
    """
    Manages camera interactions using OpenCV for efficient frame recording.

    This class provides a high-level interface to connect to, configure, and read
    frames from cameras compatible with OpenCV's VideoCapture. It supports both
    synchronous and asynchronous frame reading. For async reading, it spawns a
    separate process for each camera to avoid GIL limitations and improve stability.

    An OpenCVCamera instance requires a camera index (e.g., 0) or a device path
    (e.g., '/dev/video0' on Linux).

    Example:
        ```python
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.configuration_opencv import OpenCVCameraConfig

        # Asynchronous reading (recommended)
        config = OpenCVCameraConfig(index_or_path=0)
        camera = OpenCVCamera(config)
        camera.connect() # Starts background process
        async_image = camera.async_read()
        camera.disconnect()

        # Synchronous reading (main process only, no background process)
        config_sync = OpenCVCameraConfig(index_or_path=0)
        camera_sync = OpenCVCamera(config_sync)
        # Manually call connect_sync to create videocapture in main process
        camera_sync.connect(warmup=False)
        sync_image = camera_sync.read()
        camera_sync.disconnect()
        ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        """
        Initializes the OpenCVCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        # For synchronous reading
        self.videocapture: cv2.VideoCapture | None = None

        # For asynchronous reading
        self.process: Process | None = None
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
        """Checks if the camera is currently connected and opened."""
        # If a background process is running, connection status is determined by it
        if self.process is not None:
            return self.process.is_alive() and self._is_connected_async
        # Otherwise, check the synchronous VideoCapture object
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the OpenCV camera.

        For asynchronous reading (the default behavior of `async_read`), this method
        starts a background process to handle camera I/O.

        For synchronous reading, this method initializes the OpenCV VideoCapture object
        in the main process.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If the specified camera index/path is not found or fails to open.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # This method is now a router. If `async_read` is intended, it will
        # start the process. If `read` is intended, it will connect synchronously.
        # The user's call to `async_read` later will trigger process creation if needed.
        # For now, we connect synchronously for warmup and synchronous reads.
        self._connect_sync(warmup)

    def _connect_sync(self, warmup: bool):
        """Connects the camera in the current process for synchronous reading."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        cv2.setNumThreads(1)
        self.videocapture = cv2.VideoCapture(self.index_or_path, self.backend)

        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(
                f"Failed to open {self}. Run `lerobot-find-cameras opencv` to find available cameras."
            )

        self._configure_capture_settings()

        if warmup and self.warmup_s > 0:
            logger.info(f"Warming up camera {self} by reading and discarding 6 frames...")
            for i in range(6):
                try:
                    self.read()
                    logger.debug(f"Read warmup frame {i + 1}/6.")
                except Exception as e:
                    logger.warning(f"Failed to read warmup frame {i + 1}/6: {e}")
                    # If a warmup frame fails, the camera is likely not going to work.
                    # We can break early instead of waiting for more failures.
                    break

        logger.info(f"{self} connected synchronously.")

    def _configure_capture_settings(self) -> None:
        """
        Applies the specified FOURCC, FPS, width, and height settings to the connected camera.
        """
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
        """Validates and sets the camera's frames per second (FPS)."""
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        if self.fps is None:
            raise ValueError(f"{self} FPS is not set")

        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _validate_fourcc(self) -> None:
        """Validates and sets the camera's FOURCC code."""
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
        """Validates and sets the camera's frame capture width and height."""
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")

        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not width_success or self.capture_width != actual_width:
            raise RuntimeError(
                f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=})."
            )

        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not height_success or self.capture_height != actual_height:
            raise RuntimeError(
                f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=})."
            )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # This static method remains unchanged
        found_cameras_info = []
        targets_to_scan: list[str | int]
        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = [int(i) for i in range(MAX_OPENCV_INDEX)]

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                # ... (rest of the find_cameras logic is unchanged)
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)
                default_fourcc_code = camera.get(cv2.CAP_PROP_FOURCC)
                default_fourcc_code_int = int(default_fourcc_code)
                default_fourcc = "".join([chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])
                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "fourcc": default_fourcc,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }
                found_cameras_info.append(camera_info)
                camera.release()
        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Reads a single frame synchronously from the camera.
        This is a blocking call and only works if the camera was not connected for async reading.
        """
        if self.process is not None and self.process.is_alive():
            raise NotImplementedError(
                "Synchronous `read` is not supported when the background process is running. "
                "Use `async_read` instead."
            )

        if not self.is_connected or self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} is not connected for synchronous reading.")

        start_time = time.perf_counter()
        ret, frame = self.videocapture.read()

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        processed_frame = self._postprocess_image(frame, color_mode)
        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")
        return processed_frame

    def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:
        # This helper method remains unchanged
        requested_color_mode = self.color_mode if color_mode is None else color_mode
        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )
        h, w, c = image.shape
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")
        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)
        return processed_image

    def _start_read_process(self) -> None:
        """Starts the background read process if it's not running."""
        if self.process is not None and self.process.is_alive():
            return

        # If there's a sync connection, disconnect it first.
        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

        self.frame_queue = Queue(maxsize=1)
        self.stop_event = Event()
        connection_status_queue = Queue(maxsize=1)

        self.process = Process(
            target=_camera_process_worker,
            args=(self.config, self.frame_queue, self.stop_event, connection_status_queue),
            name=f"{self}_read_process",
        )
        self.process.daemon = True
        self.process.start()

        # Wait for connection status from the worker
        try:
            is_success = connection_status_queue.get(timeout=10.0)
            if not is_success:
                raise ConnectionError(f"Camera worker process for {self} failed to connect.")
            self._is_connected_async = True
            logger.info(f"{self} background process started and connected.")
        except Empty:
            self._stop_read_process()
            raise TimeoutError(f"Timed out waiting for camera worker process for {self} to connect.")

    def _stop_read_process(self) -> None:
        """Signals the background read process to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.process is not None and self.process.is_alive():
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                logger.warning(f"Process for {self} did not terminate gracefully. Terminating.")
                self.process.terminate()

        self.process = None
        self.stop_event = None
        self.frame_queue = None
        self._is_connected_async = False

    def async_read(self, timeout_ms: float = 2000) -> NDArray[Any]:
        """
        Reads the latest available frame asynchronously from the background process.
        """
        if self.process is None or not self.process.is_alive():
            self._start_read_process()

        if self.frame_queue is None:
            raise RuntimeError("Internal error: Frame queue not initialized for async reading.")

        try:
            frame = self.frame_queue.get(timeout=timeout_ms / 1000.0)
            return frame
        except Empty:
            process_alive = self.process is not None and self.process.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read process alive: {process_alive}."
            )

    def disconnect(self) -> None:
        """
        Disconnects from the camera and cleans up resources.
        Stops the background read process (if running) or releases the sync VideoCapture.
        """
        if self.process is not None:
            self._stop_read_process()
            logger.info(f"{self} background process disconnected.")
        elif self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None
            logger.info(f"{self} disconnected.")
        else:
            raise DeviceNotConnectedError(f"{self} not connected.")
