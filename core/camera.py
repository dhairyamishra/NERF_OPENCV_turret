"""
Camera abstraction layer - supports OpenCV (PC) and PiCamera2 (RPi).
"""

import cv2
import numpy as np
import logging
import platform
import threading
import time

logger = logging.getLogger(__name__)


class Camera:
    """Thread-safe camera capture with support for PC webcams and RPi cameras."""

    def __init__(self, config: dict):
        self.config = config
        self.camera_index = config["system"]["camera_index"]
        self.resolution = tuple(config["system"]["resolution"])
        self.target_fps = config["system"]["fps"]
        self.platform = self._detect_platform(config["system"]["platform"])

        self._cap = None
        self._picam = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._fps = 0.0
        self._frame_count = 0

    def _detect_platform(self, configured: str) -> str:
        if configured != "auto":
            return configured
        if platform.machine().startswith("aarch64") or platform.machine().startswith("arm"):
            return "rpi"
        return "pc"

    def start(self):
        """Initialize camera and start capture thread."""
        if self.platform == "rpi":
            self._start_picamera()
        else:
            self._start_opencv()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera started on platform={self.platform}, resolution={self.resolution}")

    def _start_opencv(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.camera_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def _start_picamera(self):
        try:
            from picamera2 import Picamera2
            self._picam = Picamera2()
            cam_config = self._picam.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self._picam.configure(cam_config)
            self._picam.start()
        except ImportError:
            logger.warning("picamera2 not available, falling back to OpenCV")
            self.platform = "pc"
            self._start_opencv()

    def _capture_loop(self):
        fps_timer = time.time()
        fps_count = 0

        while self._running:
            frame = self._read_raw()
            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1

                fps_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    self._fps = fps_count / elapsed
                    fps_count = 0
                    fps_timer = time.time()
            else:
                time.sleep(0.001)

    def _read_raw(self) -> np.ndarray | None:
        if self.platform == "rpi" and self._picam is not None:
            frame = self._picam.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self._cap is not None:
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the latest frame (thread-safe). Returns the shared buffer directly
        for performance - caller must copy if mutation is needed."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def stop(self):
        """Release camera resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()
        logger.info("Camera stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
