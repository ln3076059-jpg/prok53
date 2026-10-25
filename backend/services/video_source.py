from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import urlparse

import cv2


class VideoSource(ABC):
    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def frame_count(self) -> int: ...

    @abstractmethod
    def read(self): ...

    @abstractmethod
    def close(self) -> None: ...


class OpenCVVideoSource(VideoSource):
    def __init__(self, source: str | int):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("video source cannot be opened")

    @property
    def fps(self) -> float:
        return float(self.capture.get(cv2.CAP_PROP_FPS) or 25.0)

    @property
    def frame_count(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    def read(self):
        try:
            ok, frame = self.capture.read()
        except (SystemError, MemoryError):
            import gc

            gc.collect()
            try:
                ok, frame = self.capture.read()
            except Exception:
                return False, None
        if not ok or frame is None:
            return False, None
        return True, frame

    def grab(self) -> bool:
        try:
            return bool(self.capture.grab())
        except Exception:
            return False

    def close(self) -> None:
        self.capture.release()


class FileVideoSource(OpenCVVideoSource):
    pass


class WebcamSource(OpenCVVideoSource):
    """Experimental source. Production reconnect/backpressure is not implemented."""

    def __init__(self, source: int):
        if isinstance(source, bool) or not isinstance(source, int) or source < 0:
            raise ValueError("webcam source must be a non-negative integer device index")
        super().__init__(source)


class RTSPSource(OpenCVVideoSource):
    """Experimental source. Production reconnect/backpressure is not implemented."""

    def __init__(self, source: str):
        parsed = urlparse(source)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("RTSP source must use rtsp:// or rtsps:// with a hostname")
        super().__init__(source)
