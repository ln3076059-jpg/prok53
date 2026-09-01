from __future__ import annotations

from abc import ABC, abstractmethod

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
            raise ValueError("video source cannot be opened")

    @property
    def fps(self) -> float:
        return float(self.capture.get(cv2.CAP_PROP_FPS) or 25.0)

    @property
    def frame_count(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    def read(self):
        return self.capture.read()

    def close(self) -> None:
        self.capture.release()


class FileVideoSource(OpenCVVideoSource):
    pass


class WebcamSource(OpenCVVideoSource):
    """Experimental source. Production reconnect/backpressure is not implemented."""


class RTSPSource(OpenCVVideoSource):
    """Experimental source. Production reconnect/backpressure is not implemented."""
