import unittest
from unittest.mock import MagicMock, patch

import cv2

from reconhecimento.camera.capture import Camera


class CameraCaptureTest(unittest.TestCase):
    @patch("reconhecimento.camera.capture.cv2.VideoCapture")
    @patch("reconhecimento.camera.capture.os.name", "nt")
    def test_windows_uses_directshow(self, video_capture: MagicMock) -> None:
        capture = video_capture.return_value
        capture.isOpened.return_value = True
        capture.get.return_value = 640.0

        camera = Camera(camera_index=1)

        video_capture.assert_called_once_with(1, cv2.CAP_DSHOW)
        camera.release()

    @patch("reconhecimento.camera.capture.cv2.VideoCapture")
    @patch("reconhecimento.camera.capture.os.name", "posix")
    def test_other_platforms_use_default_backend(self, video_capture: MagicMock) -> None:
        capture = video_capture.return_value
        capture.isOpened.return_value = True
        capture.get.return_value = 640.0

        camera = Camera(camera_index=1)

        video_capture.assert_called_once_with(1)
        camera.release()


if __name__ == "__main__":
    unittest.main()
