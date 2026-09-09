import os
import unittest
from unittest.mock import patch

from reconhecimento.config import camera_index_from_env
from reconhecimento.display import _first_name, _layout_metrics, _safe_display_index


class CameraIndexConfigurationTest(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_camera_index_defaults_to_zero(self) -> None:
        self.assertEqual(camera_index_from_env(), 0)

    @patch.dict(os.environ, {"CAMERA_INDEX": "1"}, clear=True)
    def test_camera_index_accepts_custom_value(self) -> None:
        self.assertEqual(camera_index_from_env(), 1)

    @patch.dict(os.environ, {"CAMERA_INDEX": "abc"}, clear=True)
    def test_camera_index_rejects_non_numeric_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid CAMERA_INDEX: abc"):
            camera_index_from_env()

    @patch.dict(os.environ, {"CAMERA_INDEX": "-1"}, clear=True)
    def test_camera_index_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid CAMERA_INDEX: -1"):
            camera_index_from_env()


class DisplayHelpersTest(unittest.TestCase):
    def test_first_name_is_safe(self) -> None:
        self.assertEqual(_first_name("  Alexandre Santana "), "Alexandre")
        self.assertEqual(_first_name(None), "")
        self.assertEqual(_first_name("  "), "")

    def test_display_index_falls_back_to_primary(self) -> None:
        self.assertEqual(_safe_display_index("invalid", 2), 0)
        self.assertEqual(_safe_display_index("2", 2), 0)
        self.assertEqual(_safe_display_index("-1", 2), 0)

    def test_display_index_accepts_existing_screen(self) -> None:
        self.assertEqual(_safe_display_index("1", 2), 1)

    def test_portrait_layout_uses_screen_proportions(self) -> None:
        margin, preview_width, preview_height = _layout_metrics(768, 1366)
        self.assertEqual(margin, 23)
        self.assertEqual(preview_width, 722)
        self.assertEqual(preview_height, 683)
        self.assertLessEqual(preview_width, 768 - 2 * margin)


if __name__ == "__main__":
    unittest.main()
