import os
import unittest
from unittest.mock import patch

from reconhecimento.config import (
    access_direction_from_env,
    camera_index_from_env,
    device_indices_from_args,
    display_index_from_env,
    event_cooldowns_from_env,
    result_timeout_seconds_from_env,
)
from reconhecimento.display import (
    _first_name,
    _layout_metrics,
    _safe_display_index,
    _validated_display_index,
)


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

    @patch.dict(os.environ, {"DISPLAY_INDEX": "2"}, clear=True)
    def test_display_index_accepts_custom_value(self) -> None:
        self.assertEqual(display_index_from_env(), 2)


class CommandLineIndexConfigurationTest(unittest.TestCase):
    @patch.dict(os.environ, {"CAMERA_INDEX": "0", "DISPLAY_INDEX": "0"}, clear=True)
    def test_command_line_overrides_environment(self) -> None:
        self.assertEqual(device_indices_from_args(["--camera", "2", "--monitor", "1"]), (2, 1))

    @patch.dict(os.environ, {"CAMERA_INDEX": "3", "DISPLAY_INDEX": "2"}, clear=True)
    def test_short_options_override_environment(self) -> None:
        self.assertEqual(device_indices_from_args(["-c", "1", "-m", "0"]), (1, 0))

    @patch.dict(os.environ, {"CAMERA_INDEX": "3", "DISPLAY_INDEX": "2"}, clear=True)
    def test_environment_remains_the_default(self) -> None:
        self.assertEqual(device_indices_from_args([]), (3, 2))

    @patch.dict(os.environ, {}, clear=True)
    def test_negative_command_line_index_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            device_indices_from_args(["--camera", "-1"])


class AccessDirectionConfigurationTest(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_direction_defaults_to_entry(self):
        self.assertEqual("ENTRY", access_direction_from_env())

    @patch.dict(os.environ, {"ACCESS_DIRECTION": " exit "}, clear=True)
    def test_direction_is_normalized(self):
        self.assertEqual("EXIT", access_direction_from_env())

    @patch.dict(os.environ, {"ACCESS_DIRECTION": "side"}, clear=True)
    def test_invalid_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid ACCESS_DIRECTION"):
            access_direction_from_env()


class AccessTimeoutConfigurationTest(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_timeouts_default_to_one_second(self):
        self.assertEqual(1.0, result_timeout_seconds_from_env())
        self.assertEqual((1.0, 1.0), event_cooldowns_from_env())

    @patch.dict(
        os.environ,
        {
            "ACCESS_RESULT_TIMEOUT_SECONDS": "1.5",
            "ACCESS_EVENT_COOLDOWN_SECONDS": "2",
            "UNKNOWN_EVENT_COOLDOWN_SECONDS": "3",
        },
        clear=True,
    )
    def test_timeouts_accept_configured_values(self):
        self.assertEqual(1.5, result_timeout_seconds_from_env())
        self.assertEqual((2.0, 3.0), event_cooldowns_from_env())

    @patch.dict(os.environ, {"ACCESS_RESULT_TIMEOUT_SECONDS": "invalid"}, clear=True)
    def test_invalid_timeout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid ACCESS_RESULT_TIMEOUT_SECONDS"):
            result_timeout_seconds_from_env()

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

    def test_runtime_display_index_rejects_missing_monitor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Monitor 2 não está disponível"):
            _validated_display_index("2", 2)

    def test_runtime_display_index_accepts_external_monitor(self) -> None:
        self.assertEqual(_validated_display_index("1", 2), 1)

    def test_portrait_layout_uses_screen_proportions(self) -> None:
        margin, preview_width, preview_height = _layout_metrics(768, 1366)
        self.assertEqual(margin, 23)
        self.assertEqual(preview_width, 722)
        self.assertEqual(preview_height, 683)
        self.assertLessEqual(preview_width, 768 - 2 * margin)


if __name__ == "__main__":
    unittest.main()
