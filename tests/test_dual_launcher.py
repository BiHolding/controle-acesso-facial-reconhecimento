import os
import unittest
from unittest.mock import patch

from reconhecimento.dual_launcher import (
    StationConfig,
    child_environment,
    station_configs_from_env,
    validate_displays,
)


class DualLauncherConfigurationTest(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_two_distinct_cameras_and_displays(self):
        first, second = station_configs_from_env()

        self.assertEqual(
            (0, 0, "ENTRY"),
            (first.camera_index, first.display_index, first.direction),
        )
        self.assertEqual(
            (1, 1, "EXIT"),
            (second.camera_index, second.display_index, second.direction),
        )
        self.assertEqual("ENTRADA_PRINCIPAL", first.access_point)
        self.assertEqual("SAIDA_PRINCIPAL", second.access_point)

    @patch.dict(os.environ, {"STATION_2_CAMERA_INDEX": "0"}, clear=True)
    def test_rejects_the_same_camera_for_both_stations(self):
        with self.assertRaisesRegex(ValueError, "mesma webcam"):
            station_configs_from_env()

    @patch.dict(os.environ, {"STATION_2_DISPLAY_INDEX": "0"}, clear=True)
    def test_rejects_the_same_display_for_both_stations(self):
        with self.assertRaisesRegex(ValueError, "mesmo monitor"):
            station_configs_from_env()

    @patch.dict(
        os.environ,
        {"STATION_1_DEVICE_ID": "7", "STATION_2_DEVICE_ID": "7"},
        clear=True,
    )
    def test_rejects_duplicate_nonzero_device_ids(self):
        with self.assertRaisesRegex(ValueError, "DEVICE_ID diferente"):
            station_configs_from_env()

    @patch.dict(os.environ, {"STATION_2_DIRECTION": "ENTRY"}, clear=True)
    def test_requires_one_entry_and_one_exit(self):
        with self.assertRaisesRegex(ValueError, "uma estação como ENTRY"):
            station_configs_from_env()

    @patch.dict(
        os.environ,
        {
            "STATION_1_DIRECTION": "EXIT",
            "STATION_1_ACCESS_POINT": "VIP_EXIT_01",
            "STATION_2_DIRECTION": "ENTRY",
            "STATION_2_ACCESS_POINT": "VIP_ENTRANCE_01",
        },
        clear=True,
    )
    def test_entry_and_exit_can_be_swapped_between_hardware(self):
        first, second = station_configs_from_env()

        self.assertEqual(("EXIT", "VIP_EXIT_01"), (first.direction, first.access_point))
        self.assertEqual(("ENTRY", "VIP_ENTRANCE_01"), (second.direction, second.access_point))

    def test_rejects_missing_second_display(self):
        stations = (
            StationConfig(1, 0, 0, 0, "ENTRY", "VIP_ENTRANCE_01"),
            StationConfig(2, 1, 1, 0, "EXIT", "VIP_EXIT_01"),
        )

        with self.assertRaisesRegex(RuntimeError, "Apenas um monitor"):
            validate_displays(stations, 1)

    @patch.dict(os.environ, {"KEEP_ME": "yes"}, clear=True)
    def test_builds_an_isolated_environment_for_each_station(self):
        environment = child_environment(
            StationConfig(2, 4, 3, 19, "EXIT", "VIP_EXIT_02")
        )

        self.assertEqual("4", environment["CAMERA_INDEX"])
        self.assertEqual("3", environment["DISPLAY_INDEX"])
        self.assertEqual("19", environment["DEVICE_ID"])
        self.assertEqual("2", environment["STATION_NUMBER"])
        self.assertEqual("EXIT", environment["ACCESS_DIRECTION"])
        self.assertEqual("VIP_EXIT_02", environment["ACCESS_POINT"])
        self.assertEqual("false", environment["FACE_ENROLLMENT_ENABLED"])
        self.assertEqual("yes", environment["KEEP_ME"])

    @patch.dict(os.environ, {"FACE_ENROLLMENT_ENABLED": "true"}, clear=True)
    def test_only_the_primary_station_keeps_enrollment_enabled(self):
        environment = child_environment(
            StationConfig(1, 0, 0, 11, "ENTRY", "VIP_ENTRANCE_01")
        )

        self.assertEqual("true", environment["FACE_ENROLLMENT_ENABLED"])

    @patch.dict(
        os.environ,
        {"STATION_1_DEVICE_KEY": "entry-key", "DEVICE_KEY": "fallback-key"},
        clear=True,
    )
    def test_station_receives_its_own_api_key(self):
        environment = child_environment(
            StationConfig(1, 0, 0, 11, "ENTRY", "ENTRADA_PRINCIPAL")
        )

        self.assertEqual("entry-key", environment["DEVICE_KEY"])


if __name__ == "__main__":
    unittest.main()
