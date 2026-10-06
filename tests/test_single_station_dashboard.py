import os
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from reconhecimento.operator_launcher import (
    operator_dashboard_enabled,
    start_operator_dashboard,
    stop_operator_dashboard,
)


class SingleStationDashboardTest(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_dashboard_is_enabled_by_default(self):
        self.assertTrue(operator_dashboard_enabled())

    @patch.dict(os.environ, {"OPERATOR_DASHBOARD_ENABLED": "false"}, clear=True)
    @patch("reconhecimento.operator_launcher.subprocess.Popen")
    def test_disabled_dashboard_is_not_started(self, popen):
        self.assertIsNone(start_operator_dashboard())
        popen.assert_not_called()

    @patch.dict(
        os.environ,
        {"OPERATOR_DASHBOARD_ENABLED": "true", "OPERATOR_DISPLAY_INDEX": "0"},
        clear=True,
    )
    @patch("reconhecimento.operator_launcher.subprocess.Popen")
    def test_dashboard_starts_as_a_separate_local_process(self, popen):
        process = MagicMock()
        popen.return_value = process

        self.assertIs(process, start_operator_dashboard())

        command = popen.call_args.args[0]
        self.assertEqual("-m", command[1])
        self.assertEqual("reconhecimento.operator_dashboard", command[2])

    def test_dashboard_is_terminated_with_the_station(self):
        process = MagicMock()
        process.poll.return_value = None

        stop_operator_dashboard(process)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_dashboard_is_killed_when_it_does_not_terminate(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("dashboard", 5), None]

        stop_operator_dashboard(process)

        process.kill.assert_called_once_with()
        self.assertEqual(2, process.wait.call_count)


if __name__ == "__main__":
    unittest.main()
