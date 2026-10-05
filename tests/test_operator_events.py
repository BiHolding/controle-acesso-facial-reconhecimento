import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from reconhecimento.operator_events import OperatorEvent, publish_operator_result


class OperatorEventTest(unittest.TestCase):
    def test_round_trip_preserves_the_operator_decision(self):
        event = OperatorEvent(
            station=1,
            direction="ENTRY",
            state="authorized",
            name="Convidada Teste",
            reason="AUTHORIZED",
            occurred_at="2026-10-05T12:00:00+00:00",
        )

        self.assertEqual(event, OperatorEvent.from_bytes(event.to_bytes()))

    def test_rejects_an_event_from_an_unknown_station(self):
        payload = (
            b'{"station":9,"direction":"ENTRY","state":"authorized",'
            b'"name":"Teste","reason":"AUTHORIZED",'
            b'"occurred_at":"2026-10-05T12:00:00+00:00"}'
        )

        with self.assertRaisesRegex(ValueError, "Estação"):
            OperatorEvent.from_bytes(payload)

    @patch.dict(
        os.environ,
        {"STATION_NUMBER": "2", "OPERATOR_EVENT_PORT": "38000"},
        clear=True,
    )
    @patch("reconhecimento.operator_events.socket.socket")
    def test_publishes_only_to_the_local_notebook(self, socket_factory):
        channel = MagicMock()
        socket_factory.return_value.__enter__.return_value = channel
        result = SimpleNamespace(
            state="denied",
            name="Pessoa Teste",
            reason="DUPLICATE_EXIT",
            direction="EXIT",
        )

        publish_operator_result(result, "EXIT")

        payload, address = channel.sendto.call_args.args
        event = OperatorEvent.from_bytes(payload)
        self.assertEqual(("127.0.0.1", 38000), address)
        self.assertEqual(2, event.station)
        self.assertEqual("Pessoa Teste", event.name)
        self.assertEqual("denied", event.state)

    @patch("reconhecimento.operator_events.socket.socket")
    def test_ignores_non_terminal_camera_guidance(self, socket_factory):
        result = SimpleNamespace(
            state="guidance", name=None, reason="TOO_DARK", direction="ENTRY"
        )

        publish_operator_result(result, "ENTRY")

        socket_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
