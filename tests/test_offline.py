import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from reconhecimento.api.client import ApiUnavailableError, RecognitionResult
from reconhecimento.offline import EncryptedFaceCache, HybridRecognitionService, OfflineAccessStore
from reconhecimento.recognition.matcher import InMemoryFaceIndex


def unit_embedding(axis: int = 0) -> np.ndarray:
    value = np.zeros(512, dtype=np.float32)
    value[axis] = 1.0
    return value


class EncryptedFaceCacheTest(unittest.TestCase):
    def test_round_trip_without_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = EncryptedFaceCache(
                Path(directory) / "cache.bin",
                protect=lambda value: b"protected:" + value,
                unprotect=lambda value: value.removeprefix(b"protected:"),
            )
            cache.save({"42": (unit_embedding(), "Convidado VIP")})

            loaded = cache.load()

            self.assertEqual("Convidado VIP", loaded["42"][1])
            np.testing.assert_allclose(unit_embedding(), loaded["42"][0])

    def test_corrupt_cache_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.bin"
            path.write_bytes(b"invalid")
            cache = EncryptedFaceCache(path, unprotect=lambda value: value)
            self.assertEqual({}, cache.load())


class OfflineAccessStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = OfflineAccessStore(Path(self.directory.name) / "state.sqlite3", 2)

    def test_entry_duplicate_exit_and_fifo_queue(self) -> None:
        first = self.store.record_offline("1", "ENTRADA_PRINCIPAL", "ENTRY")
        duplicate = self.store.record_offline("1", "ENTRADA_PRINCIPAL", "ENTRY")
        exit_decision = self.store.record_offline("1", "SAIDA_PRINCIPAL", "EXIT")

        self.assertTrue(first.allowed)
        self.assertEqual("DUPLICATE_ENTRY", duplicate.reason)
        self.assertTrue(exit_decision.allowed)
        self.assertEqual(2, self.store.pending_count())
        self.assertEqual("ENTRY", self.store.next_event()["direction"])

    def test_capacity_is_enforced_offline(self) -> None:
        self.assertTrue(self.store.record_offline("1", "IN", "ENTRY").allowed)
        self.assertTrue(self.store.record_offline("2", "IN", "ENTRY").allowed)
        denied = self.store.record_offline("3", "IN", "ENTRY")
        self.assertFalse(denied.allowed)
        self.assertEqual("CAPACITY_FULL", denied.reason)


class HybridRecognitionServiceTest(unittest.TestCase):
    def test_authorizes_cached_guest_when_api_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OfflineAccessStore(Path(directory) / "state.sqlite3", 400)
            index = InMemoryFaceIndex(0.6)
            index.replace({"42": unit_embedding()})
            client = MagicMock()
            client.recognize.side_effect = ApiUnavailableError("offline")
            service = HybridRecognitionService(
                client, index, lambda: {"42": "Convidado VIP"}, store,
                "ENTRY", "ENTRADA_PRINCIPAL",
            )

            result = service.recognize(unit_embedding())

            self.assertTrue(result.allowed)
            self.assertEqual("OFFLINE_AUTHORIZED", result.reason)
            self.assertEqual("Convidado VIP", result.name)
            self.assertEqual(1, store.pending_count())

    def test_online_result_reconciles_local_presence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OfflineAccessStore(Path(directory) / "state.sqlite3", 400)
            index = InMemoryFaceIndex(0.6)
            index.replace({"42": unit_embedding()})
            client = MagicMock()
            client.recognize.return_value = RecognitionResult(
                True, True, "AUTHORIZED", guest_id="42", participant_type="guest",
                participant_id="42", name="VIP", similarity=1.0, direction="ENTRY",
            )
            service = HybridRecognitionService(
                client, index, lambda: {"42": "VIP"}, store,
                "ENTRY", "ENTRADA_PRINCIPAL",
            )

            self.assertTrue(service.recognize(unit_embedding()).allowed)
            duplicate = store.record_offline("42", "ENTRADA_PRINCIPAL", "ENTRY")
            self.assertEqual("DUPLICATE_ENTRY", duplicate.reason)

    def test_pending_queue_keeps_later_decisions_offline_to_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OfflineAccessStore(Path(directory) / "state.sqlite3", 400)
            store.record_offline("10", "ENTRADA_PRINCIPAL", "ENTRY")
            index = InMemoryFaceIndex(0.6)
            index.replace({"42": unit_embedding()})
            client = MagicMock()
            service = HybridRecognitionService(
                client, index, lambda: {"42": "VIP"}, store,
                "ENTRY", "ENTRADA_PRINCIPAL",
            )

            result = service.recognize(unit_embedding())

            self.assertTrue(result.allowed)
            client.recognize.assert_not_called()
            self.assertEqual(2, store.pending_count())

    def test_authorizes_cached_client_offline_without_identity_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OfflineAccessStore(Path(directory) / "state.sqlite3", 400)
            index = InMemoryFaceIndex(0.6)
            index.replace({"CLIENT:42": unit_embedding()})
            client = MagicMock()
            client.recognize.side_effect = ApiUnavailableError("offline")
            service = HybridRecognitionService(client, index, lambda: {"CLIENT:42": "Representante VIP"}, store, "ENTRY", "ENTRADA_PRINCIPAL")

            result = service.recognize(unit_embedding())

            self.assertTrue(result.allowed)
            self.assertEqual("client", result.participant_type)
            self.assertEqual("42", result.user_id)


if __name__ == "__main__":
    unittest.main()
