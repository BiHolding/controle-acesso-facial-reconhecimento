"""Testes unitários para FaceSyncThread com MySQL mockado."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reconhecimento.recognition.matcher import InMemoryFaceIndex
from reconhecimento.sync.face_sync import FaceSyncThread


def _unit_embedding() -> np.ndarray:
    """Cria embedding unitário para testes."""
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


class FaceSyncThreadTest(unittest.TestCase):
    """Testes de sincronização periódica."""

    def _make_sync(
        self,
        eligible: dict | None = None,
        checksums: dict | None = None,
        interval: float = 0.1,
    ) -> tuple[FaceSyncThread, MagicMock, InMemoryFaceIndex]:
        mock_repo = MagicMock()
        mock_repo.model = "buffalo_l"
        mock_repo.dimension = 512
        mock_repo.normalization = "l2"

        if eligible is not None:
            mock_repo.load_eligible_embeddings.return_value = eligible
        else:
            mock_repo.load_eligible_embeddings.return_value = {}

        index = InMemoryFaceIndex(threshold=0.6)
        sync = FaceSyncThread(
            repository=mock_repo,
            index=index,
            interval_seconds=interval,
        )
        return sync, mock_repo, index

    def test_initial_sync_populates_index(self) -> None:
        embedding = _unit_embedding()
        eligible = {"1": (embedding, "João")}
        sync, mock_repo, index = self._make_sync(eligible=eligible)

        sync.start()
        sync.join(timeout=2.0)

        self.assertEqual(sync.last_sync_count, 1)
        self.assertFalse(sync.sync_error)
        match = index.match(embedding)
        self.assertIsNotNone(match)
        self.assertEqual(match.guest_id, "1")

    def test_sync_updates_names(self) -> None:
        embedding = _unit_embedding()
        eligible = {"1": (embedding, "João")}
        sync, mock_repo, index = self._make_sync(eligible=eligible)

        sync.start()
        sync.join(timeout=2.0)

        self.assertIn("1", sync.guest_names)
        self.assertEqual(sync.guest_names["1"], "João")

    def test_sync_replaces_snapshot(self) -> None:
        embedding = _unit_embedding()
        eligible = {"1": (embedding, "João")}
        sync, mock_repo, index = self._make_sync(eligible=eligible)

        sync.start()
        sync.join(timeout=2.0)

        # Simula mudança no banco
        embedding2 = np.zeros(512, dtype=np.float32)
        embedding2[1] = 1.0
        mock_repo.load_eligible_embeddings.return_value = {"2": (embedding2, "Maria")}
        sync.force_sync()
        sync.join(timeout=2.0)

        # Guest antigo não deve mais existir
        self.assertIsNone(index.match(embedding))
        # Guest novo deve existir
        match = index.match(embedding2)
        self.assertIsNotNone(match)
        self.assertEqual(match.guest_id, "2")

    def test_sync_error_sets_flag(self) -> None:
        mock_repo = MagicMock()
        mock_repo.model = "buffalo_l"
        mock_repo.dimension = 512
        mock_repo.normalization = "l2"
        mock_repo.load_eligible_embeddings.side_effect = Exception("DB down")

        index = InMemoryFaceIndex(threshold=0.6)
        sync = FaceSyncThread(
            repository=mock_repo,
            index=index,
            interval_seconds=10.0,
        )

        sync.start()
        sync.join(timeout=2.0)
        sync.stop()

        self.assertTrue(sync.sync_error)

    def test_sync_error_preserves_old_index(self) -> None:
        embedding = _unit_embedding()
        eligible = {"1": (embedding, "João")}
        sync, mock_repo, index = self._make_sync(eligible=eligible)

        sync.start()
        sync.join(timeout=2.0)

        # Simula falha no banco
        mock_repo.load_eligible_embeddings.side_effect = Exception("DB down")
        sync.force_sync()
        sync.join(timeout=2.0)

        # Index antigo deve permanecer válido
        match = index.match(embedding)
        self.assertIsNotNone(match)

    def test_stop_terminates_thread(self) -> None:
        sync, mock_repo, index = self._make_sync(interval=10.0)

        sync.start()
        time.sleep(0.05)
        sync.stop()
        sync.join(timeout=2.0)

        self.assertFalse(sync.is_alive())

    def test_on_sync_complete_callback(self) -> None:
        callback = MagicMock()
        embedding = _unit_embedding()
        eligible = {"1": (embedding, "João")}
        sync, mock_repo, index = self._make_sync(eligible=eligible)
        sync.on_sync_complete = callback

        sync.start()
        sync.join(timeout=2.0)
        sync.stop()

        self.assertTrue(callback.called)
        callback.assert_any_call(1)

    def test_empty_eligible_populates_empty_index(self) -> None:
        sync, mock_repo, index = self._make_sync(eligible={})

        sync.start()
        sync.join(timeout=2.0)

        self.assertEqual(sync.last_sync_count, 0)
        self.assertIsNone(index.match(_unit_embedding()))


if __name__ == "__main__":
    unittest.main()
