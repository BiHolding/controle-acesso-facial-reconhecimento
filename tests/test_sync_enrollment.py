"""Testes unitários para FaceSyncThread com enrollment."""

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reconhecimento.recognition.matcher import InMemoryFaceIndex
from reconhecimento.sync.face_sync import FaceSyncThread


def _unit_embedding() -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


class FaceSyncThreadEnrollmentTest(unittest.TestCase):
    """Testes de integração do enrollment no FaceSyncThread."""

    def _make_sync(
        self,
        eligible: dict | None = None,
        interval: float = 0.1,
        enrollment_enabled: bool = False,
    ) -> tuple[FaceSyncThread, MagicMock, MagicMock, InMemoryFaceIndex]:
        mock_repo = MagicMock()
        mock_repo.model = "buffalo_l"
        mock_repo.dimension = 512
        mock_repo.normalization = "l2"

        if eligible is not None:
            mock_repo.load_eligible_embeddings.return_value = eligible
        else:
            mock_repo.load_eligible_embeddings.return_value = {}

        mock_repo.load_checksums.return_value = {}
        mock_repo.find_pending_enrollment.return_value = []

        mock_pipeline = MagicMock()
        mock_pipeline.enroll_pending_guests.return_value = []

        index = InMemoryFaceIndex(threshold=0.6)
        sync = FaceSyncThread(
            repository=mock_repo,
            index=index,
            interval_seconds=interval,
            enrollment_enabled=enrollment_enabled,
        )
        sync.set_enrollment_pipeline(mock_pipeline)

        return sync, mock_repo, mock_pipeline, index

    def test_enrollment_not_run_when_disabled(self) -> None:
        sync, mock_repo, mock_pipeline, index = self._make_sync(
            enrollment_enabled=False
        )

        sync.start()
        sync.join(timeout=2.0)
        sync.stop()

        mock_pipeline.enroll_pending_guests.assert_not_called()

    def test_enrollment_run_when_enabled(self) -> None:
        sync, mock_repo, mock_pipeline, index = self._make_sync(
            enrollment_enabled=True
        )

        sync.start()
        sync.join(timeout=2.0)
        sync.stop()

        mock_pipeline.enroll_pending_guests.assert_called()

    def test_enrollment_interval_respected(self) -> None:
        sync, mock_repo, mock_pipeline, index = self._make_sync(
            enrollment_enabled=True,
            interval=0.05,
        )
        sync._enrollment_interval = 10.0  # Long interval

        sync.start()
        sync.join(timeout=1.0)
        sync.stop()

        # Should only be called once (initial sync)
        self.assertEqual(mock_pipeline.enroll_pending_guests.call_count, 1)

    def test_enrollment_failure_does_not_break_sync(self) -> None:
        sync, mock_repo, mock_pipeline, index = self._make_sync(
            enrollment_enabled=True,
            eligible={"1": (_unit_embedding(), "João")},
        )
        mock_pipeline.enroll_pending_guests.side_effect = Exception("Enrollment error")

        sync.start()
        sync.join(timeout=2.0)
        sync.stop()

        # Sync should still work
        self.assertEqual(sync.last_sync_count, 1)
        match = index.match(_unit_embedding())
        self.assertIsNotNone(match)

    def test_enrollment_success_reloads_embeddings(self) -> None:
        embedding = _unit_embedding()
        eligible = {"1": (embedding, "João")}
        sync, mock_repo, mock_pipeline, index = self._make_sync(
            eligible=eligible,
            enrollment_enabled=True,
        )

        # Mock enrollment to return success
        from reconhecimento.enrollment.pipeline import EnrollmentResult

        mock_pipeline.enroll_pending_guests.return_value = [
            EnrollmentResult(guest_id="2", success=True, reason="OK")
        ]

        sync.start()
        sync.join(timeout=2.0)
        sync.stop()

        # Should have loaded embeddings
        self.assertEqual(sync.last_sync_count, 1)
        match = index.match(embedding)
        self.assertIsNotNone(match)

    def test_enrollment_enabled_property(self) -> None:
        sync, _, _, _ = self._make_sync(enrollment_enabled=True)
        self.assertTrue(sync.enrollment_enabled)

        sync2, _, _, _ = self._make_sync(enrollment_enabled=False)
        self.assertFalse(sync2.enrollment_enabled)

    def test_last_enrollment_time(self) -> None:
        sync, _, _, _ = self._make_sync(enrollment_enabled=True)

        self.assertIsNone(sync.last_enrollment_time)

        sync.start()
        sync.join(timeout=2.0)
        sync.stop()

        self.assertIsNotNone(sync.last_enrollment_time)


if __name__ == "__main__":
    unittest.main()
