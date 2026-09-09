"""Testes unitários para os métodos de enrollment do repositório."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from reconhecimento.database.repository import (
    EMBEDDING_BYTE_LENGTH,
    FaceDatabaseError,
    FaceRepository,
)


def _unit_embedding() -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


class FindPendingEnrollmentTest(unittest.TestCase):
    """Testes de find_pending_enrollment."""

    def _make_repo(self) -> FaceRepository:
        repo = FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
        )
        return repo

    def test_find_pending(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"guest_id": 1, "photo_reference": "guests/abc.jpg"},
            {"guest_id": 2, "photo_reference": "guests/def.png"},
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.find_pending_enrollment()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["guest_id"], "1")
        self.assertEqual(result[0]["photo_reference"], "guests/abc.jpg")

    def test_find_pending_empty(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.find_pending_enrollment()

        self.assertEqual(result, [])

    def test_find_pending_db_error(self) -> None:
        import pymysql

        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = pymysql.Error("DB error")
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        with self.assertRaises(FaceDatabaseError):
            repo.find_pending_enrollment()


class FindEmbeddingByGuestTest(unittest.TestCase):
    """Testes de find_embedding_by_guest."""

    def _make_repo(self) -> FaceRepository:
        return FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
        )

    def test_find_existing(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "photo_checksum": "abc123",
            "revision": 2,
        }
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.find_embedding_by_guest("1")

        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["photo_checksum"], "abc123")
        self.assertEqual(result["revision"], 2)

    def test_find_nonexistent(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.find_embedding_by_guest("999")

        self.assertIsNone(result)


class UpsertEmbeddingTest(unittest.TestCase):
    """Testes de upsert_embedding."""

    def _make_repo(self) -> FaceRepository:
        return FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
        )

    def test_insert_new_embedding(self) -> None:
        repo = self._make_repo()
        embedding = _unit_embedding()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock find_embedding_by_guest returns None (no existing)
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        # Should not raise
        repo.upsert_embedding("1", embedding, "abc123", revision=0)

        # Verify INSERT was called
        mock_cursor.execute.assert_called()
        call_args = mock_cursor.execute.call_args
        self.assertIn("INSERT", call_args[0][0])

    def test_update_existing_embedding(self) -> None:
        repo = self._make_repo()
        embedding = _unit_embedding()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock find_embedding_by_guest returns existing
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "photo_checksum": "old_checksum",
            "revision": 0,
        }
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        # Should not raise
        repo.upsert_embedding("1", embedding, "new_checksum", revision=1)

        # Verify UPDATE was called
        mock_cursor.execute.assert_called()
        call_args = mock_cursor.execute.call_args
        self.assertIn("UPDATE", call_args[0][0])

    def test_invalid_blob_size(self) -> None:
        repo = self._make_repo()
        embedding = np.zeros(100, dtype=np.float32)  # Wrong size

        with self.assertRaises(FaceDatabaseError):
            repo.upsert_embedding("1", embedding, "abc123")


class LoadChecksumsTest(unittest.TestCase):
    """Testes de load_checksums."""

    def _make_repo(self) -> FaceRepository:
        return FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
        )

    def test_load_checksums(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"guest_id": 1, "photo_checksum": "abc123"},
            {"guest_id": 2, "photo_checksum": "def456"},
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.load_checksums()

        self.assertEqual(len(result), 2)
        self.assertEqual(result["1"], "abc123")
        self.assertEqual(result["2"], "def456")

    def test_load_checksums_empty(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.load_checksums()

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
