"""Testes unitários para FaceRepository com conexão MySQL mockada."""

import struct
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pymysql

from reconhecimento.database.repository import (
    EMBEDDING_BYTE_LENGTH,
    EMBEDDING_DIMENSION,
    FaceDatabaseError,
    FaceRepository,
)


def _make_embedding_blob(vector: np.ndarray) -> bytes:
    """Cria BLOB PHP pack('g*') a partir de numpy float32 array."""
    return vector.astype(np.float32).tobytes()


def _unit_embedding() -> np.ndarray:
    """Cria embedding unitário para testes."""
    embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


class FaceRepositoryBlobDecodeTest(unittest.TestCase):
    """Testes de decodificação de BLOB sem conexão real."""

    def test_decode_valid_blob(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        embedding = _unit_embedding()
        blob = _make_embedding_blob(embedding)

        result = repo._decode_blob(blob)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (EMBEDDING_DIMENSION,))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_decode_none_returns_none(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        self.assertIsNone(repo._decode_blob(None))

    def test_decode_wrong_length_returns_none(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        self.assertIsNone(repo._decode_blob(b"\x00" * 100))

    def test_decode_empty_returns_none(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        self.assertIsNone(repo._decode_blob(b""))

    def test_decode_normalizes_vector(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        embedding = np.ones(EMBEDDING_DIMENSION, dtype=np.float32) * 5.0
        blob = _make_embedding_blob(embedding)

        result = repo._decode_blob(blob)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_decode_nan_in_blob_returns_none(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        embedding[0] = float("nan")
        blob = _make_embedding_blob(embedding)

        result = repo._decode_blob(blob)
        self.assertIsNone(result)

    def test_decode_inf_in_blob_returns_none(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        embedding[0] = float("inf")
        blob = _make_embedding_blob(embedding)

        result = repo._decode_blob(blob)
        self.assertIsNone(result)

    def test_decode_zero_norm_returns_none(self) -> None:
        repo = FaceRepository.__new__(FaceRepository)
        embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        blob = _make_embedding_blob(embedding)

        result = repo._decode_blob(blob)
        self.assertIsNone(result)


class FaceRepositoryInitTest(unittest.TestCase):
    """Testes de inicialização do repositório."""

    @patch.dict("os.environ", {
        "DB_HOST": "testhost",
        "DB_PORT": "3307",
        "DB_NAME": "testdb",
        "DB_USER": "testuser",
        "DB_PASSWORD": "testpass",
        "DEVICE_ID": "42",
        "ACCESS_POINT": "test_door",
        "FACE_MODEL_VERSION": "v2",
    })
    def test_init_from_env(self) -> None:
        repo = FaceRepository()

        self.assertEqual(repo.host, "testhost")
        self.assertEqual(repo.port, 3307)
        self.assertEqual(repo.database, "testdb")
        self.assertEqual(repo.user, "testuser")
        self.assertEqual(repo.password, "testpass")
        self.assertEqual(repo.device_id, 42)
        self.assertEqual(repo.access_point, "test_door")
        self.assertEqual(repo.model_version, "v2")

    def test_init_defaults(self) -> None:
        repo = FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
        )
        self.assertEqual(repo.host, "h")
        self.assertEqual(repo.port, 3306)
        self.assertEqual(repo.database, "db")


class FaceRepositoryLoadTest(unittest.TestCase):
    """Testes de load_eligible_embeddings com MySQL mockado."""

    def _make_repo(self) -> FaceRepository:
        repo = FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
        )
        return repo

    def test_load_eligible_embeddings(self) -> None:
        repo = self._make_repo()
        embedding = _unit_embedding()
        blob = _make_embedding_blob(embedding)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "guest_id": 1,
                "embedding": blob,
                "photo_checksum": "abc123",
                "guest_name": "João Silva",
            },
            {
                "guest_id": 2,
                "embedding": blob,
                "photo_checksum": "def456",
                "guest_name": "Maria Santos",
            },
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.load_eligible_embeddings()

        self.assertEqual(len(result), 2)
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertEqual(result["1"][1], "João Silva")
        self.assertEqual(result["2"][1], "Maria Santos")
        np.testing.assert_array_almost_equal(result["1"][0], embedding / np.linalg.norm(embedding))

    def test_load_empty_result(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.load_eligible_embeddings()
        self.assertEqual(result, {})

    def test_load_db_error_raises(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = pymysql.Error("Connection lost")
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        with self.assertRaises(FaceDatabaseError):
            repo.load_eligible_embeddings()


class FaceRepositoryRevalidateTest(unittest.TestCase):
    """Testes de revalidate_guest com MySQL mockado."""

    def _make_repo(self) -> FaceRepository:
        repo = FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
        )
        return repo

    def test_revalidate_eligible_guest(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "name": "João",
            "guest_status": "completed",
            "client_status": "active",
        }
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        eligible, name = repo.revalidate_guest("1")

        self.assertTrue(eligible)
        self.assertEqual(name, "João")

    def test_revalidate_inactive_guest(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "name": "João",
            "guest_status": "pending_completion",
            "client_status": "active",
        }
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        eligible, name = repo.revalidate_guest("1")

        self.assertFalse(eligible)

    def test_revalidate_inactive_client(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "name": "João",
            "guest_status": "completed",
            "client_status": "inactive",
        }
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        eligible, name = repo.revalidate_guest("1")

        self.assertFalse(eligible)

    def test_revalidate_nonexistent_guest(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        eligible, name = repo.revalidate_guest("999")

        self.assertFalse(eligible)
        self.assertEqual(name, "")


class FaceRepositoryLogEventTest(unittest.TestCase):
    """Testes de log_access_event com MySQL mockado."""

    def _make_repo(self, device_id: int = 42) -> FaceRepository:
        repo = FaceRepository(
            host="h", port=3306, database="db",
            user="u", password="p",
            device_id=device_id,
            access_point="vip_room",
        )
        return repo

    def test_log_event_success(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.log_access_event(
            guest_id="1",
            recognized=True,
            allowed=True,
            similarity=0.85,
            reason="AUTHORIZED",
        )

        self.assertTrue(result.logged)
        mock_cursor.execute.assert_called_once()

    def test_log_event_device_id_zero_writes_local(self) -> None:
        repo = self._make_repo(device_id=0)

        import tempfile, os, json

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            with unittest.mock.patch(
                "reconhecimento.database.repository.LOCAL_ACCESS_LOG", tmp_path
            ):
                result = repo.log_access_event(
                    guest_id="1",
                    recognized=True,
                    allowed=True,
                    similarity=0.85,
                    reason="AUTHORIZED",
                )

            self.assertTrue(result.logged)
            with open(tmp_path, encoding="utf-8") as f:
                line = f.readline().strip()
            event = json.loads(line)
            self.assertEqual(event["guest_id"], "1")
            self.assertTrue(event["recognized"])
            self.assertTrue(event["allowed"])
            self.assertAlmostEqual(event["similarity"], 0.85)
            self.assertEqual(event["reason"], "AUTHORIZED")
            self.assertEqual(event["access_point"], "vip_room")
            self.assertNotIn("frame", event)
            self.assertNotIn("photo", event)
            self.assertNotIn("embedding", event)
        finally:
            os.unlink(tmp_path)

    def test_log_event_db_error_returns_false(self) -> None:
        repo = self._make_repo()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = pymysql.Error("DB error")
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connection = mock_conn

        result = repo.log_access_event(
            guest_id="1",
            recognized=True,
            allowed=True,
            similarity=0.85,
            reason="AUTHORIZED",
        )

        self.assertFalse(result.logged)


if __name__ == "__main__":
    unittest.main()
