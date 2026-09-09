"""Testes unitários para o pipeline de enrollment."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reconhecimento.enrollment.pipeline import (
    EnrollmentPipeline,
    EnrollmentResult,
)


def _unit_embedding() -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


class EnrollmentPipelineTest(unittest.TestCase):
    """Testes do pipeline de enrollment."""

    def setUp(self) -> None:
        self.repository = MagicMock()
        self.detector = MagicMock()
        self.embedder = MagicMock()
        self.ftp_config = MagicMock()
        self.ftp_config.host = "ftp.test.com"
        self.ftp_config.port = 21
        self.ftp_config.user = "user"
        self.ftp_config.password = "pass"
        self.ftp_config.base_path = "test/path"

        self.pipeline = EnrollmentPipeline(
            repository=self.repository,
            detector=self.detector,
            embedder=self.embedder,
            ftp_config=self.ftp_config,
        )

    def test_no_pending_guests(self) -> None:
        self.repository.find_pending_enrollment.return_value = []

        results = self.pipeline.enroll_pending_guests()

        self.assertEqual(results, [])

    def test_enrollment_success(self) -> None:
        # Mock pending guests
        self.repository.find_pending_enrollment.return_value = [
            {"guest_id": "1", "photo_reference": "guests/test.jpg"}
        ]

        # Mock no existing embedding
        self.repository.find_embedding_by_guest.return_value = None

        # Mock FTP download
        with patch(
            "reconhecimento.enrollment.pipeline.download_photo_temp"
        ) as mock_download:
            # Cria imagem temporária válida
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
                import cv2
                cv2.imwrite(tmp_path, img)

            mock_download.return_value = tmp_path

            # Mock image validation
            with patch(
                "reconhecimento.enrollment.pipeline.validate_image"
            ) as mock_validate:
                from reconhecimento.enrollment.image_validator import (
                    ImageValidationResult,
                )

                mock_validate.return_value = ImageValidationResult.success(
                    1, "abc123"
                )

                # Mock embedding generation
                embedding = np.zeros(512, dtype=np.float32)
                embedding[0] = 1.0
                self.embedder.generate.return_value = embedding

                # Mock detector
                self.detector.detect.return_value = [MagicMock()]

                # Run enrollment
                results = self.pipeline.enroll_pending_guests()

                # Verify
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].success)
                self.assertEqual(results[0].guest_id, "1")
                self.repository.upsert_embedding.assert_called_once()

    def test_enrollment_skip_same_checksum(self) -> None:
        # Mock pending guests
        self.repository.find_pending_enrollment.return_value = [
            {"guest_id": "1", "photo_reference": "guests/test.jpg"}
        ]

        # Mock existing embedding with same checksum
        self.repository.find_embedding_by_guest.return_value = {
            "id": 1,
            "photo_checksum": "abc123",
            "revision": 0,
        }

        # Mock FTP download
        with patch(
            "reconhecimento.enrollment.pipeline.download_photo_temp"
        ) as mock_download:
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
                import cv2
                cv2.imwrite(tmp_path, img)

            mock_download.return_value = tmp_path

            # Mock image validation with same checksum
            with patch(
                "reconhecimento.enrollment.pipeline.validate_image"
            ) as mock_validate:
                from reconhecimento.enrollment.image_validator import (
                    ImageValidationResult,
                )

                mock_validate.return_value = ImageValidationResult.success(
                    1, "abc123"
                )

                # Run enrollment
                results = self.pipeline.enroll_pending_guests()

                # Verify - should skip, not call upsert
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].success)
                self.assertIn("idêntico", results[0].reason)
                self.repository.upsert_embedding.assert_not_called()

    def test_enrollment_ftp_failure(self) -> None:
        self.repository.find_pending_enrollment.return_value = [
            {"guest_id": "1", "photo_reference": "guests/test.jpg"}
        ]

        with patch(
            "reconhecimento.enrollment.pipeline.download_photo_temp"
        ) as mock_download:
            from reconhecimento.enrollment.ftp_client import FtpError

            mock_download.side_effect = FtpError("FTP connection failed")

            results = self.pipeline.enroll_pending_guests()

            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].success)
            self.assertIn("FTP", results[0].reason)

    def test_enrollment_invalid_image(self) -> None:
        self.repository.find_pending_enrollment.return_value = [
            {"guest_id": "1", "photo_reference": "guests/test.jpg"}
        ]

        with patch(
            "reconhecimento.enrollment.pipeline.download_photo_temp"
        ) as mock_download:
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
                import cv2
                cv2.imwrite(tmp_path, img)

            mock_download.return_value = tmp_path

            with patch(
                "reconhecimento.enrollment.pipeline.validate_image"
            ) as mock_validate:
                from reconhecimento.enrollment.image_validator import (
                    ImageValidationResult,
                )

                mock_validate.return_value = ImageValidationResult.failure(
                    "Nenhum rosto detectado"
                )

                results = self.pipeline.enroll_pending_guests()

                self.assertEqual(len(results), 1)
                self.assertFalse(results[0].success)
                self.assertIn("rosto", results[0].reason)

    def test_enrollment_db_failure(self) -> None:
        from reconhecimento.database.repository import FaceDatabaseError

        self.repository.find_pending_enrollment.return_value = [
            {"guest_id": "1", "photo_reference": "guests/test.jpg"}
        ]

        self.repository.find_embedding_by_guest.return_value = None

        with patch(
            "reconhecimento.enrollment.pipeline.download_photo_temp"
        ) as mock_download:
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
                import cv2
                cv2.imwrite(tmp_path, img)

            mock_download.return_value = tmp_path

            with patch(
                "reconhecimento.enrollment.pipeline.validate_image"
            ) as mock_validate:
                from reconhecimento.enrollment.image_validator import (
                    ImageValidationResult,
                )

                mock_validate.return_value = ImageValidationResult.success(
                    1, "abc123"
                )

                self.embedder.generate.return_value = _unit_embedding()
                self.detector.detect.return_value = [MagicMock()]

                self.repository.upsert_embedding.side_effect = FaceDatabaseError(
                    "DB error"
                )

                results = self.pipeline.enroll_pending_guests()

                self.assertEqual(len(results), 1)
                self.assertFalse(results[0].success)
                self.assertIn("Banco", results[0].reason)

    def test_enrollment_temp_cleanup(self) -> None:
        self.repository.find_pending_enrollment.return_value = [
            {"guest_id": "1", "photo_reference": "guests/test.jpg"}
        ]

        with patch(
            "reconhecimento.enrollment.pipeline.download_photo_temp"
        ) as mock_download:
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
                import cv2
                cv2.imwrite(tmp_path, img)

            mock_download.return_value = tmp_path

            with patch(
                "reconhecimento.enrollment.pipeline.validate_image"
            ) as mock_validate:
                from reconhecimento.enrollment.image_validator import (
                    ImageValidationResult,
                )

                mock_validate.return_value = ImageValidationResult.failure(
                    "Teste"
                )

                self.pipeline.enroll_pending_guests()

                # Verifica que o arquivo temporário foi removido
                self.assertFalse(os.path.exists(tmp_path))


class EnrollmentResultTest(unittest.TestCase):
    """Testes da dataclass EnrollmentResult."""

    def test_success(self) -> None:
        result = EnrollmentResult(
            guest_id="1", success=True, reason="OK", photo_checksum="abc"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.guest_id, "1")

    def test_failure(self) -> None:
        result = EnrollmentResult(
            guest_id="1", success=False, reason="Erro"
        )
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
