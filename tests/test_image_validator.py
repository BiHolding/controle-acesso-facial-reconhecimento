"""Testes unitários para validação de imagens."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reconhecimento.enrollment.image_validator import (
    ImageValidationResult,
    validate_image,
)


class ValidateImageTest(unittest.TestCase):
    """Testes de validação de imagem."""

    def setUp(self) -> None:
        self.detector = MagicMock()

    def test_nonexistent_file(self) -> None:
        result = validate_image("/nonexistent/image.jpg", self.detector)
        self.assertFalse(result.valid)
        self.assertIn("inacessível", result.reason)

    def test_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            result = validate_image(tmp_path, self.detector)
            self.assertFalse(result.valid)
            self.assertIn("vazio", result.reason)
        finally:
            os.unlink(tmp_path)

    def test_invalid_image(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"not an image")
            tmp_path = f.name

        try:
            result = validate_image(tmp_path, self.detector)
            self.assertFalse(result.valid)
            self.assertIn("inválida", result.reason)
        finally:
            os.unlink(tmp_path)

    def test_valid_image_with_one_face(self) -> None:
        # Cria uma imagem válida (branca)
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Salva como JPEG
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            import cv2
            cv2.imwrite(tmp_path, img)

            # Mock detector retorna 1 face
            mock_face = MagicMock()
            self.detector.detect.return_value = [mock_face]

            result = validate_image(tmp_path, self.detector)

            self.assertTrue(result.valid)
            self.assertEqual(result.face_count, 1)
            self.assertEqual(len(result.photo_checksum), 64)  # SHA-256
        finally:
            os.unlink(tmp_path)

    def test_image_with_no_faces(self) -> None:
        # Cria uma imagem válida
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            import cv2
            cv2.imwrite(tmp_path, img)

            # Mock detector retorna 0 faces
            self.detector.detect.return_value = []

            result = validate_image(tmp_path, self.detector)

            self.assertFalse(result.valid)
            self.assertIn("Nenhum rosto", result.reason)
        finally:
            os.unlink(tmp_path)

    def test_image_with_multiple_faces(self) -> None:
        # Cria uma imagem válida
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            import cv2
            cv2.imwrite(tmp_path, img)

            # Mock detector retorna 2 faces
            mock_face1 = MagicMock()
            mock_face2 = MagicMock()
            self.detector.detect.return_value = [mock_face1, mock_face2]

            result = validate_image(tmp_path, self.detector)

            self.assertFalse(result.valid)
            self.assertIn("Múltiplos rostos", result.reason)
            self.assertIn("2", result.reason)
        finally:
            os.unlink(tmp_path)

    def test_small_image_rejected(self) -> None:
        # Cria imagem muito pequena
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            import cv2
            cv2.imwrite(tmp_path, img)

            result = validate_image(tmp_path, self.detector)

            self.assertFalse(result.valid)
            self.assertIn("muito pequena", result.reason)
        finally:
            os.unlink(tmp_path)

    def test_detector_exception(self) -> None:
        # Cria uma imagem válida
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            import cv2
            cv2.imwrite(tmp_path, img)

            # Mock detector lança exceção
            self.detector.detect.side_effect = Exception("Detector error")

            result = validate_image(tmp_path, self.detector)

            self.assertFalse(result.valid)
            self.assertIn("detecção", result.reason)
        finally:
            os.unlink(tmp_path)

    def test_checksum_is_consistent(self) -> None:
        # Cria imagem válida
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            import cv2
            cv2.imwrite(tmp_path, img)

            self.detector.detect.return_value = [MagicMock()]

            result1 = validate_image(tmp_path, self.detector)
            result2 = validate_image(tmp_path, self.detector)

            self.assertEqual(result1.photo_checksum, result2.photo_checksum)
        finally:
            os.unlink(tmp_path)


class ImageValidationResultTest(unittest.TestCase):
    """Testes da dataclass ImageValidationResult."""

    def test_success(self) -> None:
        result = ImageValidationResult.success(1, "abc123")
        self.assertTrue(result.valid)
        self.assertEqual(result.face_count, 1)
        self.assertEqual(result.photo_checksum, "abc123")
        self.assertEqual(result.reason, "OK")

    def test_failure(self) -> None:
        result = ImageValidationResult.failure("erro")
        self.assertFalse(result.valid)
        self.assertEqual(result.face_count, 0)
        self.assertEqual(result.photo_checksum, "")
        self.assertEqual(result.reason, "erro")


if __name__ == "__main__":
    unittest.main()
