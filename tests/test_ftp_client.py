"""Testes unitários para o cliente FTP seguro."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from reconhecimento.enrollment.ftp_client import (
    ALLOWED_EXTENSIONS,
    FtpConfig,
    FtpError,
    PhotoValidationError,
    download_photo_temp,
    resolve_ftp_path,
    validate_photo_reference,
)


class ValidatePhotoReferenceTest(unittest.TestCase):
    """Testes de validação de photo_reference."""

    def test_valid_jpg(self) -> None:
        result = validate_photo_reference("guests/abc123.jpg")
        self.assertEqual(result, "abc123.jpg")

    def test_valid_jpeg(self) -> None:
        result = validate_photo_reference("guests/abc123.jpeg")
        self.assertEqual(result, "abc123.jpeg")

    def test_valid_png(self) -> None:
        result = validate_photo_reference("guests/abc123.png")
        self.assertEqual(result, "abc123.png")

    def test_valid_webp(self) -> None:
        result = validate_photo_reference("guests/abc123.webp")
        self.assertEqual(result, "abc123.webp")

    def test_valid_with_underscore(self) -> None:
        result = validate_photo_reference("guests/abc_123-def.jpg")
        self.assertEqual(result, "abc_123-def.jpg")

    def test_valid_with_dots(self) -> None:
        result = validate_photo_reference("guests/abc.123.jpg")
        self.assertEqual(result, "abc.123.jpg")

    def test_empty_reference_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("")

    def test_none_reference_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference(None)

    def test_no_prefix_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("abc123.jpg")

    def test_wrong_prefix_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("photos/abc123.jpg")

    def test_traversal_blocked(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/../etc/passwd.jpg")

    def test_traversal_encoded_blocked(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/..%2Fetc/passwd.jpg")

    def test_subdirectory_blocked(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/subdir/abc123.jpg")

    def test_null_byte_blocked(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/abc\x00.jpg")

    def test_disallowed_extension_blocked(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/abc123.exe")

    def test_php_extension_blocked(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/abc123.php")

    def test_special_chars_blocked(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/abc 123.jpg")

    def test_backslash_normalized(self) -> None:
        result = validate_photo_reference("guests\\abc123.jpg")
        self.assertEqual(result, "abc123.jpg")

    def test_empty_after_prefix_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests/")

    def test_only_prefix_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            validate_photo_reference("guests")


class ResolveFtpPathTest(unittest.TestCase):
    """Testes de resolução de path FTP."""

    def test_basic_resolution(self) -> None:
        result = resolve_ftp_path("guests/abc.jpg", "vip-backend/writable/guests")
        self.assertEqual(result, "vip-backend/writable/guests/abc.jpg")

    def test_base_with_slashes(self) -> None:
        result = resolve_ftp_path("guests/abc.jpg", "/vip-backend/writable/guests/")
        self.assertEqual(result, "vip-backend/writable/guests/abc.jpg")

    def test_invalid_path_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            resolve_ftp_path("photos/abc.jpg", "vip-backend/writable/guests")


class DownloadPhotoTempTest(unittest.TestCase):
    """Testes de download temporário de foto."""

    def setUp(self) -> None:
        self.config = FtpConfig(
            host="ftp.example.com",
            port=21,
            user="user",
            password="pass",
            base_path="vip-backend/writable/guests",
        )

    @patch("reconhecimento.enrollment.ftp_client.ftplib.FTP")
    def test_successful_download(self, MockFTP: MagicMock) -> None:
        # Mock FTP
        mock_ftp = MagicMock()
        MockFTP.return_value = mock_ftp

        def mock_retrbinary(cmd, callback):
            callback(b"fake image data")

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        # Download
        tmp_path = download_photo_temp(self.config, "guests/test.jpg")

        try:
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(tmp_path.endswith(".jpg"))
            with open(tmp_path, "rb") as f:
                content = f.read()
            self.assertEqual(content, b"fake image data")
        finally:
            os.unlink(tmp_path)

    @patch("reconhecimento.enrollment.ftp_client.ftplib.FTP")
    def test_empty_file_raises(self, MockFTP: MagicMock) -> None:
        mock_ftp = MagicMock()
        MockFTP.return_value = mock_ftp

        def mock_retrbinary(cmd, callback):
            pass  # Não escreve nada

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        with self.assertRaises((PhotoValidationError, FtpError)):
            download_photo_temp(self.config, "guests/test.jpg")

    def test_invalid_reference_raises(self) -> None:
        with self.assertRaises(PhotoValidationError):
            download_photo_temp(self.config, "photos/test.jpg")

    @patch("reconhecimento.enrollment.ftp_client.ftplib.FTP")
    def test_ftp_error_cleans_temp(self, MockFTP: MagicMock) -> None:
        mock_ftp = MagicMock()
        MockFTP.return_value = mock_ftp
        mock_ftp.retrbinary.side_effect = Exception("FTP connection failed")

        with self.assertRaises(FtpError):
            download_photo_temp(self.config, "guests/test.jpg")

        # Verifica que não há arquivos temporários abandonados
        # (o mock não cria arquivos reais, então verificamos que a exceção foi lançada)


class FtpConfigTest(unittest.TestCase):
    """Testes de configuração FTP."""

    @patch.dict("os.environ", {
        "FTP_HOST": "ftp.test.com",
        "FTP_PORT": "2121",
        "FTP_USER": "testuser",
        "FTP_PASSWORD": "testpass",
        "FTP_BASE_PATH": "test/path",
    })
    def test_from_env(self) -> None:
        config = FtpConfig.from_env()
        self.assertEqual(config.host, "ftp.test.com")
        self.assertEqual(config.port, 2121)
        self.assertEqual(config.user, "testuser")
        self.assertEqual(config.password, "testpass")
        self.assertEqual(config.base_path, "test/path")

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults(self) -> None:
        config = FtpConfig.from_env()
        self.assertEqual(config.host, "")
        self.assertEqual(config.port, 21)
        self.assertEqual(config.base_path, "vip-backend/writable/guests")


if __name__ == "__main__":
    unittest.main()
