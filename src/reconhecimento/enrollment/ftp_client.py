"""Cliente FTP seguro para download de fotos de Guests.

Validação de path, download temporário e cleanup automático.
FTP usado apenas no ciclo de sync, nunca durante matching por frame.
"""

import ftplib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from dotenv import load_dotenv

load_dotenv()

# Extensões permitidas para fotos
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Tamanho máximo da foto (10MB)
MAX_PHOTO_SIZE = 10 * 1024 * 1024

# Timeout para operações FTP (segundos)
FTP_TIMEOUT = 30


@dataclass(frozen=True)
class FtpConfig:
    """Configuração FTP a partir do .env."""
    host: str
    port: int
    user: str
    password: str
    base_path: str

    @classmethod
    def from_env(cls) -> "FtpConfig":
        return cls(
            host=os.getenv("FTP_HOST", ""),
            port=int(os.getenv("FTP_PORT", "21")),
            user=os.getenv("FTP_USER", ""),
            password=os.getenv("FTP_PASSWORD", ""),
            base_path=os.getenv("FTP_BASE_PATH", "vip-backend/writable/guests"),
        )


class FtpError(RuntimeError):
    """Falha na operação FTP."""


class PhotoValidationError(RuntimeError):
    """Foto inválida ou não encontrada."""


def validate_photo_reference(photo_reference: str) -> str:
    """Valida e normaliza photo_reference, retornando o filename seguro.

   /photo_reference deve ter formato: guests/<filename seguro>
    Retorna apenas o filename (sem prefixo guests/).

    Raises:
        PhotoValidationError: se o path é inseguro ou inválido.
    """
    if not photo_reference or not isinstance(photo_reference, str):
        raise PhotoValidationError("photo_reference vazio ou inválido")

    # Normaliza separadores
    normalized = photo_reference.replace("\\", "/").strip("/")

    # Verifica prefixo
    if not normalized.startswith("guests/"):
        raise PhotoValidationError(
            f"photo_reference deve começar com 'guests/': {photo_reference}"
        )

    # Extrai o path relativo após guests/
    relative = normalized[len("guests/"):]
    if not relative:
        raise PhotoValidationError("photo_reference sem filename")

    # Verifica traversal
    if ".." in relative:
        raise PhotoValidationError(f"Path traversal bloqueado: {photo_reference}")

    # Verifica slashes (deve ser apenas filename)
    if "/" in relative:
        raise PhotoValidationError(
            f"Path contém subdiretórios inesperados: {photo_reference}"
        )

    # Verifica null bytes
    if "\x00" in relative:
        raise PhotoValidationError("Path contém null bytes")

    # Verifica extensão
    suffix = Path(relative).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise PhotoValidationError(
            f"Extensão não permitida '{suffix}': {photo_reference}"
        )

    # Verifica se o filename é seguro (sem caracteres especiais)
    if not all(c.isalnum() or c in "._-" for c in relative):
        raise PhotoValidationError(f"Filename contém caracteres inválidos: {relative}")

    return relative


def resolve_ftp_path(photo_reference: str, base_path: str) -> str:
    """Resolve o caminho completo no FTP.

    Mapeia guests/<arquivo> para base_path/<arquivo>.
    """
    filename = validate_photo_reference(photo_reference)
    # Normaliza base_path
    normalized_base = base_path.strip("/")
    return f"{normalized_base}/{filename}"


def download_photo_temp(
    ftp_config: FtpConfig,
    photo_reference: str,
) -> str:
    """Baixa foto do FTP para arquivo temporário.

    Retorna o caminho do arquivo temporário.
    O chamador é responsável por deletar o arquivo.

    Raises:
        FtpError: se o download falhar.
        PhotoValidationError: se o path é inseguro.
    """
    # Valida o path primeiro (fail fast)
    remote_path = resolve_ftp_path(photo_reference, ftp_config.base_path)

    # Cria arquivo temporário
    suffix = Path(remote_path).suffix
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix="face_enroll_",
    )
    tmp_path = tmp_file.name

    try:
        # Conecta ao FTP
        ftp = ftplib.FTP()
        ftp.connect(ftp_config.host, ftp_config.port, timeout=FTP_TIMEOUT)
        ftp.login(ftp_config.user, ftp_config.password)
        ftp.voidcmd("TYPE I")  # Modo binário

        # Download
        with open(tmp_path, "wb") as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)

        # Verifica tamanho
        file_size = os.path.getsize(tmp_path)
        if file_size == 0:
            os.unlink(tmp_path)
            raise PhotoValidationError(f"Foto vazia: {photo_reference}")
        if file_size > MAX_PHOTO_SIZE:
            os.unlink(tmp_path)
            raise PhotoValidationError(
                f"Foto excede tamanho máximo ({file_size} > {MAX_PHOTO_SIZE})"
            )

        # Desconecta
        try:
            ftp.quit()
        except Exception:
            pass

        return tmp_path

    except (PhotoValidationError, FtpError):
        # Cleanup em caso de erro
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    except Exception as exc:
        # Cleanup em caso de erro inesperado
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise FtpError(f"Falha no download FTP: {exc}") from exc
