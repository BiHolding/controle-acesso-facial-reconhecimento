"""Validação de imagens para enrollment facial.

Valida:
- Arquivo existe e não está vazio
- MIME/imagem válida
- Decodificável pelo OpenCV
- Exatamente um rosto detectado
"""

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from reconhecimento.recognition.detector import FaceDetector


@dataclass(frozen=True)
class ImageValidationResult:
    """Resultado da validação de imagem."""
    valid: bool
    face_count: int
    photo_checksum: str
    reason: str

    @classmethod
    def success(cls, face_count: int, photo_checksum: str) -> "ImageValidationResult":
        return cls(
            valid=True,
            face_count=face_count,
            photo_checksum=photo_checksum,
            reason="OK",
        )

    @classmethod
    def failure(cls, reason: str) -> "ImageValidationResult":
        return cls(
            valid=False,
            face_count=0,
            photo_checksum="",
            reason=reason,
        )


def validate_image(
    image_path: str,
    detector: FaceDetector,
) -> ImageValidationResult:
    """Valida imagem para enrollment.

    Verifica:
    1. Arquivo existe
    2. Tamanho > 0
    3. Decodificável pelo OpenCV
    4. Exatamente 1 rosto detectado

    Args:
        image_path: caminho para o arquivo de imagem
        detector: detector de faces (buffalo_l)

    Returns:
        ImageValidationResult com status e motivo
    """
    # Verifica existência
    try:
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
    except (OSError, IOError) as exc:
        return ImageValidationResult.failure(f"Arquivo inacessível: {exc}")

    # Verifica tamanho
    if len(raw_bytes) == 0:
        return ImageValidationResult.failure("Arquivo vazio")

    # Calcula checksum SHA-256
    photo_checksum = hashlib.sha256(raw_bytes).hexdigest()

    # Decodifica com OpenCV
    nparr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return ImageValidationResult.failure("Imagem inválida ou corrompida")

    # Verifica dimensões mínimas
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return ImageValidationResult.failure(
            f"Imagem muito pequena ({w}x{h}), mínimo 100x100"
        )

    # Detecta rostos
    try:
        faces = detector.detect(image)
    except Exception as exc:
        return ImageValidationResult.failure(f"Falha na detecção de faces: {exc}")

    face_count = len(faces)

    if face_count == 0:
        return ImageValidationResult.failure("Nenhum rosto detectado na imagem")

    if face_count > 1:
        return ImageValidationResult.failure(
            f"Múltiplos rostos detectados ({face_count}), esperado exatamente 1"
        )

    return ImageValidationResult.success(face_count, photo_checksum)
