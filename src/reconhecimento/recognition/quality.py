"""Avaliação leve e local da captura antes do reconhecimento facial.

As métricas servem apenas para orientar a pessoa no totem. Nenhum frame,
recorte facial ou resultado de qualidade é persistido.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceQualityResult:
    acceptable: bool
    reason: str
    message: str
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    face_width_ratio: float = 0.0

    @classmethod
    def ok(
        cls,
        *,
        brightness: float,
        contrast: float,
        sharpness: float,
        face_width_ratio: float,
    ) -> "FaceQualityResult":
        return cls(
            acceptable=True,
            reason="OK",
            message="Perfeito. Mantenha o rosto nessa posição por um instante.",
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            face_width_ratio=face_width_ratio,
        )


class FaceQualityAssessor:
    """Produz feedback objetivo sem criar um novo subsistema biométrico."""

    def __init__(
        self,
        *,
        minimum_brightness: float = 55.0,
        maximum_brightness: float = 215.0,
        minimum_contrast: float = 18.0,
        minimum_sharpness: float = 45.0,
        minimum_face_width_ratio: float = 0.20,
        center_tolerance_x: float = 0.24,
        center_tolerance_y: float = 0.25,
    ) -> None:
        self.minimum_brightness = minimum_brightness
        self.maximum_brightness = maximum_brightness
        self.minimum_contrast = minimum_contrast
        self.minimum_sharpness = minimum_sharpness
        self.minimum_face_width_ratio = minimum_face_width_ratio
        self.center_tolerance_x = center_tolerance_x
        self.center_tolerance_y = center_tolerance_y

    def assess(self, frame: np.ndarray, face: object) -> FaceQualityResult:
        if frame is None or frame.ndim < 2 or frame.size == 0:
            return self._failure("INVALID_FRAME", "Não foi possível analisar a câmera. Procure nossa equipe.")

        frame_height, frame_width = frame.shape[:2]
        bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
        if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
            return self._failure("INVALID_FACE", "Posicione seu rosto novamente no centro da tela.")

        x1, y1, x2, y2 = bbox[:4]
        x1 = int(max(0, min(frame_width - 1, x1)))
        y1 = int(max(0, min(frame_height - 1, y1)))
        x2 = int(max(x1 + 1, min(frame_width, x2)))
        y2 = int(max(y1 + 1, min(frame_height, y2)))

        face_width_ratio = (x2 - x1) / max(1, frame_width)
        if face_width_ratio < self.minimum_face_width_ratio:
            return self._failure(
                "FACE_TOO_FAR",
                "Aproxime-se um pouco da câmera.",
                face_width_ratio=face_width_ratio,
            )

        center_x = ((x1 + x2) / 2) / max(1, frame_width)
        center_y = ((y1 + y2) / 2) / max(1, frame_height)
        if abs(center_x - 0.5) > self.center_tolerance_x or abs(center_y - 0.5) > self.center_tolerance_y:
            return self._failure(
                "FACE_OFF_CENTER",
                "Centralize seu rosto no contorno da tela.",
                face_width_ratio=face_width_ratio,
            )

        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return self._failure("INVALID_FACE", "Posicione seu rosto novamente no centro da tela.")

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if face_region.ndim == 3 else face_region
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        metrics = {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "face_width_ratio": face_width_ratio,
        }
        if brightness < self.minimum_brightness:
            return self._failure("TOO_DARK", "Seu rosto está escuro. Procure uma luz de frente.", **metrics)
        if brightness > self.maximum_brightness:
            return self._failure("TOO_BRIGHT", "Há muita luz no rosto. Afaste-se da luz direta.", **metrics)
        if contrast < self.minimum_contrast:
            return self._failure("LOW_CONTRAST", "Melhore a iluminação do rosto para continuarmos.", **metrics)
        if sharpness < self.minimum_sharpness:
            return self._failure("BLURRY", "Fique imóvel por um instante e olhe para a câmera.", **metrics)

        return FaceQualityResult.ok(**metrics)

    @staticmethod
    def _failure(reason: str, message: str, **metrics: float) -> FaceQualityResult:
        return FaceQualityResult(False, reason, message, **metrics)
