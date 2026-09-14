"""Pipeline de enrollment automático via FTP.

Orquestra: download -> validação -> embedding -> persistência.
Executado apenas no ciclo de sync, nunca durante matching por frame.
"""

import os
import threading
import time
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv

from reconhecimento.bootstrap import ensure_onnxruntime_loaded

ensure_onnxruntime_loaded()

from reconhecimento.database.repository import FaceDatabaseError, FaceRepository
from reconhecimento.api.client import EnrollmentClient, RecognitionApiError
from reconhecimento.enrollment.ftp_client import (
    FtpConfig,
    FtpError,
    PhotoValidationError,
    download_photo_temp,
)
from reconhecimento.enrollment.image_validator import (
    ImageValidationResult,
    validate_image,
)
from reconhecimento.recognition.detector import FaceDetector
from reconhecimento.recognition.embedder import FaceEmbedder

load_dotenv()


@dataclass(frozen=True)
class EnrollmentResult:
    """Resultado de uma tentativa de enrollment."""
    guest_id: str
    success: bool
    reason: str
    photo_checksum: str = ""
    participant_type: str = "GUEST"


class EnrollmentPipeline:
    """Pipeline de enrollment automático para Guests sem embedding.

    Executa apenas no ciclo de sync. Não interfere com reconhecimento.
    """

    def __init__(
        self,
        repository: FaceRepository,
        detector: FaceDetector | None,
        embedder: FaceEmbedder | None,
        ftp_config: FtpConfig | None = None,
        enrollment_client: EnrollmentClient | None = None,
    ) -> None:
        self.repository = repository
        self.detector = detector
        self.embedder = embedder
        self.ftp_config = ftp_config or FtpConfig.from_env()
        self.enrollment_client = enrollment_client

    def enroll_pending_guests(self) -> list[EnrollmentResult]:
        """Enrolla todos os Guests pendentes de enrollment.

        Returns:
            Lista de resultados de enrollment.
        """
        # 1. Encontra guests pendentes
        try:
            pending = self.repository.find_pending_enrollment()
        except FaceDatabaseError as exc:
            print(f"[ENROLL] Falha ao buscar guests pendentes: {exc}")
            return []

        if not pending:
            return []

        if self.detector is None:
            self.detector = FaceDetector()
        if self.embedder is None:
            self.embedder = FaceEmbedder()

        print(f"[ENROLL] {len(pending)} guest(s) pendente(s) de enrollment")
        results = []

        for guest_info in pending:
            guest_id = guest_info.get("participant_id", guest_info.get("guest_id"))
            participant_type = guest_info.get("participant_type", "GUEST")
            photo_reference = guest_info["photo_reference"]

            try:
                result = self._enroll_single_guest(guest_id, photo_reference, participant_type)
                results.append(result)
            except Exception as exc:
                print(f"[ENROLL] Erro inesperado no guest {guest_id}: {exc}")
                results.append(EnrollmentResult(
                    guest_id=guest_id,
                    success=False,
                    reason=f"Erro inesperado: {exc}",
                ))

        # Resumo
        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count
        if success_count > 0:
            print(f"[ENROLL] {success_count} enrollment(s) bem-sucedido(s)")
        if fail_count > 0:
            print(f"[ENROLL] {fail_count} enrollment(s) falhouram")

        return results

    def _enroll_single_guest(
        self, guest_id: str, photo_reference: str, participant_type: str = "GUEST"
    ) -> EnrollmentResult:
        """Enrolla um único Guest.

        Fluxo completo: download -> validação -> embedding -> persistência.
        """
        tmp_path = None

        try:
            # 2. Download da foto
            tmp_path = download_photo_temp(self.ftp_config, photo_reference)
            print(f"[ENROLL] Guest {guest_id}: foto baixada")

            # 3. Validação da imagem
            validation = validate_image(tmp_path, self.detector)
            if not validation.valid:
                return EnrollmentResult(
                    guest_id=guest_id,
                    success=False,
                    reason=validation.reason,
                )
            print(f"[ENROLL] Guest {guest_id}: imagem válida ({validation.face_count} face)")

            # 4. Gera embedding
            embedding = self._generate_embedding(tmp_path)
            if embedding is None:
                return EnrollmentResult(
                    guest_id=guest_id,
                    success=False,
                    reason="Falha ao gerar embedding",
                )
            print(f"[ENROLL] Guest {guest_id}: embedding gerado (512D)")

            # 5. Publica pela API, que valida a fotografia e controla revisão/auditoria
            if self.enrollment_client is None:
                raise RecognitionApiError("Cliente de enrollment não configurado")
            if participant_type == "GUEST":
                self.enrollment_client.enroll_guest(guest_id, embedding, validation.photo_checksum)
            else:
                self.enrollment_client.enroll_participant(
                    participant_type.lower(), guest_id, embedding, validation.photo_checksum
                )
            print(f"[ENROLL] Guest {guest_id}: embedding publicado na API VIP")

            return EnrollmentResult(
                guest_id=guest_id,
                success=True,
                reason="Enrollment concluído",
                photo_checksum=validation.photo_checksum,
                participant_type=participant_type,
            )

        except FtpError as exc:
            return EnrollmentResult(
                guest_id=guest_id,
                success=False,
                reason=f"FTP: {exc}",
            )
        except PhotoValidationError as exc:
            return EnrollmentResult(
                guest_id=guest_id,
                success=False,
                reason=f"Foto: {exc}",
            )
        except FaceDatabaseError as exc:
            return EnrollmentResult(
                guest_id=guest_id,
                success=False,
                reason=f"Banco: {exc}",
            )
        except RecognitionApiError as exc:
            return EnrollmentResult(
                guest_id=guest_id,
                success=False,
                reason=f"API: {exc}",
            )
        finally:
            # 7. Cleanup do arquivo temporário
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _generate_embedding(self, image_path: str) -> np.ndarray | None:
        """Gera embedding a partir de uma imagem.

        Abre a imagem, detecta o rosto e gera o embedding.
        """
        try:
            import cv2

            if self.detector is None or self.embedder is None:
                return None

            # Lê a imagem
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ENROLL] Falha ao ler imagem: {image_path}")
                return None

            # Detecta faces
            faces = self.detector.detect(image)
            if len(faces) != 1:
                print(f"[ENROLL] Esperado 1 face, encontrado {len(faces)}")
                return None

            # Gera embedding
            embedding = self.embedder.generate(faces[0], image)

            # Valida embedding
            if embedding.shape != (512,):
                print(f"[ENROLL] Dimensão inválida: {embedding.shape}")
                return None
            if not np.all(np.isfinite(embedding)):
                print("[ENROLL] Embedding contém valores não finitos")
                return None
            norm = float(np.linalg.norm(embedding))
            if norm <= np.finfo(np.float32).eps:
                print("[ENROLL] Embedding com norma zero")
                return None

            return (embedding / norm).astype(np.float32)

        except Exception as exc:
            print(f"[ENROLL] Erro ao gerar embedding: {exc}")
            return None


class EnrollmentSyncThread(threading.Thread):
    """Executa enrollment em background sem interferir no atendimento da estação."""

    def __init__(self, pipeline: EnrollmentPipeline, interval_seconds: float = 30.0) -> None:
        super().__init__(daemon=True, name="enrollment-sync")
        self.pipeline = pipeline
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.pipeline.enroll_pending_guests()
            except Exception as exc:
                print(f"[WARN] Falha no ciclo de enrollment: {exc}")
            self._stop_event.wait(self.interval_seconds)
