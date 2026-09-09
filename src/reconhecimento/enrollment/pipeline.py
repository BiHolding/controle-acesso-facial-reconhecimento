"""Pipeline de enrollment automático via FTP.

Orquestra: download -> validação -> embedding -> persistência.
Executado apenas no ciclo de sync, nunca durante matching por frame.
"""

import os
import tempfile
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv

from reconhecimento.database.repository import FaceDatabaseError, FaceRepository
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


class EnrollmentPipeline:
    """Pipeline de enrollment automático para Guests sem embedding.

    Executa apenas no ciclo de sync. Não interfere com reconhecimento.
    """

    def __init__(
        self,
        repository: FaceRepository,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        ftp_config: FtpConfig | None = None,
    ) -> None:
        self.repository = repository
        self.detector = detector
        self.embedder = embedder
        self.ftp_config = ftp_config or FtpConfig.from_env()

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

        print(f"[ENROLL] {len(pending)} guest(s) pendente(s) de enrollment")
        results = []

        for guest_info in pending:
            guest_id = guest_info["guest_id"]
            photo_reference = guest_info["photo_reference"]

            try:
                result = self._enroll_single_guest(guest_id, photo_reference)
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
        self, guest_id: str, photo_reference: str
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

            # 4. Verifica se checksum já existe (skip se igual)
            existing = self.repository.find_embedding_by_guest(guest_id)
            if existing is not None:
                existing_checksum = existing.get("photo_checksum", "")
                if existing_checksum == validation.photo_checksum:
                    print(f"[ENROLL] Guest {guest_id}: checksum idêntico, skip")
                    return EnrollmentResult(
                        guest_id=guest_id,
                        success=True,
                        reason="Checksum idêntico, enrollment já existe",
                        photo_checksum=validation.photo_checksum,
                    )

            # 5. Gera embedding
            embedding = self._generate_embedding(tmp_path)
            if embedding is None:
                return EnrollmentResult(
                    guest_id=guest_id,
                    success=False,
                    reason="Falha ao gerar embedding",
                )
            print(f"[ENROLL] Guest {guest_id}: embedding gerado (512D)")

            # 6. Persiste no banco
            revision = 0
            if existing is not None:
                revision = existing.get("revision", 0) + 1

            self.repository.upsert_embedding(
                guest_id=guest_id,
                embedding=embedding,
                photo_checksum=validation.photo_checksum,
                revision=revision,
            )
            print(f"[ENROLL] Guest {guest_id}: embedding persistido (rev={revision})")

            return EnrollmentResult(
                guest_id=guest_id,
                success=True,
                reason="Enrollment concluído",
                photo_checksum=validation.photo_checksum,
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

            return embedding

        except Exception as exc:
            print(f"[ENROLL] Erro ao gerar embedding: {exc}")
            return None
