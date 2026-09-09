"""Sincronização periódica de embeddings faciais com MySQL.

Thread daemon que a cada N segundos recarrega embeddings elegíveis do banco
e atualiza atomicamente o FaceIndex em memória. Opcionalmente executa enrollment
automático via FTP para Guests sem embedding.
"""

import os
import threading
import time
from collections.abc import Callable

from reconhecimento.database.repository import FaceDatabaseError, FaceRepository
from reconhecimento.recognition.matcher import InMemoryFaceIndex

SYNC_INTERVAL_DEFAULT = 30
ENROLLMENT_INTERVAL_DEFAULT = 30


class FaceSyncThread(threading.Thread):
    """Thread daemon que sincroniza embeddings do MySQL para o FaceIndex.

    A sincronização é atômica: substitui o snapshot inteiro do index.
    Se o banco falha, o index anterior permanece válido.
    Opcionalmente executa enrollment automático via FTP.
    """

    def __init__(
        self,
        repository: FaceRepository,
        index: InMemoryFaceIndex,
        interval_seconds: float | None = None,
        on_sync_complete: Callable[[int], None] | None = None,
        enrollment_enabled: bool | None = None,
        enrollment_interval_seconds: float | None = None,
    ) -> None:
        super().__init__(daemon=True, name="face-sync")
        self.repository = repository
        self.index = index
        self.interval_seconds = float(
            interval_seconds
            or os.getenv("FACE_SYNC_INTERVAL_SECONDS", str(SYNC_INTERVAL_DEFAULT))
        )
        self.on_sync_complete = on_sync_complete

        # Enrollment config — explicit param overrides env var
        if enrollment_enabled is not None:
            self._enrollment_enabled = bool(enrollment_enabled)
        else:
            self._enrollment_enabled = (
                os.getenv("FACE_ENROLLMENT_ENABLED", "false").lower()
                in {"1", "true", "yes"}
            )
        self._enrollment_interval = float(
            enrollment_interval_seconds
            or os.getenv("FACE_ENROLLMENT_INTERVAL_SECONDS", str(ENROLLMENT_INTERVAL_DEFAULT))
        )
        self._last_enrollment_time: float | None = None
        self._enrollment_pipeline = None

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

        # Estado para revalidation
        self._guest_names: dict[str, str] = {}
        self._photo_checksums: dict[str, str] = {}

        # Estatísticas
        self._last_sync_count = 0
        self._last_sync_time: float | None = None
        self._sync_error = False

    @property
    def last_sync_count(self) -> int:
        return self._last_sync_count

    @property
    def last_sync_time(self) -> float | None:
        return self._last_sync_time

    @property
    def sync_error(self) -> bool:
        return self._sync_error

    @property
    def enrollment_enabled(self) -> bool:
        return self._enrollment_enabled

    @property
    def last_enrollment_time(self) -> float | None:
        return self._last_enrollment_time

    def set_enrollment_pipeline(self, pipeline: object) -> None:
        """Define o pipeline de enrollment (opcional)."""
        self._enrollment_pipeline = pipeline

    @property
    def guest_names(self) -> dict[str, str]:
        """Cópia do mapeamento guest_id -> name para revalidation."""
        return dict(self._guest_names)

    @property
    def photo_checksums(self) -> dict[str, str]:
        """Cópia do mapeamento guest_id -> photo_checksum para revalidation."""
        return dict(self._photo_checksums)

    def stop(self) -> None:
        """Sinaliza parada e acorda a thread."""
        self._stop_event.set()
        self._wake_event.set()

    def force_sync(self) -> None:
        """Força sincronização imediata (acorda a thread)."""
        self._wake_event.set()

    def run(self) -> None:
        # Sincronização inicial imediata
        self._do_sync()

        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=self.interval_seconds)
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            self._do_sync()

    def _do_sync(self) -> None:
        """Executa uma sincronização completa."""
        # 1. Enrollment automático (se habilitado e intervalo atingido)
        self._maybe_run_enrollment()

        # 2. Carrega embeddings elegíveis
        try:
            eligible = self.repository.load_eligible_embeddings()
        except Exception as exc:
            print(f"[WARN] Falha na sincronização de embeddings: {exc}")
            self._sync_error = True
            return

        # Atualiza maps de revalidation
        new_names: dict[str, str] = {}
        new_checksums: dict[str, str] = {}
        references: dict[str, object] = {}

        for guest_id, (embedding, name) in eligible.items():
            references[guest_id] = embedding
            new_names[guest_id] = name

        # Obtém checksums para revalidation
        try:
            checksums = self.repository.load_checksums()
            new_checksums.update(checksums)
        except (FaceDatabaseError, Exception) as exc:
            print(f"[WARN] Falha ao carregar checksums: {exc}")

        # Atualiza index atômicamente
        self.index.replace(references)
        self._guest_names = new_names
        self._photo_checksums = new_checksums
        self._last_sync_count = len(references)
        self._last_sync_time = time.monotonic()
        self._sync_error = False

        if self.on_sync_complete:
            self.on_sync_complete(len(references))

        if references:
            print(f"[SYNC] {len(references)} embeddings carregados do MySQL")
        else:
            print("[SYNC] Nenhum embedding elegível encontrado")

    def _maybe_run_enrollment(self) -> None:
        """Executa enrollment se habilitado e intervalo atingido."""
        if not self._enrollment_enabled:
            return
        if self._enrollment_pipeline is None:
            return

        now = time.monotonic()
        if (
            self._last_enrollment_time is not None
            and (now - self._last_enrollment_time) < self._enrollment_interval
        ):
            return

        self._last_enrollment_time = now

        try:
            results = self._enrollment_pipeline.enroll_pending_guests()
            success_count = sum(1 for r in results if r.success)
            if success_count > 0:
                print(f"[ENROLL] {success_count} enrollment(s) concluído(s)")
        except Exception as exc:
            print(f"[WARN] Erro no enrollment automático: {exc}")
