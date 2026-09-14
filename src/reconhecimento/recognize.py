"""Loop principal de reconhecimento facial integrado à API VIP.

Fluxo:
  1. Captura frame da câmera
  2. A cada N frames, envia para o worker em background
  3. Worker detecta faces, gera embeddings
  4. Confirma estabilidade em cinco amostras e calcula um embedding médio
  5. A API VIP identifica e registra entrada/saída atomicamente
  6. Badge verde/vermelho é exibido na janela por 3 segundos
"""
import os
import sys
import threading
import time
from collections.abc import Callable

from reconhecimento.bootstrap import ensure_onnxruntime_loaded

ensure_onnxruntime_loaded()

import cv2
from dotenv import load_dotenv

from reconhecimento.config import access_direction_from_env, camera_index_from_env
from reconhecimento.recognition.confirmation import RecognitionConfirmation
from reconhecimento.recognition.detector import FaceDetector
from reconhecimento.recognition.embedder import FaceEmbedder
from reconhecimento.recognition.guard import AccessEventGuard
from reconhecimento.recognition.matcher import FaceMatch, InMemoryFaceIndex
from reconhecimento.database.repository import FaceDatabaseError, FaceRepository

# ── Constantes visuais ────────────────────────────────────────────────────────
GREEN = (50, 205, 50)
RED   = (50, 50, 220)
AMBER = (0, 180, 230)
FONT  = cv2.FONT_HERSHEY_SIMPLEX

# ── Constantes de comportamento ───────────────────────────────────────────────
SHOW_SECS        = 3.0   # segundos que o badge fica visível
PROCESS_EVERY_N  = 3     # analisa 1 a cada N frames (alivia CPU)
UNKNOWN_REQUIRED = 10    # frames sem reconhecimento para mostrar "Desconhecido"
DEBUG = os.getenv("FACE_DEBUG", "").lower() in {"1", "true", "yes"}


# ── Desenho do badge ──────────────────────────────────────────────────────────

def _draw_badge(frame, state: str, name: str | None = None) -> None:
    """Desenha uma pill colorida na parte inferior do frame."""
    h, w    = frame.shape[:2]
    if state == "authorized":
        color = GREEN
        label = f"ACESSO AUTORIZADO  {name or ''}".rstrip()
    elif state == "denied":
        color = RED
        label = "ACESSO NAO AUTORIZADO"
    elif state == "unknown":
        color = RED
        label = "PESSOA NAO IDENTIFICADA"
    else:
        color = AMBER
        label = "NAO FOI POSSIVEL VALIDAR"

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
    pad_x, pad_y = 28, 14
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    bx = (w - bw) // 2
    by = h - bh - 24
    r  = bh // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx + r, by), (bx + bw - r, by + bh), color, -1)
    cv2.ellipse(overlay, (bx + r,      by + bh // 2), (r, bh // 2), 0,  90, 270, color, -1)
    cv2.ellipse(overlay, (bx + bw - r, by + bh // 2), (r, bh // 2), 0, -90,  90, color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, label, (bx + pad_x, by + pad_y + th - 2),
                FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


# ── Resultado de reconhecimento local ─────────────────────────────────────────

class LocalRecognitionResult:
    """Resultado de reconhecimento local (substitui RecognitionResult da API)."""

    def __init__(
        self,
        recognized: bool,
        allowed: bool,
        reason: str,
        guest_id: str | None = None,
        name: str | None = None,
        similarity: float | None = None,
    ):
        self.recognized = recognized
        self.allowed = allowed
        self.reason = reason
        self.guest_id = guest_id
        self.name = name
        self.similarity = similarity


# ── Função de reconhecimento local ────────────────────────────────────────────

def _make_local_recognize(
    index: InMemoryFaceIndex,
    repository: FaceRepository,
    sync_thread: "FaceSyncThread | None" = None,
) -> Callable[[object], LocalRecognitionResult]:
    """Cria uma função de reconhecimento local com revalidation MySQL.

    O matching é local (NumPy cosine), mas a decisão de allowed requer
    revalidação no MySQL: guest completed + client active + embedding ativo.
    """
    from reconhecimento.sync.face_sync import FaceSyncThread

    def local_recognize(embedding: object) -> LocalRecognitionResult:
        match = index.match(embedding)
        if match is None:
            return LocalRecognitionResult(
                recognized=False,
                allowed=False,
                reason="UNKNOWN_FACE",
            )

        # Revalidação: guest + client + embedding no MySQL
        try:
            eligible, guest_name = repository.revalidate_guest(match.guest_id)
        except FaceDatabaseError:
            return LocalRecognitionResult(
                recognized=True,
                allowed=False,
                guest_id=match.guest_id,
                similarity=match.similarity,
                reason="DB_UNAVAILABLE",
            )

        if not eligible:
            return LocalRecognitionResult(
                recognized=True,
                allowed=False,
                guest_id=match.guest_id,
                similarity=match.similarity,
                reason="NOT_AUTHORIZED",
            )

        # Revalidação de embedding (photo_checksum consistente)
        if sync_thread is not None:
            checksums = sync_thread.photo_checksums
            current_checksum = checksums.get(match.guest_id, "")
            if current_checksum:
                try:
                    embedding_valid = repository.revalidate_embedding(
                        match.guest_id, current_checksum
                    )
                except FaceDatabaseError:
                    return LocalRecognitionResult(
                        recognized=True,
                        allowed=False,
                        guest_id=match.guest_id,
                        name=guest_name,
                        similarity=match.similarity,
                        reason="DB_UNAVAILABLE",
                    )
                if not embedding_valid:
                    return LocalRecognitionResult(
                        recognized=True,
                        allowed=False,
                        guest_id=match.guest_id,
                        name=guest_name,
                        similarity=match.similarity,
                        reason="REFERENCE_STALE",
                    )

        return LocalRecognitionResult(
            recognized=True,
            allowed=True,
            reason="AUTHORIZED",
            guest_id=match.guest_id,
            name=guest_name,
            similarity=match.similarity,
        )

    return local_recognize


# ── Worker de reconhecimento (thread em background) ───────────────────────────

class RecognitionWorker(threading.Thread):
    """
    Recebe frames via submit_frame() e processa em background.
    A thread principal nunca bloqueia — apenas lê o último resultado.
    """

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        guard: AccessEventGuard,
        recognize_fn: Callable[[object], LocalRecognitionResult] | None = None,
    ):
        super().__init__(daemon=True)
        self.detector = detector
        self.embedder = embedder
        self.guard    = guard
        self.recognize_fn = recognize_fn
        self._stop_event = threading.Event()

        # Frame pendente de processamento
        self._frame_lock    = threading.Lock()
        self._pending_frame = None
        self._frame_ready   = threading.Event()

        # Último resultado para exibição (thread-safe)
        self._result_lock  = threading.Lock()
        self.latest_result = None  # {"state": str, "name"?: str, "until": float}

        # Janela de confirmação por slot de face
        self._confirmations: dict[int, RecognitionConfirmation] = {}
        self._unknown_frames = 0

    def submit_frame(self, frame) -> None:
        """Deposita um frame. Se já havia um pendente, descarta o anterior."""
        with self._frame_lock:
            self._pending_frame = frame.copy()
        self._frame_ready.set()

    def get_result(self) -> dict | None:
        """Lê o último resultado sem bloquear."""
        with self._result_lock:
            return self.latest_result

    def stop(self) -> None:
        self._stop_event.set()
        self._frame_ready.set()

    def _set_result(self, result: dict | None) -> None:
        with self._result_lock:
            self.latest_result = result

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._frame_ready.wait()
            self._frame_ready.clear()

            if self._stop_event.is_set():
                break

            with self._frame_lock:
                frame, self._pending_frame = self._pending_frame, None

            if frame is None:
                continue

            try:
                self._process(frame)
            except Exception as exc:
                print(f"[WARN] Erro no worker: {exc}")
                self._show_validation_error()

    def _show_validation_error(self) -> None:
        self._unknown_frames = 0
        for confirmation in self._confirmations.values():
            confirmation.reset()
        self._set_result({
            "state": "error",
            "until": time.monotonic() + SHOW_SECS,
        })

    def _process(self, frame) -> None:
        faces   = self.detector.detect(frame)
        n_faces = len(faces)

        # Remove confirmações de slots que saíram do frame
        for idx in list(self._confirmations.keys()):
            if idx >= n_faces:
                del self._confirmations[idx]

        if n_faces == 0:
            self._unknown_frames = 0
            return

        if n_faces > 1:
            self._confirmations.clear()
            self._show_validation_error()
            print("[NEGADO] MAIS DE UM ROSTO NO ENQUADRAMENTO")
            return

        any_recognized = False
        successful_unknown = False
        recognize_failed = False

        for idx, face in enumerate(faces):
            if idx not in self._confirmations:
                self._confirmations[idx] = RecognitionConfirmation(required_frames=5)

            embedding = self.embedder.generate(face, frame)

            # Reconhecimento local
            try:
                if self.recognize_fn is None:
                    raise RuntimeError("Função de reconhecimento não configurada")
                local_result = self.recognize_fn(embedding)
            except Exception as exc:
                print(f"[WARN] Falha no reconhecimento: {exc}")
                recognize_failed = True
                continue

            raw_guest_id = local_result.guest_id if local_result.recognized else None
            any_recognized = any_recognized or local_result.recognized
            successful_unknown = successful_unknown or not local_result.recognized
            confirmed = self._confirmations[idx].update(raw_guest_id)

            if confirmed is None:
                continue

            # A decisão exibida deve pertencer à mesma identidade confirmada.
            if raw_guest_id != confirmed:
                continue

            name = (local_result.name or confirmed).upper()
            sim  = local_result.similarity

            self._set_result({
                "state": "authorized" if local_result.allowed else "denied",
                "name": name if local_result.allowed else None,
                "until": time.monotonic() + SHOW_SECS,
            })

            if not self.guard.can_register(confirmed):
                continue

            status = "OK" if local_result.allowed else "NEGADO"
            print(f"[{status}] {local_result.reason}")
            if DEBUG:
                sim_str = f"sim={sim:.3f}" if sim is not None else "sim=?"
                print(f"[DEBUG] guest={confirmed} name={name} {sim_str}")

        if recognize_failed:
            self._show_validation_error()
            return

        if any_recognized:
            self._unknown_frames = 0
        elif successful_unknown:
            self._unknown_frames += 1
            if self._unknown_frames >= UNKNOWN_REQUIRED:
                self._set_result({
                    "state": "unknown",
                    "until": time.monotonic() + SHOW_SECS,
                })
                self._unknown_frames = 0
                if self.guard.can_register(None):
                    print("[NEGADO] ROSTO DESCONHECIDO")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from reconhecimento.api.client import AccessControlClient, EnrollmentClient, RecognitionApiError
    from reconhecimento.display import run_display
    from reconhecimento.enrollment.pipeline import EnrollmentPipeline
    from reconhecimento.offline import (
        EncryptedFaceCache,
        HybridRecognitionService,
        OfflineAccessStore,
        OfflineReplayThread,
    )
    from reconhecimento.recognition.matcher import InMemoryFaceIndex
    from reconhecimento.sync.face_sync import FaceSyncThread

    load_dotenv()
    try:
        camera_index = camera_index_from_env()
        access_direction = access_direction_from_env()
    except ValueError as exc:
        print(exc)
        raise SystemExit(2) from None

    print(f"[CONFIG] Camera index: {camera_index} | Direction: {access_direction}")

    detector = FaceDetector()
    embedder = FaceEmbedder()
    guard    = AccessEventGuard(cooldown_seconds=10.0, unknown_cooldown_seconds=10.0)

    try:
        api_client = AccessControlClient(
            api_url=os.getenv("API_URL", ""),
            device_key=os.getenv("DEVICE_KEY", ""),
            access_point=os.getenv("ACCESS_POINT", ""),
        )
    except RecognitionApiError as exc:
        print(f"[CONFIG] {exc}")
        raise SystemExit(2) from None

    print(f"[CONFIG] API VIP online | Ponto: {api_client.access_point}")

    threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.60"))
    cache = EncryptedFaceCache()
    cached = cache.load()
    index = InMemoryFaceIndex(threshold=threshold)
    index.replace({guest_id: value[0] for guest_id, value in cached.items()})
    cached_names = {guest_id: value[1] for guest_id, value in cached.items()}
    print(f"[CACHE] {len(cached)} identidade(s) disponíveis localmente")

    enrollment_client = None
    repository = FaceRepository()
    sync = FaceSyncThread(repository=repository, index=index, encrypted_cache=cache)
    enrollment_enabled = os.getenv("FACE_ENROLLMENT_ENABLED", "false").lower() in {
        "1", "true", "yes",
    }
    if enrollment_enabled:
        try:
            enrollment_client = EnrollmentClient(
                api_url=os.getenv("API_URL", ""),
                device_key=os.getenv("ENROLLMENT_DEVICE_KEY", ""),
            )
            repository = FaceRepository()
            enrollment_pipeline = EnrollmentPipeline(
                repository=repository,
                detector=FaceDetector(),
                embedder=FaceEmbedder(),
                enrollment_client=enrollment_client,
            )
            sync.set_enrollment_pipeline(enrollment_pipeline)
            print("[CONFIG] Enrollment automático de convidados ativo")
        except (RecognitionApiError, FaceDatabaseError, ValueError) as exc:
            api_client.close()
            print(f"[CONFIG] Enrollment inválido: {exc}")
            raise SystemExit(2) from None

    sync.start()
    offline_store = OfflineAccessStore(
        maximum_capacity=int(os.getenv("VIP_MAX_CAPACITY", "400")),
    )
    hybrid = HybridRecognitionService(
        online_client=api_client,
        index=index,
        names=lambda: {**cached_names, **sync.guest_names},
        store=offline_store,
        direction=access_direction,
        access_point=api_client.access_point,
    )

    replay_thread = None
    replay_clients = []
    if os.getenv("STATION_NUMBER", "1") == "1":
        entry_key = os.getenv("STATION_1_DEVICE_KEY", "").strip()
        exit_key = os.getenv("STATION_2_DEVICE_KEY", "").strip()
        if entry_key and exit_key:
            replay_clients = [
                AccessControlClient(os.getenv("API_URL", ""), entry_key, "ENTRADA_PRINCIPAL"),
                AccessControlClient(os.getenv("API_URL", ""), exit_key, "SAIDA_PRINCIPAL"),
            ]
            replay_thread = OfflineReplayThread(
                offline_store,
                cache,
                {client.access_point: client for client in replay_clients},
            )
            replay_thread.start()
            print(f"[OFFLINE] Fila pendente: {offline_store.pending_count()}")

    # Interface Qt portrait
    exit_code = run_display(
        detector=detector,
        embedder=embedder,
        guard=guard,
        recognize_fn=hybrid.recognize,
        sync_thread=sync,
        camera_index=camera_index,
        access_direction=access_direction,
    )

    if replay_thread is not None:
        replay_thread.stop()
        replay_thread.join(timeout=3.0)
    for client in replay_clients:
        client.close()
    sync.stop()
    sync.join(timeout=3.0)
    if enrollment_client is not None:
        enrollment_client.close()
    if repository is not None:
        repository.close()
    api_client.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
