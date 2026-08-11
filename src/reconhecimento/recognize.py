"""Loop principal de reconhecimento facial.

Fluxo:
  1. Captura frame da câmera
  2. A cada N frames, envia para o worker em background
  3. Worker detecta faces, gera embeddings e chama a API
  4. API devolve {allowed, name, reason} — o log já é salvo lá
  5. Badge verde/vermelho é exibido na janela por 3 segundos
"""
import threading
import time
import cv2

from reconhecimento.camera.capture import Camera
from reconhecimento.recognition.confirmation import RecognitionConfirmation
from reconhecimento.recognition.detector import FaceDetector
from reconhecimento.recognition.embedder import FaceEmbedder
from reconhecimento.recognition.guard import AccessEventGuard
from reconhecimento.api.client import recognize, RecognitionResult

# ── Constantes visuais ────────────────────────────────────────────────────────
GREEN = (50, 205, 50)
RED   = (50, 50, 220)
FONT  = cv2.FONT_HERSHEY_SIMPLEX

# ── Constantes de comportamento ───────────────────────────────────────────────
SHOW_SECS        = 3.0   # segundos que o badge fica visível
PROCESS_EVERY_N  = 3     # analisa 1 a cada N frames (alivia CPU)
UNKNOWN_REQUIRED = 10    # frames sem reconhecimento para mostrar "Desconhecido"


# ── Desenho do badge ──────────────────────────────────────────────────────────

def _draw_badge(frame, allowed: bool, name: str) -> None:
    """Desenha uma pill colorida na parte inferior do frame."""
    h, w    = frame.shape[:2]
    color   = GREEN if allowed else RED
    label   = f"ACESSO PERMITIDO  {name}" if allowed else f"ACESSO NEGADO  {name}"

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


# ── Worker de reconhecimento (thread em background) ───────────────────────────

class RecognitionWorker(threading.Thread):
    """
    Recebe frames via submit_frame() e processa em background.
    A thread principal nunca bloqueia — apenas lê o último resultado.
    """

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder, guard: AccessEventGuard):
        super().__init__(daemon=True)
        self.detector = detector
        self.embedder = embedder
        self.guard    = guard

        # Frame pendente de processamento
        self._frame_lock    = threading.Lock()
        self._pending_frame = None
        self._frame_ready   = threading.Event()

        # Último resultado para exibição (thread-safe)
        self._result_lock  = threading.Lock()
        self.latest_result = None  # {"allowed": bool, "name": str, "until": float}

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

    def _set_result(self, result: dict | None) -> None:
        with self._result_lock:
            self.latest_result = result

    def run(self) -> None:
        while True:
            self._frame_ready.wait()
            self._frame_ready.clear()

            with self._frame_lock:
                frame, self._pending_frame = self._pending_frame, None

            if frame is None:
                continue

            try:
                self._process(frame)
            except Exception as exc:
                print(f"[WARN] Erro no worker: {exc}")

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

        any_recognized = False

        for idx, face in enumerate(faces):
            if idx not in self._confirmations:
                self._confirmations[idx] = RecognitionConfirmation(required_frames=5)

            embedding = self.embedder.generate(face)

            # Chama a API para obter reconhecimento + decisão de acesso
            try:
                api_result: RecognitionResult = recognize(embedding)
            except RuntimeError as exc:
                print(f"[WARN] Falha na API: {exc}")
                continue

            # Alimenta a janela de confirmação com o person_id retornado
            raw_person_id = api_result.person_id if api_result.recognized else None
            confirmed = self._confirmations[idx].update(raw_person_id)

            if confirmed is None:
                continue

            any_recognized = True

            if not self.guard.can_register(confirmed):
                continue

            name = (api_result.name or confirmed).upper()
            sim  = api_result.similarity

            self._set_result({
                "allowed": api_result.allowed,
                "name":    name,
                "until":   time.monotonic() + SHOW_SECS,
            })

            status = "OK" if api_result.allowed else "NEGADO"
            sim_str = f"sim={sim:.3f}" if sim is not None else "sim=?"
            print(f"[{status}] {name} — {api_result.reason} ({sim_str})")

        if not any_recognized:
            self._unknown_frames += 1
            if self._unknown_frames >= UNKNOWN_REQUIRED and self.guard.can_register(None):
                self._set_result({
                    "allowed": False,
                    "name":    "Desconhecido",
                    "until":   time.monotonic() + SHOW_SECS,
                })
                self._unknown_frames = 0
                print("[NEGADO] ROSTO DESCONHECIDO")
        else:
            self._unknown_frames = 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    camera   = Camera()
    detector = FaceDetector()
    embedder = FaceEmbedder()
    guard    = AccessEventGuard(cooldown_seconds=10.0, unknown_cooldown_seconds=10.0)
    worker   = RecognitionWorker(detector=detector, embedder=embedder, guard=guard)
    worker.start()

    frame_count = 0
    print("Sistema de reconhecimento iniciado. Pressione Q para sair.")

    try:
        while True:
            frame        = camera.read()
            frame_count += 1

            if frame_count % PROCESS_EVERY_N == 0:
                worker.submit_frame(frame)

            result = worker.get_result()
            if result and time.monotonic() < result["until"]:
                _draw_badge(frame, result["allowed"], result["name"])

            cv2.imshow("Controle de Acesso", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()


if __name__ == "__main__":
    main()
