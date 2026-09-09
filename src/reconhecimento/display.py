"""Interface Qt para reconhecimento facial — layout portrait (totem).

Resolve 768x1366 portrait como target principal, com suporte responsivo
a outras resoluções. Detecta orientação automaticamente.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt5.QtCore import (
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
    QThread,
    QRect,
)
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QFont,
    QFontMetrics,
    QColor,
    QPainter,
    QPen,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)

# ── Coisas───────────────────────────────────────────────────────────────────────

BG_DARK = "#0D1117"
BG_CARD = "#161B22"
FG_PRIMARY = "#E6EDF3"
FG_SECONDARY = "#8B949E"
ACCENT_GREEN = "#2EA043"
ACCENT_RED = "#DA3633"
ACCENT_AMBER = "#D29922"
BORDER_COLOR = "#30363D"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_POLL_MS = 33  # ~30fps

AUTHORIZED_SHOW_MS = 2000
DEFAULT_RESULT_SHOW_MS = 3000
UNKNOWN_SHOW_MS = 2000


def _first_name(name: object | None) -> str:
    parts = str(name).strip().split() if name is not None else []
    return parts[0] if parts else ""


def _safe_display_index(raw_value: str, screen_count: int) -> int:
    try:
        configured_index = int(raw_value)
    except ValueError:
        return 0
    if configured_index < 0 or configured_index >= screen_count:
        return 0
    return configured_index


def _layout_metrics(width: int, height: int) -> tuple[int, int, int]:
    """Retorna margem segura e dimensoes do viewport portrait."""
    safe_margin = max(12, round(width * 0.03))
    preview_width = max(1, min(int(width * 0.95), width - (safe_margin * 2)))
    preview_height = max(1, min(750, max(600, round(height * 0.50))))
    return safe_margin, preview_width, preview_height


# ── Dados de resultado ─────────────────────────────────────────────────────────

@dataclass
class DisplayResult:
    state: str  # "authorized" | "denied" | "unknown" | "error" | "idle"
    name: str | None = None
    similarity: float | None = None


# ── Worker de reconhecimento (thread) ──────────────────────────────────────────

class RecognitionWorkerThread(QThread):
    """Thread de reconhecimento que emite sinais Qt."""

    result_ready = pyqtSignal(object)  # DisplayResult
    frame_processed = pyqtSignal()

    def __init__(
        self,
        detector,
        embedder,
        guard,
        recognize_fn=None,
        parent=None,
    ):
        super().__init__(parent)
        self.detector = detector
        self.embedder = embedder
        self.guard = guard
        self.recognize_fn = recognize_fn

        self._stop = False
        self._pending_frame = None
        self._frame_lock = __import__("threading").Lock()
        self._frame_event = __import__("threading").Event()

        from reconhecimento.recognition.confirmation import RecognitionConfirmation
        self._confirmations: dict[int, RecognitionConfirmation] = {}
        self._unknown_frames = 0
        self._show_secs = 3.0
        self._unknown_required = 10

    def submit_frame(self, frame):
        with self._frame_lock:
            self._pending_frame = frame.copy()
        self._frame_event.set()

    def stop(self):
        self._stop = True
        self._frame_event.set()

    def run(self):
        while not self._stop:
            self._frame_event.wait()
            self._frame_event.clear()
            if self._stop:
                break

            with self._frame_lock:
                frame, self._pending_frame = self._pending_frame, None
            if frame is None:
                continue

            try:
                self._process(frame)
            except Exception as exc:
                print(f"[WARN] Erro no worker: {exc}")
                self.result_ready.emit(DisplayResult(state="error"))

    def _process(self, frame):
        faces = self.detector.detect(frame)
        n_faces = len(faces)

        for idx in list(self._confirmations.keys()):
            if idx >= n_faces:
                del self._confirmations[idx]

        if n_faces == 0:
            self._unknown_frames = 0
            return

        if n_faces > 1:
            self._confirmations.clear()
            self.result_ready.emit(DisplayState.error())
            self._unknown_frames = 0
            return

        any_recognized = False
        recognize_failed = False

        for idx, face in enumerate(faces):
            if idx not in self._confirmations:
                from reconhecimento.recognition.confirmation import RecognitionConfirmation
                self._confirmations[idx] = RecognitionConfirmation(required_frames=5)

            embedding = self.embedder.generate(face, frame)

            try:
                if self.recognize_fn is None:
                    raise RuntimeError("Funcao de reconhecimento nao configurada")
                local_result = self.recognize_fn(embedding)
            except Exception as exc:
                print(f"[WARN] Falha no reconhecimento: {exc}")
                recognize_failed = True
                continue

            raw_guest_id = local_result.guest_id if local_result.recognized else None
            any_recognized = any_recognized or local_result.recognized
            confirmed = self._confirmations[idx].update(raw_guest_id)

            if confirmed is None:
                continue

            if raw_guest_id != confirmed:
                continue

            name = local_result.name or confirmed
            sim = local_result.similarity

            validation_errors = {"DB_UNAVAILABLE", "REFERENCE_STALE"}
            if local_result.allowed:
                state = "authorized"
            elif local_result.reason in validation_errors:
                state = "error"
            else:
                state = "denied"
            self.result_ready.emit(DisplayResult(
                state=state,
                name=name if local_result.allowed else None,
                similarity=sim,
            ))

            if not self.guard.can_register(confirmed):
                continue

            status = "OK" if local_result.allowed else "NEGADO"
            print(f"[{status}] {local_result.reason}")

        if recognize_failed:
            self.result_ready.emit(DisplayState.error())
            return

        if not any_recognized:
            self._unknown_frames += 1
            if self._unknown_frames >= self._unknown_required:
                self.result_ready.emit(DisplayResult(state="unknown"))
                self._unknown_frames = 0
                if self.guard.can_register(None):
                    print("[NEGADO] ROSTO DESCONHECIDO")
        else:
            self._unknown_frames = 0


# ── DisplayState helper ────────────────────────────────────────────────────────

class DisplayState:
    @staticmethod
    def idle() -> DisplayResult:
        return DisplayResult(state="idle")

    @staticmethod
    def error() -> DisplayResult:
        return DisplayResult(state="error")


# ── Widget da câmera ───────────────────────────────────────────────────────────

class CameraWidget(QLabel):
    """Label que exibe o frame da câmera em tempo real."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(0, 0)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: QPixmap | None = None
        self._displayed_pixmap: QPixmap | None = None
        self._source_size = (CAMERA_WIDTH, CAMERA_HEIGHT)
        self._visual_crop_rect = (0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)

    def aspect_ratio(self) -> float:
        if self._pixmap is None or self._pixmap.height() == 0:
            return CAMERA_WIDTH / CAMERA_HEIGHT
        return self._pixmap.width() / self._pixmap.height()

    def visual_crop_rect(self) -> tuple[int, int, int, int]:
        return self._visual_crop_rect

    def source_size(self) -> tuple[int, int]:
        return self._source_size

    def displayed_size(self) -> tuple[int, int]:
        if self._displayed_pixmap is None:
            return (0, 0)
        return (self._displayed_pixmap.width(), self._displayed_pixmap.height())

    def update_frame(self, frame: np.ndarray):
        """Converte frame BGR do OpenCV para QPixmap e exibe."""
        h, w, ch = frame.shape
        self._source_size = (w, h)
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._apply_cover_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._apply_cover_pixmap()

    def _apply_cover_pixmap(self) -> None:
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return

        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        crop_x = max(0, (scaled.width() - self.width()) // 2)
        crop_y = max(0, (scaled.height() - self.height()) // 2)
        cropped = scaled.copy(crop_x, crop_y, self.width(), self.height())

        scale_x = scaled.width() / self._pixmap.width()
        scale_y = scaled.height() / self._pixmap.height()
        source_x = round(crop_x / scale_x)
        source_y = round(crop_y / scale_y)
        source_w = round(self.width() / scale_x)
        source_h = round(self.height() / scale_y)
        self._visual_crop_rect = (source_x, source_y, source_w, source_h)
        self._displayed_pixmap = cropped
        self.setPixmap(cropped)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        guide_width = int(self.width() * 0.58)
        guide_height = int(self.height() * 0.50)
        guide_x = (self.width() - guide_width) // 2
        guide_y = (self.height() - guide_height) // 2
        painter.setPen(QPen(QColor(230, 237, 243, 85), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(guide_x, guide_y, guide_width, guide_height, 28, 28)


# ── Widget de status ───────────────────────────────────────────────────────────

class StatusWidget(QLabel):
    """Label de mensagem de status abaixo da câmera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self._set_idle()

    def _set_idle(self):
        self.setText("Posicione seu rosto no centro da tela.")
        self.setStyleSheet(f"""
            color: {FG_SECONDARY};
            font-size: 18px;
            font-weight: 500;
            padding: 12px;
        """)

    def set_status(self, state: str, name: str | None = None):
        if state == "authorized":
            self.setText("Aproveite a experiência VIP.\nPode entrar.")
            self.setStyleSheet(f"""
                color: {ACCENT_GREEN};
                font-size: 18px;
                font-weight: 600;
                padding: 12px;
            """)
        elif state == "denied":
            self.setText("Procure nossa equipe para verificar seu cadastro.")
            self.setStyleSheet(f"""
                color: {ACCENT_RED};
                font-size: 18px;
                font-weight: 600;
                padding: 12px;
            """)
        elif state == "unknown":
            self.setText("Tente novamente olhando para a câmera.\nSe precisar, procure nossa equipe.")
            self.setStyleSheet(f"""
                color: {ACCENT_RED};
                font-size: 18px;
                font-weight: 500;
                padding: 12px;
            """)
        elif state == "error":
            self.setText("Procure nossa equipe.")
            self.setStyleSheet(f"""
                color: {ACCENT_AMBER};
                font-size: 18px;
                font-weight: 500;
                padding: 12px;
            """)
        elif state == "validating":
            self.setText("Aguarde um instante.")
            self.setStyleSheet(f"""
                color: {FG_SECONDARY};
                font-size: 18px;
                font-weight: 500;
                padding: 12px;
            """)
        else:
            self._set_idle()


# ── Tela de sucesso ────────────────────────────────────────────────────────────

class SuccessOverlay(QWidget):
    """Overlay de tela cheia exibido quando allowed=true."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self._name = ""

    def show_success(self, name: str):
        self._name = _first_name(name)
        self.setVisible(True)
        self.update()

    def hide_success(self):
        self.setVisible(False)

    def paintEvent(self, event):
        if not self.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fundo semi-transparente
        painter.fillRect(self.rect(), QColor(13, 17, 23, 230))

        w = self.width()
        h = self.height()

        # Checkmark verde grande
        cx, cy = w // 2, h // 3
        radius = min(w, h) // 8

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(ACCENT_GREEN))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Checkmark branco
        painter.setPen(QPen(QColor("white"), max(4, radius // 8), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        check_size = int(radius * 0.55)
        painter.drawLine(cx - check_size, cy, cx - check_size // 3, cy + check_size // 2)
        painter.drawLine(cx - check_size // 3, cy + check_size // 2, cx + check_size, cy - check_size // 2)

        # Texto principal
        content_margin = max(16, round(w * 0.05))
        text_width = w - content_margin * 2
        first_name = self._name
        font_large = self._fitted_font(
            "ACESSO LIBERADO", max(36, h // 20), 24, text_width, QFont.Bold
        )
        painter.setFont(font_large)
        painter.setPen(QColor(FG_PRIMARY))
        painter.drawText(
            QRect(content_margin, cy + radius + 30, text_width, h // 6),
            Qt.AlignCenter | Qt.TextWordWrap,
            "ACESSO LIBERADO",
        )

        # Nome
        welcome = f"Bem-vindo, {first_name}!" if first_name else "Bem-vindo!"
        font_name = self._fitted_font(
            welcome, max(48, h // 14), 24, text_width, QFont.Bold
        )
        painter.setFont(font_name)
        painter.setPen(QColor(ACCENT_GREEN))
        painter.drawText(
            QRect(content_margin, cy + radius + 30 + h // 6, text_width, h // 8),
            Qt.AlignCenter | Qt.TextWordWrap,
            welcome,
        )

        # Mensagem secundaria
        font_small = QFont("Segoe UI", max(18, h // 36))
        painter.setFont(font_small)
        painter.setPen(QColor(FG_SECONDARY))
        painter.drawText(
            QRect(content_margin, cy + radius + 30 + h // 6 + h // 8, text_width, h // 10),
            Qt.AlignCenter | Qt.TextWordWrap,
            "Aproveite a experiência VIP.",
        )

        font_action = QFont("Segoe UI", max(22, h // 28), QFont.Bold)
        painter.setFont(font_action)
        painter.setPen(QColor(FG_PRIMARY))
        painter.drawText(
            QRect(content_margin, cy + radius + 30 + h // 6 + h // 8 + h // 10, text_width, h // 10),
            Qt.AlignCenter | Qt.TextWordWrap,
            "Pode entrar.",
        )

        painter.end()

    @staticmethod
    def _fitted_font(text: str, maximum: int, minimum: int, width: int, weight: int) -> QFont:
        font = QFont("Segoe UI")
        font.setWeight(weight)
        for pixel_size in range(maximum, minimum - 1, -1):
            font.setPixelSize(pixel_size)
            if QFontMetrics(font).horizontalAdvance(text) <= width:
                break
        return font


# ── Janela principal ───────────────────────────────────────────────────────────

class PortraitWindow(QMainWindow):
    """Janela principal com layout portrait para totem."""

    def __init__(self, screen, camera_index: int = 0):
        super().__init__()
        self.setWindowTitle("VIP ISP Evolution — Reconhecimento Facial")
        self.setWindowFlag(Qt.FramelessWindowHint, True)

        # Detectar orientação
        screen_geometry = screen.geometry()
        self._is_portrait = screen_geometry.height() > screen_geometry.width()
        self._screen_size = screen_geometry.size()

        # Centralizar e configurar
        self._setup_geometry(screen_geometry)
        self._build_ui()
        self._apply_style()
        self._update_camera_frame_constraints()

        # Câmera
        self._camera = None
        self._camera_index = camera_index

        # Timer para polling da câmera
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._frame_count = 0

        # Controle de estado
        self._current_state = "idle"
        self._success_until = 0.0

    def _setup_geometry(self, screen):
        """Posiciona a janela na geometria global da tela selecionada."""
        self.setGeometry(screen)

    def _build_ui(self):
        """Constrói a interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        safe_margin, preview_width, preview_height = _layout_metrics(
            self._screen_size.width(), self._screen_size.height()
        )
        main_layout.setContentsMargins(safe_margin, 0, safe_margin, 0)
        main_layout.setSpacing(2)
        self._main_layout = main_layout

        # ── TOPO: Branding ────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 4)
        header_layout.setAlignment(Qt.AlignCenter)

        brand = QLabel("ISP EVOLUTION")
        brand.setAlignment(Qt.AlignCenter)
        brand.setWordWrap(True)
        brand.setObjectName("brand")
        header_layout.addWidget(brand)

        subtitle = QLabel("SALA VIP")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")
        header_layout.addWidget(subtitle)

        main_layout.addWidget(header)

        # ── CENTRO SUPERIOR: Instrução ───────────────────────────────
        self._instruction = QLabel("Olhe para a câmera")
        self._instruction.setAlignment(Qt.AlignCenter)
        self._instruction.setWordWrap(True)
        self._instruction.setObjectName("instruction")
        self._instruction.setContentsMargins(0, 0, 0, 0)
        self._instruction.setMaximumHeight(42)
        main_layout.addWidget(self._instruction)

        # ── CENTRO: Câmera ───────────────────────────────────────────
        camera_frame = QFrame()
        camera_frame.setObjectName("cameraFrame")
        camera_frame.setMaximumWidth(preview_width)
        camera_frame.setFixedHeight(preview_height)
        camera_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._camera_frame = camera_frame
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(4, 4, 4, 4)
        camera_layout.setAlignment(Qt.AlignCenter)

        self._camera_widget = CameraWidget()
        self._camera_widget.setMinimumHeight(max(1, preview_height - 8))
        camera_layout.addWidget(self._camera_widget)

        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.setAlignment(Qt.AlignCenter)
        camera_row.addStretch()
        camera_row.addWidget(camera_frame, stretch=1)
        camera_row.addStretch()
        main_layout.addLayout(camera_row, stretch=1)

        # ── CENTRO INFERIOR: Status ──────────────────────────────────
        self._status = StatusWidget()
        main_layout.addWidget(self._status)

        # ── Overlay de sucesso ────────────────────────────────────────
        self._success_overlay = SuccessOverlay(central)
        # Posiciona sobre todo o central

    def _apply_style(self):
        """Aplica tema dark."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_DARK};
            }}
            QWidget#header {{
                background-color: {BG_CARD};
                border-bottom: 1px solid {BORDER_COLOR};
            }}
            QLabel#brand {{
                color: {FG_PRIMARY};
                font-size: 22px;
                font-weight: 700;
                letter-spacing: 4px;
            }}
            QLabel#subtitle {{
                color: {ACCENT_GREEN};
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 6px;
                margin-top: 2px;
            }}
            QLabel#instruction {{
                color: {FG_SECONDARY};
                font-size: 20px;
                font-weight: 500;
                padding: 8px;
                background-color: {BG_DARK};
            }}
            QFrame#cameraFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
                margin: 0px;
            }}
            QLabel#footerText {{
                color: {FG_SECONDARY};
                font-size: 11px;
            }}
            QFrame#footer {{
                background-color: {BG_CARD};
                border-top: 1px solid {BORDER_COLOR};
            }}
        """)

    def resizeEvent(self, event):
        """Redimensiona overlay de sucesso ao redimensionar janela."""
        super().resizeEvent(event)
        if hasattr(self, "_success_overlay"):
            self._success_overlay.setGeometry(self.centralWidget().rect())
        if hasattr(self, "_main_layout"):
            safe_margin, preview_width, preview_height = _layout_metrics(
                self.width(), self.height()
            )
            self._main_layout.setContentsMargins(safe_margin, 0, safe_margin, 0)
            self._camera_frame.setMaximumWidth(preview_width)
            self._camera_frame.setFixedHeight(preview_height)
            self._camera_widget.setMinimumHeight(max(1, preview_height - 8))
            self._update_camera_frame_constraints()

    def _update_camera_frame_constraints(self) -> None:
        frame_margins = self._camera_frame.layout().contentsMargins()
        content_height = max(
            1,
            self._camera_frame.height() - frame_margins.top() - frame_margins.bottom(),
        )
        self._camera_widget.setFixedHeight(content_height)

    # ── API pública ──────────────────────────────────────────────────

    def start_camera(self):
        """Inicia a câmera e o timer de polling."""
        from reconhecimento.camera.capture import Camera

        if self._camera is not None:
            return
        try:
            self._camera = Camera(
                camera_index=self._camera_index,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
            )
            self._timer.start(CAMERA_POLL_MS)
        except RuntimeError as exc:
            print(exc)
            self._status.set_status("error")
            self._set_instruction("Não foi possível validar seu acesso", ACCENT_AMBER)

    def stop_camera(self):
        """Para a câmera e o timer."""
        self._timer.stop()
        if self._camera:
            self._camera.release()
            self._camera = None

    def set_worker(self, worker: RecognitionWorkerThread):
        """Conecta o worker ao display."""
        self._worker = worker
        worker.result_ready.connect(self._on_result)

    def set_sync_status(self, count: int):
        """Registra status técnico sem expor dados na tela pública."""
        print(f"[DISPLAY] embeddings={count}")

    # ── Slots ────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_timer(self):
        """Poll da câmera — chamado a cada ~33ms."""
        if self._camera is None:
            return

        try:
            frame = self._camera.read()
        except RuntimeError:
            return

        self._frame_count += 1

        # Enviar para worker a cada 3 frames
        if self._frame_count % 3 == 0 and hasattr(self, "_worker"):
            self._worker.submit_frame(frame)

        # Atualizar preview (se não estiver no overlay de sucesso)
        if self._current_state != "authorized":
            self._camera_widget.update_frame(frame)
            self._update_camera_frame_constraints()

    @pyqtSlot(object)
    def _on_result(self, result: DisplayResult):
        """Recebe resultado do worker de reconhecimento."""
        state = result.state

        if state == "authorized":
            self._current_state = "authorized"
            self._success_until = time.monotonic() + 2.0
            self._success_overlay.show_success(result.name or "")
            self._status.set_status("authorized", result.name)
            self._set_instruction("ACESSO LIBERADO", ACCENT_GREEN, 22, 700)
            show_ms = AUTHORIZED_SHOW_MS

        elif state == "denied":
            self._current_state = "denied"
            self._success_until = time.monotonic() + 3.0
            self._success_overlay.hide_success()
            self._status.set_status("denied")
            self._set_instruction("Acesso não disponível", ACCENT_RED)
            show_ms = DEFAULT_RESULT_SHOW_MS

        elif state == "unknown":
            self._current_state = "unknown"
            self._success_until = time.monotonic() + 2.0
            self._success_overlay.hide_success()
            self._status.set_status("unknown")
            self._set_instruction("Não conseguimos identificar você", ACCENT_RED)
            show_ms = UNKNOWN_SHOW_MS

        elif state == "error":
            self._current_state = "error"
            self._success_until = time.monotonic() + 3.0
            self._success_overlay.hide_success()
            self._status.set_status("error")
            self._set_instruction("Não foi possível validar seu acesso", ACCENT_AMBER)
            show_ms = DEFAULT_RESULT_SHOW_MS

        elif state == "validating":
            self._current_state = "validating"
            self._success_overlay.hide_success()
            self._status.set_status("validating")
            self._set_instruction("Validando seu acesso...", FG_SECONDARY)
            show_ms = DEFAULT_RESULT_SHOW_MS

        else:
            self._return_to_idle()
            show_ms = 0

        # Timer para voltar ao idle
        if state != "idle":
            QTimer.singleShot(show_ms, self._check_return_to_idle)

    def _set_instruction(
        self,
        text: str,
        color: str,
        font_size: int = 20,
        font_weight: int = 600,
    ) -> None:
        self._instruction.setText(text)
        self._instruction.setStyleSheet(f"""
            color: {color};
            font-size: {font_size}px;
            font-weight: {font_weight};
            padding: 8px;
            background-color: {BG_DARK};
        """)

    def _check_return_to_idle(self):
        """Verifica se deve voltar ao estado idle."""
        if time.monotonic() >= self._success_until:
            self._return_to_idle()

    def _return_to_idle(self):
        """Retorna ao estado idle."""
        self._current_state = "idle"
        self._success_overlay.hide_success()
        self._status.set_status("idle")
        self._set_instruction("Olhe para a câmera", FG_SECONDARY, 20, 500)

    def keyPressEvent(self, event):
        """ESC/Q encerra; F11 alterna fullscreen para manutenção."""
        if event.key() in {Qt.Key_Escape, Qt.Key_Q}:
            self.close()
        elif event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Cleanup ao fechar."""
        self.stop_camera()
        if hasattr(self, "_worker"):
            self._worker.stop()
        event.accept()


# ── Função de inicialização ────────────────────────────────────────────────────

def run_display(
    detector,
    embedder,
    guard,
    recognize_fn=None,
    repository=None,
    sync_thread=None,
    camera_index: int = 0,
):
    """Executa o loop Qt com reconhecimento facial."""
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    screens = app.screens()
    configured_display = os.getenv("DISPLAY_INDEX", "0")
    display_index = _safe_display_index(configured_display, len(screens))
    try:
        parsed_display = int(configured_display)
    except ValueError:
        parsed_display = -1
    if parsed_display != display_index:
        print(
            f"[DISPLAY] Invalid DISPLAY_INDEX: {configured_display}; "
            f"using {display_index}"
        )
    selected_screen = screens[display_index]
    geometry = selected_screen.geometry()
    available = selected_screen.availableGeometry()

    print(f"[DISPLAY] configured index={configured_display}")
    print(f"[DISPLAY] selected screen={selected_screen.name()}")
    print(
        f"[DISPLAY] Geometry: {geometry.width()}x{geometry.height()} "
        f"at ({geometry.x()},{geometry.y()})"
    )
    print(
        f"[DISPLAY] Available: {available.width()}x{available.height()} "
        f"at ({available.x()},{available.y()})"
    )
    print(f"[DISPLAY] DPR: {selected_screen.devicePixelRatio():g}")
    print(f"[DISPLAY] Logical DPI: {selected_screen.logicalDotsPerInch():g}")
    print(f"[DISPLAY] Physical DPI: {selected_screen.physicalDotsPerInch():g}")

    window = PortraitWindow(screen=selected_screen, camera_index=camera_index)
    window.winId()
    window.windowHandle().setScreen(selected_screen)
    window.setGeometry(geometry)

    # Worker
    worker = RecognitionWorkerThread(
        detector=detector,
        embedder=embedder,
        guard=guard,
        recognize_fn=recognize_fn,
    )
    window.set_worker(worker)

    # Sync status
    if sync_thread:
        window.set_sync_status(sync_thread.last_sync_count)

    # Iniciar
    worker.start()
    window.start_camera()
    window.showFullScreen()

    exit_code = app.exec_()

    # Cleanup
    worker.stop()
    worker.wait(2000)
    if repository:
        repository.close()

    return exit_code
