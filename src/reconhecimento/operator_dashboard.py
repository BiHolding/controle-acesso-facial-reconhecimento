"""Painel operacional exibido na tela do notebook."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import os
import socket
import sys

from dotenv import load_dotenv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from reconhecimento.operator_events import MAX_EVENT_BYTES, OperatorEvent, event_port_from_env


COLORS = {
    "authorized": ("#143D2B", "#55D98B", "AUTORIZADO"),
    "denied": ("#431F25", "#FF6B75", "NEGADO"),
    "unknown": ("#431F25", "#FF6B75", "NÃO IDENTIFICADO"),
    "error": ("#493719", "#F7C948", "NÃO VALIDADO"),
}

REASON_LABELS = {
    "DUPLICATE_ENTRY": "Entrada já registrada",
    "DUPLICATE_EXIT": "Sem entrada em aberto",
    "CAPACITY_FULL": "Capacidade máxima atingida",
    "CLIENT_INACTIVE": "Cliente inativo",
    "GUEST_NOT_ELIGIBLE": "Convidado não elegível",
    "CLIENT_NOT_ELIGIBLE": "Cliente não elegível",
    "NOT_AUTHORIZED": "Acesso não autorizado",
    "UNKNOWN_FACE": "Pessoa não identificada",
    "SERVICE_UNAVAILABLE": "Serviço indisponível",
    "DB_UNAVAILABLE": "Banco de dados indisponível",
}


def operator_display_index_from_env() -> int:
    raw = os.getenv("OPERATOR_DISPLAY_INDEX", "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"OPERATOR_DISPLAY_INDEX inválido: {raw}") from exc
    if value < 0:
        raise ValueError(f"OPERATOR_DISPLAY_INDEX inválido: {raw}")
    return value


class StationCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("stationCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(10)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.status = QLabel("AGUARDANDO")
        self.status.setObjectName("cardStatus")
        self.person = QLabel("Nenhuma leitura realizada")
        self.person.setObjectName("cardPerson")
        self.person.setWordWrap(True)
        self.detail = QLabel("Estação pronta")
        self.detail.setObjectName("cardDetail")
        self.detail.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addStretch()
        layout.addWidget(self.status)
        layout.addWidget(self.person)
        layout.addWidget(self.detail)
        layout.addStretch()

    def update_event(self, event: OperatorEvent) -> None:
        background, accent, label = COLORS[event.state]
        self.setStyleSheet(
            f"QFrame#stationCard {{ background:{background}; border:2px solid {accent}; border-radius:18px; }}"
        )
        self.status.setText(label)
        self.status.setStyleSheet(f"color:{accent}; font-size:26px; font-weight:800;")
        self.person.setText(event.name or "Pessoa não identificada")
        self.detail.setText(REASON_LABELS.get(event.reason or "", event.reason or "Decisão registrada"))


class OperatorDashboard(QMainWindow):
    def __init__(self, display_index: int, port: int):
        super().__init__()
        self.setWindowTitle("ISP Evolution — Painel de acesso")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self._history: deque[OperatorEvent] = deque(maxlen=12)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", port))
        self._socket.setblocking(False)
        self._build_ui()
        screens = QApplication.screens()
        selected = screens[display_index] if display_index < len(screens) else screens[0]
        self.setGeometry(selected.geometry())
        self.winId()
        self.windowHandle().setScreen(selected)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_events)
        self._timer.start(80)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(34, 28, 34, 30)
        layout.setSpacing(22)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        eyebrow = QLabel("SALA VIP • OPERAÇÃO")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Controle de acesso em tempo real")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Acompanhe as autorizações e negativas das duas estações.")
        subtitle.setObjectName("subtitle")
        titles.addWidget(eyebrow)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        live = QLabel("● AO VIVO")
        live.setObjectName("live")
        header.addWidget(live, alignment=Qt.AlignTop)
        layout.addLayout(header)

        cards = QGridLayout()
        cards.setSpacing(18)
        self._entry_card = StationCard("ENTRADA")
        self._exit_card = StationCard("SAÍDA")
        cards.addWidget(self._entry_card, 0, 0)
        cards.addWidget(self._exit_card, 0, 1)
        layout.addLayout(cards, stretch=2)

        history_title = QLabel("ÚLTIMOS ACESSOS")
        history_title.setObjectName("historyTitle")
        layout.addWidget(history_title)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["HORÁRIO", "MOVIMENTO", "PESSOA", "RESULTADO"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(2, self._table.horizontalHeader().Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._table, stretch=2)

        self.setStyleSheet("""
            QWidget#root { background:#07151F; color:#F4F8FB; }
            QLabel#eyebrow { color:#47D7FF; font-size:13px; font-weight:700; letter-spacing:3px; }
            QLabel#pageTitle { color:#FFFFFF; font-size:34px; font-weight:800; }
            QLabel#subtitle { color:#9DB0BD; font-size:15px; }
            QLabel#live { color:#55D98B; background:#123226; border-radius:14px; padding:8px 14px; font-weight:800; }
            QFrame#stationCard { background:#10232E; border:1px solid #28404D; border-radius:18px; }
            QLabel#cardTitle { color:#9DB0BD; font-size:15px; font-weight:700; letter-spacing:3px; }
            QLabel#cardStatus { color:#9DB0BD; font-size:26px; font-weight:800; }
            QLabel#cardPerson { color:#FFFFFF; font-size:28px; font-weight:800; }
            QLabel#cardDetail { color:#C0CDD5; font-size:15px; }
            QLabel#historyTitle { color:#9DB0BD; font-size:13px; font-weight:700; letter-spacing:2px; }
            QTableWidget { background:#0D202B; color:#ECF3F7; border:1px solid #28404D; border-radius:12px; gridline-color:#203744; font-size:14px; }
            QHeaderView::section { background:#132B37; color:#9DB0BD; border:0; padding:10px; font-weight:700; }
        """)

    def _poll_events(self) -> None:
        while True:
            try:
                payload, _address = self._socket.recvfrom(MAX_EVENT_BYTES + 1)
            except BlockingIOError:
                return
            except OSError as exc:
                print(f"[OPERATOR] Falha ao receber evento: {exc}")
                return
            try:
                self._apply_event(OperatorEvent.from_bytes(payload))
            except (ValueError, UnicodeError) as exc:
                print(f"[OPERATOR] Evento ignorado: {exc}")

    def _apply_event(self, event: OperatorEvent) -> None:
        card = self._entry_card if event.direction == "ENTRY" else self._exit_card
        card.update_event(event)
        self._history.appendleft(event)
        self._table.setRowCount(len(self._history))
        for row, item in enumerate(self._history):
            try:
                moment = datetime.fromisoformat(item.occurred_at).astimezone().strftime("%H:%M:%S")
            except ValueError:
                moment = "--:--:--"
            result = COLORS[item.state][2]
            values = [moment, "Entrada" if item.direction == "ENTRY" else "Saída", item.name or "Não identificado", result]
            for column, value in enumerate(values):
                self._table.setItem(row, column, QTableWidgetItem(value))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._socket.close()
        event.accept()


def main() -> None:
    load_dotenv()
    try:
        display_index = operator_display_index_from_env()
        port = event_port_from_env()
    except ValueError as exc:
        print(f"[OPERATOR] {exc}")
        raise SystemExit(2) from None
    app = QApplication(sys.argv)
    dashboard = OperatorDashboard(display_index, port)
    dashboard.showFullScreen()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
