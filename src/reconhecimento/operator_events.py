"""Canal local e efêmero entre as estações e o painel do operador."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import socket


TERMINAL_STATES = frozenset({"authorized", "denied", "unknown", "error"})
VALID_DIRECTIONS = frozenset({"ENTRY", "EXIT"})
DEFAULT_EVENT_PORT = 37651
MAX_EVENT_BYTES = 4096


@dataclass(frozen=True)
class OperatorEvent:
    station: int
    direction: str
    state: str
    name: str | None
    reason: str | None
    occurred_at: str

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OperatorEvent":
        if len(payload) > MAX_EVENT_BYTES:
            raise ValueError("Evento operacional excede o tamanho permitido")
        raw = json.loads(payload.decode("utf-8"))
        if not isinstance(raw, dict) or set(raw) != {
            "station", "direction", "state", "name", "reason", "occurred_at",
        }:
            raise ValueError("Evento operacional fora do contrato")
        station = raw["station"]
        direction = raw["direction"]
        state = raw["state"]
        name = raw["name"]
        reason = raw["reason"]
        occurred_at = raw["occurred_at"]
        if type(station) is not int or station not in {1, 2}:
            raise ValueError("Estação operacional inválida")
        if direction not in VALID_DIRECTIONS or state not in TERMINAL_STATES:
            raise ValueError("Resultado operacional inválido")
        if name is not None and (not isinstance(name, str) or len(name) > 160):
            raise ValueError("Nome operacional inválido")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 80):
            raise ValueError("Motivo operacional inválido")
        if not isinstance(occurred_at, str) or len(occurred_at) > 40:
            raise ValueError("Data operacional inválida")
        return cls(station, direction, state, name, reason, occurred_at)


def event_port_from_env() -> int:
    raw = os.getenv("OPERATOR_EVENT_PORT", str(DEFAULT_EVENT_PORT))
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"OPERATOR_EVENT_PORT inválida: {raw}") from exc
    if not 1024 <= port <= 65535:
        raise ValueError(f"OPERATOR_EVENT_PORT inválida: {raw}")
    return port


def publish_operator_result(result, fallback_direction: str) -> None:
    """Publica apenas decisões finais via UDP no próprio notebook."""
    if result.state not in TERMINAL_STATES:
        return
    try:
        station = int(os.getenv("STATION_NUMBER", "1"))
        direction = result.direction or fallback_direction
        event = OperatorEvent(
            station=station,
            direction=direction,
            state=result.state,
            name=(result.name or "").strip() or None,
            reason=(result.reason or "").strip() or None,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        payload = event.to_bytes()
        if len(payload) > MAX_EVENT_BYTES:
            raise ValueError("Evento operacional excede o tamanho permitido")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
            channel.sendto(payload, ("127.0.0.1", event_port_from_env()))
    except (OSError, ValueError, TypeError) as exc:
        # O painel é observacional: uma falha nele nunca pode bloquear o acesso.
        print(f"[OPERATOR] Não foi possível atualizar o painel: {exc}")
