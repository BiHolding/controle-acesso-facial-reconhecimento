import argparse
import os
from collections.abc import Sequence


def _non_negative_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("o indice deve ser um numero inteiro") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("o indice nao pode ser negativo")
    return index


def camera_index_from_env() -> int:
    raw_value = os.getenv("CAMERA_INDEX", "0")
    try:
        camera_index = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"[CAMERA] Invalid CAMERA_INDEX: {raw_value}") from exc
    if camera_index < 0:
        raise ValueError(f"[CAMERA] Invalid CAMERA_INDEX: {raw_value}")
    return camera_index


def display_index_from_env() -> int:
    raw_value = os.getenv("DISPLAY_INDEX", "0")
    try:
        display_index = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"[DISPLAY] Invalid DISPLAY_INDEX: {raw_value}") from exc
    if display_index < 0:
        raise ValueError(f"[DISPLAY] Invalid DISPLAY_INDEX: {raw_value}")
    return display_index


def _seconds_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"[CONFIG] Invalid {name}: {raw_value}") from exc
    if seconds < 0:
        raise ValueError(f"[CONFIG] Invalid {name}: {raw_value}")
    return seconds


def result_timeout_seconds_from_env() -> float:
    return _seconds_from_env("ACCESS_RESULT_TIMEOUT_SECONDS", 1.0)


def event_cooldowns_from_env() -> tuple[float, float]:
    return (
        _seconds_from_env("ACCESS_EVENT_COOLDOWN_SECONDS", 1.0),
        _seconds_from_env("UNKNOWN_EVENT_COOLDOWN_SECONDS", 1.0),
    )


def device_indices_from_args(argv: Sequence[str] | None = None) -> tuple[int, int]:
    parser = argparse.ArgumentParser(
        prog="reconhecer",
        description="Reconhecimento facial da Sala VIP ISP Evolution",
    )
    parser.add_argument(
        "-c", "--camera", "--camera-index",
        dest="camera_index",
        type=_non_negative_index,
        help="indice da webcam (padrao: CAMERA_INDEX do .env ou 0)",
    )
    parser.add_argument(
        "-m", "--monitor", "--display-index",
        dest="display_index",
        type=_non_negative_index,
        help="indice do monitor (padrao: DISPLAY_INDEX do .env ou 0)",
    )
    arguments = parser.parse_args(argv)

    camera_index = arguments.camera_index
    if camera_index is None:
        camera_index = camera_index_from_env()

    display_index = arguments.display_index
    if display_index is None:
        display_index = display_index_from_env()

    return camera_index, display_index


def access_direction_from_env() -> str:
    direction = os.getenv("ACCESS_DIRECTION", "ENTRY").strip().upper()
    if direction not in {"ENTRY", "EXIT"}:
        raise ValueError(f"[ACCESS] Invalid ACCESS_DIRECTION: {direction}")
    return direction
