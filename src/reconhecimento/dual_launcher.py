"""Inicializa duas estações faciais isoladas no mesmo computador."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
import re
import subprocess
import sys
import time

from dotenv import load_dotenv


@dataclass(frozen=True)
class StationConfig:
    number: int
    camera_index: int
    display_index: int
    device_id: int
    direction: str
    access_point: str


_ACCESS_POINT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _non_negative_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"[DUAL] {name} deve ser um número inteiro: {raw_value}") from exc
    if value < 0:
        raise ValueError(f"[DUAL] {name} não pode ser negativo: {raw_value}")
    return value


def _direction(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().upper()
    if value not in {"ENTRY", "EXIT"}:
        raise ValueError(f"[DUAL] {name} deve ser ENTRY ou EXIT: {value}")
    return value


def _access_point(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not _ACCESS_POINT_PATTERN.fullmatch(value):
        raise ValueError(f"[DUAL] {name} inválido: {value}")
    return value


def station_configs_from_env() -> tuple[StationConfig, StationConfig]:
    stations = (
        StationConfig(
            number=1,
            camera_index=_non_negative_int("STATION_1_CAMERA_INDEX", 0),
            display_index=_non_negative_int("STATION_1_DISPLAY_INDEX", 0),
            device_id=_non_negative_int("STATION_1_DEVICE_ID", 0),
            direction=_direction("STATION_1_DIRECTION", "ENTRY"),
            access_point=_access_point("STATION_1_ACCESS_POINT", "VIP_ENTRANCE_01"),
        ),
        StationConfig(
            number=2,
            camera_index=_non_negative_int("STATION_2_CAMERA_INDEX", 1),
            display_index=_non_negative_int("STATION_2_DISPLAY_INDEX", 1),
            device_id=_non_negative_int("STATION_2_DEVICE_ID", 0),
            direction=_direction("STATION_2_DIRECTION", "EXIT"),
            access_point=_access_point("STATION_2_ACCESS_POINT", "VIP_EXIT_01"),
        ),
    )
    if stations[0].camera_index == stations[1].camera_index:
        raise ValueError("[DUAL] As duas estações não podem usar a mesma webcam.")
    if stations[0].display_index == stations[1].display_index:
        raise ValueError("[DUAL] As duas estações não podem usar o mesmo monitor.")
    if stations[0].direction == stations[1].direction:
        raise ValueError("[DUAL] Configure uma estação como ENTRY e a outra como EXIT.")
    nonzero_device_ids = [station.device_id for station in stations if station.device_id]
    if len(nonzero_device_ids) != len(set(nonzero_device_ids)):
        raise ValueError("[DUAL] Cada estação deve ter um DEVICE_ID diferente.")
    return stations


def connected_display_count() -> int | None:
    """Retorna o número de monitores no Windows; None em outras plataformas."""
    if os.name != "nt":
        return None
    return int(ctypes.windll.user32.GetSystemMetrics(80))  # SM_CMONITORS


def validate_displays(stations: tuple[StationConfig, ...], display_count: int | None) -> None:
    if display_count is None:
        return
    if display_count < 2:
        raise RuntimeError(
            "[DUAL] Apenas um monitor foi detectado. Conecte e habilite o segundo monitor no Windows."
        )
    unavailable = [
        station.display_index
        for station in stations
        if station.display_index >= display_count
    ]
    if unavailable:
        raise RuntimeError(
            f"[DUAL] Monitor {unavailable[0]} não está disponível; "
            f"o Windows detectou {display_count} monitor(es)."
        )


def child_environment(station: StationConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CAMERA_INDEX"] = str(station.camera_index)
    environment["DISPLAY_INDEX"] = str(station.display_index)
    environment["DEVICE_ID"] = str(station.device_id)
    environment["STATION_NUMBER"] = str(station.number)
    environment["ACCESS_DIRECTION"] = station.direction
    environment["ACCESS_POINT"] = station.access_point
    if station.number != 1:
        # Evita que duas instâncias disputem o enrollment da mesma fotografia.
        environment["FACE_ENROLLMENT_ENABLED"] = "false"
    return environment


def main() -> None:
    load_dotenv()
    try:
        stations = station_configs_from_env()
        validate_displays(stations, connected_display_count())
    except (ValueError, RuntimeError) as exc:
        print(exc)
        raise SystemExit(2) from None

    processes: list[tuple[StationConfig, subprocess.Popen]] = []
    try:
        for station in stations:
            print(
                f"[DUAL] Iniciando estação {station.number}: "
                f"webcam={station.camera_index}, monitor={station.display_index}, "
                f"função={station.direction}, ponto={station.access_point}, "
                f"device={station.device_id}"
            )
            process = subprocess.Popen(
                [sys.executable, "-m", "reconhecimento.recognize"],
                env=child_environment(station),
            )
            processes.append((station, process))

        while processes:
            for station, process in tuple(processes):
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"[DUAL] Estação {station.number} encerrada (código {exit_code}).")
                    processes.remove((station, process))
            if processes:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("[DUAL] Encerrando as duas estações...")
    except OSError as exc:
        print(f"[DUAL] Não foi possível iniciar uma das estações: {exc}")
        raise SystemExit(3) from None
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
