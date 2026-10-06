"""Ciclo de vida do painel observacional na operação com uma estação."""

from __future__ import annotations

import os
import subprocess
import sys


def operator_dashboard_enabled() -> bool:
    return os.getenv("OPERATOR_DASHBOARD_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def start_operator_dashboard() -> subprocess.Popen | None:
    if not operator_dashboard_enabled():
        return None
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "reconhecimento.operator_dashboard"],
            env=os.environ.copy(),
        )
        print(
            f"[OPERATOR] Painel iniciado no monitor "
            f"{os.getenv('OPERATOR_DISPLAY_INDEX', '0')}."
        )
        return process
    except OSError as exc:
        # O painel é observacional e nunca deve impedir o controle de acesso.
        print(f"[OPERATOR] Não foi possível iniciar o painel: {exc}")
        return None


def stop_operator_dashboard(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
