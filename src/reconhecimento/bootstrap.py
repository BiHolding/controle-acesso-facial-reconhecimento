"""Bootstrap de runtime para ordem segura de importacao no Windows."""

from __future__ import annotations

import importlib
import os


def ensure_onnxruntime_loaded() -> None:
    """Load onnxruntime before PyQt5 on Windows to avoid DLL init failures."""
    if os.name != "nt":
        return
    importlib.import_module("onnxruntime")
