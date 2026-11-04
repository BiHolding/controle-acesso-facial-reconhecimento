"""Bootstrap de runtime para ordem segura de importacao no Windows."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace


def ensure_onnxruntime_loaded() -> None:
    """Load onnxruntime before PyQt5 on Windows to avoid DLL init failures."""
    if os.name != "nt":
        return
    importlib.import_module("onnxruntime")


def _application_control_blocked(exc: ImportError) -> bool:
    message = str(exc).casefold()
    return (
        "controle de aplicativo" in message and "bloqueou" in message
    ) or "application control" in message


def _install_runtime_only_onnx_stub() -> None:
    """Fornece apenas os metadados usados pelo InsightFace em inferência.

    A execução dos modelos continua sob responsabilidade do onnxruntime. O
    pacote onnx completo só seria necessário para editar ou converter modelos.
    """
    for module_name in tuple(sys.modules):
        if module_name == "onnx" or module_name.startswith("onnx."):
            sys.modules.pop(module_name, None)
        elif module_name == "ml_dtypes" or module_name.startswith("ml_dtypes."):
            sys.modules.pop(module_name, None)

    graph = SimpleNamespace(node=(), initializer=())
    model = SimpleNamespace(graph=graph)
    stub = ModuleType("onnx")
    stub.__dict__.update(
        {
            "__version__": "runtime-only",
            "load": lambda *_args, **_kwargs: model,
            "numpy_helper": SimpleNamespace(to_array=lambda value: value),
        }
    )
    sys.modules["onnx"] = stub


def ensure_insightface_runtime() -> None:
    """Prepara o InsightFace sem reduzir as proteções do Windows."""
    ensure_onnxruntime_loaded()
    if os.name == "nt":
        current = sys.modules.get("onnx")
        if getattr(current, "__version__", None) == "runtime-only":
            return
        _install_runtime_only_onnx_stub()
        print(
            "[RUNTIME] Inferência facial configurada via onnxruntime "
            "(modo compatível com o Controle de Aplicativos)."
        )
        return
    try:
        importlib.import_module("onnx")
    except ImportError as exc:
        if not _application_control_blocked(exc):
            raise
        _install_runtime_only_onnx_stub()
        print(
            "[RUNTIME] ONNX completo bloqueado pelo Controle de Aplicativos; "
            "usando inferência segura via onnxruntime."
        )
