"""Project-local Python startup customizations."""

from reconhecimento.bootstrap import ensure_onnxruntime_loaded

# On Windows, PyQt5 loaded before onnxruntime can cause DLL initialization
# failure. Load onnxruntime first.
ensure_onnxruntime_loaded()
