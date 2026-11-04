import sys
import unittest
from unittest.mock import patch

from reconhecimento.bootstrap import (
    _application_control_blocked,
    _install_runtime_only_onnx_stub,
    ensure_onnxruntime_loaded,
    ensure_insightface_runtime,
)

# On Windows, PyQt5 loaded before onnxruntime can cause DLL initialization
# failure. Load onnxruntime first during test discovery.
ensure_onnxruntime_loaded()


class BootstrapTest(unittest.TestCase):
    @patch("reconhecimento.bootstrap.importlib.import_module")
    @patch("reconhecimento.bootstrap.os.name", "nt")
    def test_windows_preloads_onnxruntime(self, import_module) -> None:
        ensure_onnxruntime_loaded()

        import_module.assert_called_once_with("onnxruntime")

    @patch("reconhecimento.bootstrap.importlib.import_module")
    @patch("reconhecimento.bootstrap.os.name", "posix")
    def test_other_platforms_skip_preload(self, import_module) -> None:
        ensure_onnxruntime_loaded()

        import_module.assert_not_called()

    def test_detects_windows_application_control_error(self) -> None:
        error = ImportError(
            "DLL load failed: Uma política de Controle de Aplicativo bloqueou este arquivo."
        )

        self.assertTrue(_application_control_blocked(error))

    def test_runtime_stub_exposes_insightface_metadata_contract(self) -> None:
        previous = sys.modules.get("onnx")
        try:
            _install_runtime_only_onnx_stub()
            import onnx

            model = onnx.load("modelo.onnx")
            self.assertEqual((), model.graph.node)
            self.assertTrue(hasattr(onnx, "numpy_helper"))
        finally:
            sys.modules.pop("onnx", None)
            if previous is not None:
                sys.modules["onnx"] = previous

    @patch("reconhecimento.bootstrap.ensure_onnxruntime_loaded")
    @patch("reconhecimento.bootstrap.os.name", "nt")
    def test_windows_uses_runtime_only_onnx_without_loading_blocked_dll(
        self, ensure_runtime
    ) -> None:
        previous = sys.modules.pop("onnx", None)
        try:
            with patch("reconhecimento.bootstrap.importlib.import_module") as importer:
                ensure_insightface_runtime()

            ensure_runtime.assert_called_once_with()
            importer.assert_not_called()
            self.assertEqual("runtime-only", sys.modules["onnx"].__version__)
        finally:
            sys.modules.pop("onnx", None)
            if previous is not None:
                sys.modules["onnx"] = previous


if __name__ == "__main__":
    unittest.main()
