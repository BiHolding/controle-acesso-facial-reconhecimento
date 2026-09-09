import unittest
from unittest.mock import patch

from reconhecimento.bootstrap import ensure_onnxruntime_loaded

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


if __name__ == "__main__":
    unittest.main()
