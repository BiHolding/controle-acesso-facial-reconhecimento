import unittest
from unittest.mock import MagicMock

import numpy as np

from reconhecimento.api.client import RecognitionResult
from reconhecimento.display import RecognitionWorkerThread


class RecognitionWorkerThreadTest(unittest.TestCase):
    def test_calls_api_once_after_five_stable_samples(self) -> None:
        detector = MagicMock()
        detector.detect.return_value = [MagicMock()]
        embedder = MagicMock()
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        embedder.generate.return_value = embedding
        recognize_fn = MagicMock(return_value=RecognitionResult(
            recognized=True,
            allowed=True,
            reason="AUTHORIZED",
            guest_id="1",
            participant_type="guest",
            participant_id="1",
            name="Convidado Teste",
            similarity=0.9,
            direction="ENTRY",
        ))
        worker = RecognitionWorkerThread(
            detector=detector,
            embedder=embedder,
            guard=MagicMock(),
            recognize_fn=recognize_fn,
        )
        worker._quality = MagicMock()
        worker._quality.assess.return_value = MagicMock(acceptable=True)
        emitted = []
        worker.result_ready.connect(emitted.append)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(5):
            worker._process(frame)

        recognize_fn.assert_called_once()
        probe = recognize_fn.call_args.args[0]
        self.assertAlmostEqual(float(np.linalg.norm(probe)), 1.0, places=5)
        self.assertEqual("authorized", emitted[-1].state)
        self.assertEqual("ENTRY", emitted[-1].direction)

        worker._process(frame)
        recognize_fn.assert_called_once()

    def test_clears_samples_when_quality_drops(self) -> None:
        detector = MagicMock()
        detector.detect.return_value = [MagicMock()]
        embedder = MagicMock()
        embedder.generate.return_value = np.ones(512, dtype=np.float32)
        worker = RecognitionWorkerThread(detector, embedder, MagicMock(), MagicMock())
        worker._quality = MagicMock()
        worker._quality.assess.side_effect = [
            MagicMock(acceptable=True),
            MagicMock(acceptable=False, reason="TOO_DARK", message="Mais luz"),
        ]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        worker._process(frame)
        self.assertEqual(1, len(worker._embedding_samples))
        worker._process(frame)
        self.assertEqual(0, len(worker._embedding_samples))


if __name__ == "__main__":
    unittest.main()
