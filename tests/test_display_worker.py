import unittest
from unittest.mock import MagicMock

import numpy as np

from reconhecimento.api.client import RecognitionResult
from reconhecimento.display import RecognitionWorkerThread, _robust_embedding_fusion


def unit_embedding(axis: int) -> np.ndarray:
    value = np.zeros(512, dtype=np.float32)
    value[axis] = 1.0
    return value


class RecognitionWorkerThreadTest(unittest.TestCase):
    def _authorized_result(self) -> RecognitionResult:
        return RecognitionResult(
            recognized=True,
            allowed=True,
            reason="AUTHORIZED",
            guest_id="1",
            participant_type="guest",
            participant_id="1",
            name="Convidado Teste",
            similarity=0.9,
            direction="ENTRY",
        )

    def test_two_highly_consistent_samples_use_fast_path(self) -> None:
        detector = MagicMock()
        detector.detect.return_value = [MagicMock()]
        embedder = MagicMock()
        embedder.generate.return_value = unit_embedding(0)
        recognize_fn = MagicMock(return_value=self._authorized_result())
        worker = RecognitionWorkerThread(detector, embedder, MagicMock(), recognize_fn)
        worker._quality = MagicMock()
        worker._quality.assess.return_value = MagicMock(acceptable=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        worker._process(frame)
        recognize_fn.assert_not_called()
        worker._process(frame)

        recognize_fn.assert_called_once()
        self.assertAlmostEqual(float(np.linalg.norm(recognize_fn.call_args.args[0])), 1.0, places=6)

    def test_third_sample_discards_one_divergent_outlier(self) -> None:
        detector = MagicMock()
        detector.detect.return_value = [MagicMock()]
        embedder = MagicMock()
        embedder.generate.side_effect = [unit_embedding(0), unit_embedding(1), unit_embedding(0)]
        recognize_fn = MagicMock(return_value=self._authorized_result())
        worker = RecognitionWorkerThread(detector, embedder, MagicMock(), recognize_fn)
        worker._quality = MagicMock()
        worker._quality.assess.return_value = MagicMock(acceptable=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(3):
            worker._process(frame)

        recognize_fn.assert_called_once()
        np.testing.assert_allclose(recognize_fn.call_args.args[0], unit_embedding(0), atol=1e-6)

    def test_three_inconsistent_samples_are_rejected_without_api_call(self) -> None:
        detector = MagicMock()
        detector.detect.return_value = [MagicMock()]
        embedder = MagicMock()
        embedder.generate.side_effect = [unit_embedding(0), unit_embedding(1), unit_embedding(2)]
        recognize_fn = MagicMock()
        worker = RecognitionWorkerThread(detector, embedder, MagicMock(), recognize_fn)
        worker._quality = MagicMock()
        worker._quality.assess.return_value = MagicMock(acceptable=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(3):
            worker._process(frame)

        recognize_fn.assert_not_called()
        self.assertEqual(0, len(worker._embedding_samples))

    def test_calls_api_once_after_three_stable_samples(self) -> None:
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

        for _ in range(3):
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

    def test_keeps_the_identified_name_for_the_operator_when_access_is_denied(self) -> None:
        detector = MagicMock()
        detector.detect.return_value = [MagicMock()]
        embedder = MagicMock()
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        embedder.generate.return_value = embedding
        recognize_fn = MagicMock(return_value=RecognitionResult(
            recognized=True,
            allowed=False,
            reason="DUPLICATE_ENTRY",
            guest_id="2",
            participant_type="guest",
            participant_id="2",
            name="Convidada Identificada",
            similarity=0.91,
            direction="ENTRY",
        ))
        worker = RecognitionWorkerThread(detector, embedder, MagicMock(), recognize_fn)
        worker._quality = MagicMock()
        worker._quality.assess.return_value = MagicMock(acceptable=True)
        emitted = []
        worker.result_ready.connect(emitted.append)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(3):
            worker._process(frame)

        self.assertEqual("denied", emitted[-1].state)
        self.assertEqual("Convidada Identificada", emitted[-1].name)


class RobustEmbeddingFusionTest(unittest.TestCase):
    def test_returns_none_without_two_consistent_samples(self) -> None:
        self.assertIsNone(_robust_embedding_fusion((unit_embedding(0), unit_embedding(1), unit_embedding(2))))

    def test_rejects_invalid_embedding_shape(self) -> None:
        with self.assertRaises(ValueError):
            _robust_embedding_fusion((np.ones(16, dtype=np.float32), unit_embedding(0)))


if __name__ == "__main__":
    unittest.main()
