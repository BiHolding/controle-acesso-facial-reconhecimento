import math
import unittest

import numpy as np

from reconhecimento.recognition.matcher import InMemoryFaceIndex, normalize_embedding


def axis(index: int, scale: float = 1.0) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = scale
    return vector


class InMemoryFaceIndexTest(unittest.TestCase):
    def test_identical_embedding_matches(self) -> None:
        index = InMemoryFaceIndex(threshold=0.8)
        index.replace({"guest-1": axis(0)})

        match = index.match(axis(0))

        self.assertIsNotNone(match)
        self.assertEqual(match.guest_id, "guest-1")
        self.assertAlmostEqual(match.similarity, 1.0, places=6)

    def test_similarity_above_threshold_matches(self) -> None:
        query = axis(0, 0.9)
        query[1] = math.sqrt(1.0 - 0.9**2)
        index = InMemoryFaceIndex(threshold=0.8)
        index.replace({"guest-1": axis(0)})

        match = index.match(query)

        self.assertIsNotNone(match)
        self.assertAlmostEqual(match.similarity, 0.9, places=5)

    def test_similarity_below_threshold_does_not_match(self) -> None:
        index = InMemoryFaceIndex(threshold=0.8)
        index.replace({"guest-1": axis(0)})

        self.assertIsNone(index.match(axis(1)))

    def test_embeddings_are_normalized(self) -> None:
        normalized = normalize_embedding(axis(0, scale=25.0))

        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=6)

    def test_empty_index_does_not_match(self) -> None:
        index = InMemoryFaceIndex(threshold=0.8)

        self.assertIsNone(index.match(axis(0)))


if __name__ == "__main__":
    unittest.main()
