"""Índice de referência para uso no serviço servidor, não no PC da porta."""

from dataclasses import dataclass

import numpy as np

EMBEDDING_DIMENSION = 512


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"Embedding deve ter {EMBEDDING_DIMENSION} dimensões")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Embedding deve conter somente valores finitos")

    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("Embedding não pode ter norma zero")
    return vector / norm


@dataclass(frozen=True)
class FaceMatch:
    guest_id: str
    similarity: float


class InMemoryFaceIndex:
    """Busca cosseno exata adequada a um conjunto pequeno/médio de Guests."""

    def __init__(self, threshold: float) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Threshold deve estar entre -1 e 1")
        self.threshold = threshold
        self._snapshot: tuple[tuple[str, ...], np.ndarray] = (
            (),
            np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32),
        )

    def replace(self, references: dict[str, np.ndarray]) -> None:
        guest_ids = list(references)
        if not guest_ids:
            self._snapshot = ((), np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32))
            return

        embeddings = np.stack(
            [normalize_embedding(references[guest_id]) for guest_id in guest_ids]
        )
        self._snapshot = (tuple(guest_ids), embeddings)

    def match(self, embedding: np.ndarray) -> FaceMatch | None:
        guest_ids, embeddings = self._snapshot
        if not guest_ids:
            return None

        query = normalize_embedding(embedding)
        similarities = embeddings @ query
        best_index = int(np.argmax(similarities))
        similarity = float(similarities[best_index])
        if similarity < self.threshold:
            return None
        return FaceMatch(guest_ids[best_index], similarity)
