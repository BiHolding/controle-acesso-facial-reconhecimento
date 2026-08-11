import numpy as np


class FaceEmbedder:

    @staticmethod
    def generate(face) -> np.ndarray:
        """Retorna embedding L2-normalizado de 512 dimensões a partir de uma face detectada."""
        embedding = face.embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Embedding inválido (norma zero).")
        return embedding / norm
