import numpy as np
import cv2
from pathlib import Path
from insightface.model_zoo.arcface_onnx import ArcFaceONNX


class FaceEmbedder:
    _TEMPLATE = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self) -> None:
        model_file = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
        self.model = ArcFaceONNX(model_file=str(model_file))
        self.model.prepare(ctx_id=0)

    def generate(self, face, frame=None) -> np.ndarray:
        """Retorna embedding L2-normalizado de 512 dimensões a partir de uma face detectada."""
        if frame is None:
            raise ValueError("Frame original é obrigatório para gerar embedding facial.")

        keypoints = np.asarray(getattr(face, 'kps', None), dtype=np.float32)
        if keypoints.shape != (5, 2):
            raise ValueError("Keypoints faciais inválidos para alinhamento.")

        matrix, _ = cv2.estimateAffinePartial2D(keypoints, self._TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("Não foi possível alinhar a face detectada.")

        aligned = cv2.warpAffine(frame, matrix, (112, 112), borderValue=0.0)
        embedding = np.asarray(self.model.get_feat(aligned)[0], dtype=np.float32)

        norm = float(np.linalg.norm(embedding))
        if embedding.shape != (512,):
            raise ValueError("Embedding inválido (dimensão diferente de 512).")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding inválido (valor não finito).")
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("Embedding inválido (norma zero).")

        return embedding / norm
