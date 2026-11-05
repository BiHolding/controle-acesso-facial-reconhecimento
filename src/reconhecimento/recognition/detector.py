import os

from reconhecimento.bootstrap import ensure_insightface_runtime

ensure_insightface_runtime()

from insightface.app import FaceAnalysis


class FaceDetector:

    def __init__(self):
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection"],
        )
        # Limiar levemente mais sensível reduz falhas em iluminação real; as
        # etapas de qualidade, consistência temporal e matching ainda impedem
        # que uma detecção fraca autorize alguém.
        try:
            threshold = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.45"))
        except ValueError:
            threshold = 0.45
        if not 0.30 <= threshold <= 0.80:
            threshold = 0.45
        # 320x320 mantém a detecção rápida em CPU para rostos próximos do totem.
        self.model.prepare(ctx_id=0, det_size=(320, 320), det_thresh=threshold)

    def detect(self, frame):
        """Retorna lista de faces detectadas no frame."""
        return self.model.get(frame)
