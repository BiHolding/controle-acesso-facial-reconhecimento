from insightface.app import FaceAnalysis


class FaceDetector:

    def __init__(self):
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection"],
        )
        # 320x320 é suficiente para tempo real na CPU e reduz latência vs 640x640
        self.model.prepare(ctx_id=0, det_size=(320, 320))

    def detect(self, frame):
        """Retorna lista de faces detectadas no frame."""
        return self.model.get(frame)
