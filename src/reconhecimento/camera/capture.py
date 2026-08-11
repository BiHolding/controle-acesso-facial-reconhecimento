import cv2


class Camera:

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            raise RuntimeError("Não foi possível abrir a câmera.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # evita frames antigos acumulados

    def read(self):
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError("Não foi possível capturar o frame.")
        return frame

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()
