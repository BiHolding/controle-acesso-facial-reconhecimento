import os

import cv2


class Camera:

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        print(f"[CAMERA] index={camera_index}")
        if os.name == "nt":
            print("[CAMERA] backend=CAP_DSHOW")
            self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            print("[CAMERA] backend=default")
            self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            print("[CAMERA] opened=false")
            self.capture.release()
            raise RuntimeError(
                f"[CAMERA] Nao foi possivel abrir a camera no index {camera_index}."
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # evita frames antigos acumulados

        print("[CAMERA] opened=true")
        print(f"[CAMERA] width={self.capture.get(cv2.CAP_PROP_FRAME_WIDTH):g}")
        print(f"[CAMERA] height={self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT):g}")
        print(f"[CAMERA] fps={self.capture.get(cv2.CAP_PROP_FPS):g}")

    def read(self):
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError("Não foi possível capturar o frame.")
        return frame

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()
