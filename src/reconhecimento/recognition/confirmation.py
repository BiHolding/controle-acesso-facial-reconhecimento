from collections import deque


class RecognitionConfirmation:
    """
    Janela deslizante de N observações.
    Só confirma uma pessoa quando as N observações concordam.
    Evita falsos positivos por frames ruidosos.
    """

    def __init__(self, required_frames: int = 5):
        if required_frames < 1:
            raise ValueError("required_frames deve ser positivo")
        self.required_frames = required_frames
        self.history = deque(maxlen=required_frames)

    def update(self, guest_id: str | None) -> str | None:
        """Retorna o guest_id confirmado, ou None se ainda não há consenso."""
        self.history.append(guest_id)

        if len(self.history) < self.required_frames:
            return None

        person = self.history[0]
        if person is not None and all(item == person for item in self.history):
            return person

        return None

    def reset(self) -> None:
        self.history.clear()
