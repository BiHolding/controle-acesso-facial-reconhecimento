from collections import Counter, deque


class RecognitionConfirmation:
    """
    Janela deslizante de N frames.
    Só confirma uma pessoa quando ela aparece em >50% dos últimos N frames.
    Evita falsos positivos por frames ruidosos.
    """

    def __init__(self, required_frames: int = 5):
        self.required_frames = required_frames
        self.history = deque(maxlen=required_frames)

    def update(self, person_id: str | None) -> str | None:
        """Retorna o person_id confirmado, ou None se ainda não há consenso."""
        self.history.append(person_id)

        if len(self.history) < self.required_frames:
            return None

        counts = Counter(self.history)
        person, count = counts.most_common(1)[0]

        if person is not None and count > self.required_frames // 2:
            return person

        return None
