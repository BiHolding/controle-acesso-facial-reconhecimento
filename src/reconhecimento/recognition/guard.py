import time


class AccessEventGuard:
    """
    Cooldown por pessoa: evita registrar o mesmo evento várias vezes seguidas.
    Exemplo: cooldown_seconds=10 → o mesmo person_id não gera novo evento por 10s.
    """

    def __init__(
        self,
        cooldown_seconds: float = 10.0,
        unknown_cooldown_seconds: float = 10.0,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.unknown_cooldown_seconds = unknown_cooldown_seconds
        self._last_events: dict[str, float] = {}
        self._last_unknown: float | None = None

    def can_register(self, person_id: str | None) -> bool:
        now = time.monotonic()

        if person_id is None:
            if self._last_unknown is None or (now - self._last_unknown) >= self.unknown_cooldown_seconds:
                self._last_unknown = now
                return True
            return False

        last = self._last_events.get(person_id)
        if last is None or (now - last) >= self.cooldown_seconds:
            self._last_events[person_id] = now
            return True

        return False
