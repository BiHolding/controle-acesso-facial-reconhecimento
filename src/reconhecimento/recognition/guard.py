import time
from collections.abc import Callable


class AccessEventGuard:
    """
    Cooldown por pessoa: evita registrar o mesmo evento várias vezes seguidas.
    Exemplo: cooldown_seconds=10 → o mesmo guest_id não gera novo evento por 10s.
    """

    def __init__(
        self,
        cooldown_seconds: float = 10.0,
        unknown_cooldown_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if cooldown_seconds < 0 or unknown_cooldown_seconds < 0:
            raise ValueError("Cooldown não pode ser negativo")
        self.cooldown_seconds = cooldown_seconds
        self.unknown_cooldown_seconds = unknown_cooldown_seconds
        self._clock = clock
        self._last_events: dict[str, float] = {}
        self._last_unknown: float | None = None

    def can_register(self, guest_id: str | None) -> bool:
        now = self._clock()

        if guest_id is None:
            if self._last_unknown is None or (now - self._last_unknown) >= self.unknown_cooldown_seconds:
                self._last_unknown = now
                return True
            return False

        last = self._last_events.get(guest_id)
        if last is None or (now - last) >= self.cooldown_seconds:
            self._last_events[guest_id] = now
            return True

        return False
