import unittest

from reconhecimento.recognition.confirmation import RecognitionConfirmation
from reconhecimento.recognition.guard import AccessEventGuard


class RecognitionConfirmationTest(unittest.TestCase):
    def test_requires_all_configured_observations(self) -> None:
        confirmation = RecognitionConfirmation(required_frames=5)

        for _ in range(4):
            self.assertIsNone(confirmation.update("guest-1"))
        self.assertEqual(confirmation.update("guest-1"), "guest-1")

    def test_alternating_identities_do_not_confirm(self) -> None:
        confirmation = RecognitionConfirmation(required_frames=5)

        results = [confirmation.update(identity) for identity in ["a", "b", "a", "b", "a"]]

        self.assertTrue(all(result is None for result in results))

    def test_unknown_does_not_confirm(self) -> None:
        confirmation = RecognitionConfirmation(required_frames=3)

        self.assertIsNone(confirmation.update(None))
        self.assertIsNone(confirmation.update(None))
        self.assertIsNone(confirmation.update(None))


class AccessEventGuardTest(unittest.TestCase):
    def test_cooldown_is_independent_per_guest(self) -> None:
        now = [100.0]
        guard = AccessEventGuard(10.0, 10.0, clock=lambda: now[0])

        self.assertTrue(guard.can_register("a"))
        self.assertFalse(guard.can_register("a"))
        self.assertTrue(guard.can_register("b"))
        now[0] += 10.0
        self.assertTrue(guard.can_register("a"))

    def test_unknown_has_its_own_cooldown(self) -> None:
        now = [100.0]
        guard = AccessEventGuard(10.0, 5.0, clock=lambda: now[0])

        self.assertTrue(guard.can_register(None))
        self.assertFalse(guard.can_register(None))
        now[0] += 5.0
        self.assertTrue(guard.can_register(None))


if __name__ == "__main__":
    unittest.main()
