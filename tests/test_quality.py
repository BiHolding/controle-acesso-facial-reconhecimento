import unittest
from types import SimpleNamespace

import numpy as np

from reconhecimento.recognition.quality import FaceQualityAssessor


class FaceQualityAssessorTest(unittest.TestCase):
    def setUp(self):
        self.assessor = FaceQualityAssessor()
        self.face = SimpleNamespace(bbox=np.array([180, 100, 460, 420], dtype=np.float32))

    @staticmethod
    def detailed_frame() -> np.ndarray:
        yy, xx = np.indices((480, 640))
        pattern = np.where((xx // 8 + yy // 8) % 2 == 0, 75, 185).astype(np.uint8)
        return np.dstack([pattern, pattern, pattern])

    def test_accepts_a_centered_well_lit_sharp_face(self):
        result = self.assessor.assess(self.detailed_frame(), self.face)

        self.assertTrue(result.acceptable)
        self.assertEqual("OK", result.reason)

    def test_guides_the_person_when_the_face_is_too_dark(self):
        frame = np.full((480, 640, 3), 25, dtype=np.uint8)

        result = self.assessor.assess(frame, self.face)

        self.assertFalse(result.acceptable)
        self.assertEqual("TOO_DARK", result.reason)
        self.assertIn("luz", result.message.lower())

    def test_guides_the_person_when_the_face_is_overexposed(self):
        frame = np.full((480, 640, 3), 240, dtype=np.uint8)

        result = self.assessor.assess(frame, self.face)

        self.assertFalse(result.acceptable)
        self.assertEqual("TOO_BRIGHT", result.reason)
        self.assertIn("muita luz", result.message.lower())

    def test_guides_the_person_when_contrast_is_too_low(self):
        yy, xx = np.indices((480, 640))
        pattern = np.where((xx + yy) % 2 == 0, 120, 130).astype(np.uint8)
        frame = np.dstack([pattern, pattern, pattern])

        result = self.assessor.assess(frame, self.face)

        self.assertFalse(result.acceptable)
        self.assertEqual("LOW_CONTRAST", result.reason)
        self.assertIn("iluminação", result.message.lower())

    def test_guides_the_person_when_the_face_is_too_far(self):
        distant_face = SimpleNamespace(bbox=np.array([285, 190, 355, 290], dtype=np.float32))

        result = self.assessor.assess(self.detailed_frame(), distant_face)

        self.assertEqual("FACE_TOO_FAR", result.reason)
        self.assertIn("aproxime", result.message.lower())

    def test_guides_the_person_when_the_face_is_off_center(self):
        off_center = SimpleNamespace(bbox=np.array([0, 100, 150, 420], dtype=np.float32))

        result = self.assessor.assess(self.detailed_frame(), off_center)

        self.assertEqual("FACE_OFF_CENTER", result.reason)

    def test_rejects_a_blurred_low_detail_capture(self):
        gradient = np.tile(np.linspace(45, 205, 640, dtype=np.uint8), (480, 1))
        frame = np.dstack([gradient, gradient, gradient])

        result = self.assessor.assess(frame, self.face)

        self.assertEqual("BLURRY", result.reason)
        self.assertIn("imóvel", result.message)


if __name__ == "__main__":
    unittest.main()
