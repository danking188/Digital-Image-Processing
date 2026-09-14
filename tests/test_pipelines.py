import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from image_io import write_image
from test_smoke import load_module


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task1 = load_module("task1代码.py")
        cls.task2 = load_module("task2代码.py")

    def test_wiener_identity_psf_does_not_shift_image(self):
        source = np.zeros((31, 31), dtype=np.float32)
        source[10, 12] = 1
        psf = np.zeros((3, 3), dtype=np.float32)
        psf[1, 1] = 1
        restored = self.task1.wiener_deconv(source, psf, K=0)
        np.testing.assert_allclose(restored, source, atol=1e-6)

    def test_invalid_psf(self):
        for args in [(0, 1, 1), (2, 1, 1), (3, 0, 1), (3, float("nan"), 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.task1.psf_gaussian(*args)

    def test_tiny_image_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            self.task1.find_crosshair_by_template(np.zeros((10, 10), np.float32))

    def test_roi_validation(self):
        source = np.zeros((80, 100, 3), np.uint8)
        for roi in [(-1, 0, 20, 20), (0, 0, 0, 20), (99, 0, 20, 20), (0, 0, 10), (0.5, 0, 10, 10)]:
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                self.task2.count_particles(source, roi=roi)

    def test_empty_image_and_background(self):
        with self.assertRaises(ValueError):
            self.task2.count_particles(np.zeros((0, 0, 3), np.uint8))
        count, nonoverlap, *_ = self.task2.count_particles(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual((count, nonoverlap), (0, 0))

    def test_particle_pipeline_writes_readable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.zeros((220, 320, 3), np.uint8)
            for center in [(70, 90), (230, 90)]:
                cv2.circle(source, center, 25, (255, 255, 255), -1)
            path = Path(directory) / "input.png"
            write_image(path, source)
            result = self.task2.run_particle_count(str(path), str(Path(directory) / "results"))
            self.assertEqual((result["num_all"], result["num_nonoverlap"]), (2, 2))
            for name in ["all", "nonoverlap", "binary", "watershed"]:
                self.assertIsNotNone(cv2.imread(result[name]))

    def test_vessel_pipeline_writes_readable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.zeros((200, 200), np.uint8)
            cv2.line(source, (150, 135), (150, 165), 255, 3)
            cv2.line(source, (135, 150), (165, 150), 255, 3)
            cv2.line(source, (20, 30), (100, 110), 180, 2)
            source = cv2.GaussianBlur(source, (9, 9), 1.5)
            path = Path(directory) / "input.png"
            write_image(path, source)
            result = self.task1.run_vessel_enhancement(str(path), str(Path(directory) / "results"))
            for name in ["final", "restored", "psf"]:
                self.assertIsNotNone(cv2.imread(result[name]))
            self.assertTrue(np.isfinite(result["template_score"]))

    def test_failed_output_write_is_not_success(self):
        with patch("image_io.cv2.imwrite", return_value=False), self.assertRaises(OSError):
            write_image(Path("unused.png"), np.zeros((4, 4), np.uint8))


if __name__ == "__main__":
    unittest.main()
