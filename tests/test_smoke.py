import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(file_name: str):
    spec = importlib.util.spec_from_file_location(file_name, ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task1 = load_module("task1代码.py")
        cls.task2 = load_module("task2代码.py")

    def test_task1_psf_is_normalized(self):
        psf = self.task1.psf_gaussian(21, 2.0, 3.0)
        self.assertEqual(psf.shape, (21, 21))
        self.assertAlmostEqual(float(psf.sum()), 1.0, places=5)

    def test_task1_uint8_conversion_clips_range(self):
        values = np.array([-1.0, 0.0, 0.5, 2.0], dtype=np.float32)
        result = self.task1.to_u8(values)
        self.assertEqual(result.tolist(), [0, 0, 128, 255])

    def test_task2_ensure_odd(self):
        self.assertEqual(self.task2.ensure_odd(4), 5)
        self.assertEqual(self.task2.ensure_odd(5), 5)

    def test_task2_nonoverlap_rule(self):
        props = {
            1: (0.0, 0.0, 2.0, 12),
            2: (10.0, 0.0, 2.0, 12),
            3: (11.0, 0.0, 2.0, 12),
        }
        nonoverlap, overlap = self.task2.nonoverlap_by_circle(props)
        self.assertIn(1, nonoverlap)
        self.assertIn(2, overlap)
        self.assertIn(3, overlap)


if __name__ == "__main__":
    unittest.main()
