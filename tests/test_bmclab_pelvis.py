"""Synthetic checks for BMClab's angular pelvis channels and centering."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PREP = Path(__file__).resolve().parents[1] / "data/prepare/bmclab"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, PREP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapping = load_module("s5_channel_mapping")
centering = load_module("s6_pelvis_demean")


class BMClabPelvis(unittest.TestCase):
    def setUp(self):
        phase = np.linspace(0, 2 * np.pi, 101, dtype=np.float32)
        self.cycle = np.tile(np.arange(34, dtype=np.float32)[:, None], (1, 101))
        self.cycle[:3] = np.stack((12 + 4 * np.sin(phase),
                                   -3 + 2 * np.cos(phase),
                                   5 + 6 * np.sin(phase)))
        self.powers = np.tile(np.arange(6, dtype=np.float32)[:, None], (1, 101))

    def build(self, side="r", grf_presence=1.0, grf_mask_category="pass"):
        return mapping.build_54ch(
            self.cycle, cycle_side=side, grf_presence=grf_presence,
            grf_mask_category=grf_mask_category, computed_powers=self.powers)

    def test_step4_angles_populate_both_slots_for_either_cycle_side(self):
        for side in ("r", "l"):
            with self.subTest(side=side):
                data, mask = self.build(side, grf_presence=0.0,
                                        grf_mask_category="both_missing")
                for source, slots in enumerate(((48, 51), (49, 52), (50, 53))):
                    for slot in slots:
                        np.testing.assert_array_equal(data[:, slot], self.cycle[source])
                        self.assertTrue(mask[slot])
                self.assertFalse(mask[18:40].any())

    def test_nonfinite_angle_invalidates_whole_channel_and_both_slots(self):
        for value in (np.nan, np.inf, -np.inf):
            for source, slots in enumerate(((48, 51), (49, 52), (50, 53))):
                with self.subTest(value=value, source=source):
                    saved = self.cycle[source, 17]
                    self.cycle[source, 17] = value
                    data, mask = self.build()
                    self.assertTrue(np.isnan(data[:, slots]).all())
                    self.assertFalse(mask[list(slots)].any())
                    other = [ch for ch in range(48, 54) if ch not in slots]
                    self.assertTrue(mask[other].all())
                    self.cycle[source, 17] = saved

    def test_pelvis_changes_leave_all_other_channels_unchanged(self):
        original, original_mask = self.build()
        self.cycle[:3] += 25
        changed, changed_mask = self.build()
        np.testing.assert_array_equal(changed[:, :48], original[:, :48])
        np.testing.assert_array_equal(changed_mask, original_mask)
        np.testing.assert_allclose(changed[:, 48:54], original[:, 48:54] + 25)

    def test_centering_preserves_excursion_other_channels_and_masks(self):
        self.cycle[1, 17] = np.nan
        data, mask = self.build()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            data_path = out / "synthetic.npy"
            mask_path = out / "synthetic_mask.npy"
            np.save(data_path, data[None])
            np.save(mask_path, mask[None])
            original_mask_bytes = mask_path.read_bytes()
            with patch.object(centering, "OUT", out), redirect_stdout(io.StringIO()):
                centering.main()
            centered = np.load(data_path)[0]
            np.testing.assert_array_equal(centered[:, :48], data[:, :48])
            self.assertEqual(mask_path.read_bytes(), original_mask_bytes)
            self.assertTrue(np.isnan(centered[:, [49, 52]]).all())
            observed = [48, 50, 51, 53]
            np.testing.assert_allclose(centered[:, observed].mean(axis=0), 0, atol=1e-6)
            np.testing.assert_allclose(np.ptp(centered[:, observed], axis=0),
                                       np.ptp(data[:, observed], axis=0), atol=1e-6)

    def test_centering_empty_directory_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(centering, "OUT", Path(directory)), redirect_stdout(io.StringIO()):
                centering.main()


if __name__ == "__main__":
    unittest.main()
