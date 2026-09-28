"""Regression checks for input units, masks and the backbone forward pass."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import pathogait_api as api
from pathogait.models.dit1d import DiT_B_1D


def synthetic_angle_fixture():
    """Deterministic schema fixture, not participant data or a benchmark."""
    mean, scale = api.channel_stats()
    phase = np.arange(100, dtype=np.float32) / 100
    return mean[api.INPUT_CH][None, None, :] + (
        .1 * scale[api.INPUT_CH][None, None, :]
        * np.sin(2 * np.pi * phase)[None, :, None])


class InputContract(unittest.TestCase):
    def test_units_and_target_hiding(self):
        mean, scale = api.channel_stats()
        angles = np.broadcast_to(mean[api.INPUT_CH] + .2 * scale[api.INPUT_CH], (2, 100, 16)).copy()
        z, valid = api.prepare_angles(angles)
        np.testing.assert_allclose(z[:, api.INPUT_CH], .2, atol=1e-6)
        np.testing.assert_allclose(api.physical(z)[:, api.INPUT_CH].transpose(0, 2, 1), angles, atol=1e-5)
        self.assertTrue(valid[:, api.TARGET_CH].all())
        self.assertTrue((z[:, api.TARGET_CH] == 0).all())
        # Affine conversion changes RMSE by a channel's physical scale; range nRMSE is unchanged.
        ref = np.linspace(-.8, .8, 100)
        pred = ref + .1
        for ch in (10, 22):
            a, b = ref * scale[ch] + mean[ch], pred * scale[ch] + mean[ch]
            self.assertAlmostEqual(np.sqrt(np.mean((a-b)**2)) / np.ptp(a), .1 / 1.6, places=6)
        self.assertEqual(api.TARGET_UNITS, ["Nm/kg"] * 6 + ["BW"] * 2)

    def test_missing_channel_and_rejections(self):
        angles = synthetic_angle_fixture()
        angles[:, :, 0] = np.nan
        z, valid = api.prepare_angles(angles)
        self.assertFalse(valid[0, 0])
        self.assertTrue(np.isfinite(z).all())
        invalid = angles.copy()
        invalid[0, 5, 1] = np.nan
        for data in (invalid, np.full((100, 16), np.nan), np.zeros((101, 16)), np.full((100, 16), np.inf)):
            with self.assertRaises(ValueError):
                api.prepare_angles(data)
        with self.assertRaises(ValueError):
            api.prepare_angles(synthetic_angle_fixture(), np.zeros(16))
        with self.assertRaises(ValueError):
            api.prepare_angles(synthetic_angle_fixture(), np.full(16, 2))

    def test_backbone_forward(self):
        model = DiT_B_1D(hidden_size=24, depth=1, num_heads=4).eval()
        ids = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            result = model(torch.zeros(2, 40, 100), ids, ids, ids, ids)
        self.assertEqual(tuple(result.shape), (2, 40, 100))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
