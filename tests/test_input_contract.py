"""Upload contract checks that run without checkpoint files."""
from pathlib import Path
from argparse import Namespace
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import pathogait_api
from pathogait_api import PathoGait, prepare_angles, physical, INPUT_CH, TARGET_CH, seed_values, MASKS
from eval_common import build_known_mask
from drop_channels import DROP_16CH
from pathogait.data.pelvis_contract import ANGLES, LEGACY


class InputContract(unittest.TestCase):
    def test_checkpoint_constructor_records_pelvis_representation(self):
        states = [({}, LEGACY), ({"args": {}}, LEGACY),
                  ({"args": {"bmclab_pelvis_representation": ANGLES}}, ANGLES),
                  ({"args": Namespace(bmclab_pelvis_representation=ANGLES)}, ANGLES)]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'synthetic.pt'
            checkpoint.write_bytes(b'synthetic')
            for state, expected in states:
                with self.subTest(state=state):
                    with patch.object(pathogait_api, 'load_model',
                                      return_value=(None, state, 'test', object())), \
                            patch.object(pathogait_api, 'DDPMScheduler'):
                        api = PathoGait(checkpoint)
                    self.assertEqual(api.bmclab_pelvis_representation, expected)

    def test_legacy_bmclab_rejects_observed_pelvis_before_sampling(self):
        api = PathoGait.__new__(PathoGait)
        api.model = SimpleNamespace(num_sources=12, num_cohorts=4, num_severities=6)
        api.device, api.scheduler, api.token = torch.device('cpu'), None, None
        # A single observed pelvis channel in one cycle must reject the batch.
        x = np.zeros((2, 100, 16), np.float32)
        x[:, :, 10:] = np.nan
        x[1, :, 15] = 2.0
        with patch.object(pathogait_api, 'tweedie_inpaint_drop') as sampler:
            with self.assertRaisesRegex(ValueError, 'expects pelvis translations'):
                api.predict(x, 'bmclab_pd')
            sampler.assert_not_called()

    def test_legacy_bmclab_accepts_each_way_to_omit_pelvis(self):
        api = PathoGait.__new__(PathoGait)
        api.model = SimpleNamespace(num_sources=12, num_cohorts=4, num_severities=6)
        api.device, api.scheduler, api.token = torch.device('cpu'), None, None
        api.bmclab_pelvis_representation = LEGACY
        full = np.zeros((100, 16), np.float32)
        missing = full.copy()
        missing[:, 10:] = np.nan
        observed = np.ones(16, dtype=bool)
        observed[10:] = False
        for name, x, kwargs in (('configuration', full, {'input_set': 'no_pelvis'}),
                                ('NaN', missing, {}),
                                ('mask', full, {'mask': observed})):
            with self.subTest(method=name):
                with patch.object(pathogait_api, 'tweedie_inpaint_drop',
                                  side_effect=lambda *args: args[2].clone()) as sampler:
                    _, result = api.predict(x, 'bmclab_pd', seeds=1, **kwargs)
                sampler.assert_called_once()
                self.assertFalse(sampler.call_args.args[3][:, 34:40].any())
                self.assertEqual(result.shape, (1, 100, 8))

    def test_angular_bmclab_and_other_cohorts_accept_pelvis_angles(self):
        api = PathoGait.__new__(PathoGait)
        api.model = SimpleNamespace(num_sources=12, num_cohorts=4, num_severities=6)
        api.device, api.scheduler, api.token = torch.device('cpu'), None, None
        x = np.zeros((100, 16), np.float32)
        for cohort, representation in (('bmclab_pd', ANGLES), ('normal', LEGACY),
                                       ('cp', LEGACY), ('vdk_stroke', LEGACY)):
            with self.subTest(cohort=cohort):
                api.bmclab_pelvis_representation = representation
                with patch.object(pathogait_api, 'tweedie_inpaint_drop',
                                  side_effect=lambda *args: args[2].clone()) as sampler:
                    api.predict(x, cohort, seeds=1)
                sampler.assert_called_once()
                self.assertTrue(sampler.call_args.args[3][:, 34:40].all())

    def test_seven_configurations_match_manuscript_channel_masks(self):
        sys.path.insert(0, str(ROOT / 'scripts/eval'))
        from partial_input_masking import CONDITIONS
        sys.path.insert(0, str(ROOT / 'demo'))
        from input_csv import INPUT_SETS
        self.assertEqual(len(MASKS), 7)
        self.assertEqual(set(INPUT_SETS), set(CONDITIONS))
        self.assertEqual({key: set(value) for key, value in MASKS.items()},
                         {key: set(value) for key, value in CONDITIONS.items()})
        self.assertIs(MASKS, pathogait_api.INPUT_MASKS)

    def test_each_configuration_masks_exact_angles_and_keeps_kinetics_hidden(self):
        api = PathoGait.__new__(PathoGait)
        api.model = SimpleNamespace(num_sources=12, num_cohorts=4, num_severities=6)
        api.device, api.scheduler, api.token = torch.device('cpu'), None, None
        x = np.zeros((100, 16), np.float32)
        for name, removed in MASKS.items():
            with self.subTest(configuration=name):
                with patch.object(pathogait_api, 'tweedie_inpaint_drop',
                                  side_effect=lambda *args: args[2].clone()) as sampler:
                    _, result = api.predict(x, 'normal', input_set=name, seeds=3)
                self.assertEqual(sampler.call_count, 3)
                self.assertEqual(result.shape, (1, 100, 8))
                self.assertTrue(np.isfinite(result).all())
                expected = np.ones(40, bool)
                expected[list(removed)] = False
                for call in sampler.call_args_list:
                    args = call.args
                    np.testing.assert_array_equal(args[3].numpy()[0], expected)
                    self.assertEqual(list(args[-1]), list(DROP_16CH))
                    known = build_known_mask(args[2], args[3]).numpy()
                    known[:, DROP_16CH] = 0
                    self.assertFalse(known[:, TARGET_CH].any())
                    self.assertEqual(set(np.flatnonzero(known[0, :, 0])),
                                     set(INPUT_CH) - set(removed))
                self.assertEqual(api.last_prediction['seed_values'], [42, 7961, 15880])
                self.assertEqual(api.last_prediction['observed_angle_mask'],
                                 expected[INPUT_CH][None].tolist())

    def test_configuration_cannot_remove_last_available_angle(self):
        api = PathoGait.__new__(PathoGait)
        api.model = SimpleNamespace(num_sources=12, num_cohorts=4, num_severities=6)
        api.device, api.scheduler, api.token = torch.device('cpu'), None, None
        for name, removed in MASKS.items():
            x = np.full((2, 100, 16), np.nan, np.float32)
            x[0] = 0
            if removed:
                x[1, :, INPUT_CH.index(removed[0])] = 0
            with self.subTest(configuration=name):
                with patch.object(pathogait_api, 'tweedie_inpaint_drop') as sampler:
                    with self.assertRaises(ValueError):
                        api.predict(x, 'normal', input_set=name)
                    sampler.assert_not_called()

    def test_missing_channels_and_hidden_kinetics(self):
        x = np.zeros((100, 16), np.float32)
        x[:, 3:6] = np.nan
        mask = np.ones(16, bool)
        mask[8] = False
        z, valid = prepare_angles(x, mask)
        known = build_known_mask(torch.from_numpy(z), torch.from_numpy(valid)).numpy()
        known[:, DROP_16CH] = 0
        self.assertEqual(z.shape, (1, 40, 100))
        self.assertFalse(valid[0, 3:6].any())
        self.assertFalse(valid[0, 8])
        self.assertFalse(known[:, TARGET_CH].any())
        self.assertEqual(int(known[:, INPUT_CH].sum()), 1200)
        np.testing.assert_allclose(physical(z)[0, 0], x[:, 0], atol=1e-5)

    def test_invalid_uploads_fail_before_model_use(self):
        invalid = [np.zeros((99, 16)), np.zeros((100, 17)),
                   np.full((100, 16), np.nan), np.full((100, 16), np.inf),
                   np.ones((100, 16), dtype=complex), np.full((100, 16), '1'),
                   np.ones((100, 16), dtype=bool)]
        partial = np.zeros((100, 16)); partial[50, 2] = np.nan
        invalid.append(partial)
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError): prepare_angles(value)
        for mask in [np.ones(15), np.full(16, 0.5), np.zeros(16), np.full(16, np.nan)]:
            with self.assertRaises(ValueError): prepare_angles(np.zeros((100, 16)), mask)

    def test_condition_defaults_do_not_invent_source_or_severity(self):
        api = PathoGait.__new__(PathoGait)
        api.model = SimpleNamespace(num_sources=12, num_cohorts=4, num_severities=6)
        self.assertEqual(api.conditions('normal'), (12, 1, 6))
        self.assertEqual(api.conditions('cp', 9, 0), (9, 0, 0))
        for cohort in ['unknown', None, 1, [], True]:
            with self.assertRaises(ValueError): api.conditions(cohort)
        for value in [True, 1.5, '2', -1, 13, np.nan]:
            with self.assertRaises(ValueError): api.conditions('cp', source_id=value)
        for value in [False, 0.5, -1, 7, '0']:
            with self.assertRaises(ValueError): api.conditions('cp', severity_id=value)

    def test_seeds_are_integer_reproducible_and_bounded(self):
        self.assertEqual(seed_values(42, 3), [42, 7961, 15880])
        for seed, count in [(True, 3), (1.5, 3), (-1, 3), (42, 0),
                            (42, 1.5), (42, True), (2**63 - 1, 3)]:
            with self.assertRaises(ValueError): seed_values(seed, count)


if __name__ == '__main__': unittest.main()
