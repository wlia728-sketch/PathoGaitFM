"""Retired training switches cannot silently alter a manuscript run."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
from pathogait.data.run_provenance import capture_settings, validate_training_scope


class TrainingScopeTests(unittest.TestCase):
    def test_reported_configuration_is_accepted(self):
        settings = {
            'PATHOGAIT_INPUT_DROP_CH': '26,27,28,29,30,31,32,33,11,12,14,15',
            'PATHOGAIT_LOSS_TARGET_CH': '10,13,16,17,18,19,22,25',
            'PATHOGAIT_DYNCYCLE_POOL': '1',
        }
        with patch.dict(os.environ, settings, clear=True):
            validate_training_scope()
            result = capture_settings(SimpleNamespace(), ROOT)
        self.assertEqual(result['environment']['PATHOGAIT_LOSS_TARGET_CH'],
                         settings['PATHOGAIT_LOSS_TARGET_CH'])

    def test_retired_switches_are_rejected_before_provenance(self):
        for key in ('PATHOGAIT_EMG_DROPOUT_CH', 'PATHOGAIT_AUX_LOSS_CH'):
            for value in ('26,27', ' '):
                with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}, clear=True):
                    with self.assertRaisesRegex(ValueError, key):
                        capture_settings(SimpleNamespace(), ROOT)

    def test_direct_forward_call_cannot_bypass_scope_check(self):
        spec = importlib.util.spec_from_file_location(
            'scope_test_stage2', ROOT / 'scripts/train/train_finetune.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for key in ('PATHOGAIT_EMG_DROPOUT_CH', 'PATHOGAIT_AUX_LOSS_CH'):
            with self.subTest(key=key), patch.dict(os.environ, {key: '26'}, clear=True):
                with self.assertRaisesRegex(ValueError, key):
                    module._step_forward(None, None, {}, 'cpu', {})


if __name__ == '__main__':
    unittest.main()
