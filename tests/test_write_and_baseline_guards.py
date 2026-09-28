"""Protect frozen artifacts and fail before comparator training on incomplete data."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(ROOT / 'model_baseline')]
import guarded_write as writer
import baseline_nondiffusion as baseline


class FrozenOutputGuard(unittest.TestCase):
    def test_valid_result_redirects_and_preserves_existing_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frozen = root / 'results/nested/result.json'
            frozen.parent.mkdir(parents=True)
            frozen.write_text('{"PCC":0.8}', encoding='utf-8')
            before = frozen.read_bytes()
            with patch.object(writer, 'ROOT', root):
                saved = writer.guarded_dump({'PCC': 0.9}, frozen)
            self.assertEqual(saved, root / 'outputs/recomputed/nested/result.json')
            self.assertEqual(frozen.read_bytes(), before)
            self.assertEqual(json.loads(saved.read_text()), {'PCC': 0.9})
            self.assertEqual(writer.safe_artifact_path(root/'results/recomputed/x.json',root),
                             root/'results/recomputed/x.json')
            self.assertEqual(writer.safe_artifact_path(root/'outputs/new.json',root),
                             root/'outputs/new.json')

    def test_npz_and_override_do_not_bypass_archive_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frozen = root/'results/example.npz'
            frozen.parent.mkdir()
            frozen.write_bytes(b'archived content')
            with patch.object(writer,'ROOT',root):
                saved = writer.guarded_savez(frozen, values=np.array([1., 2.]))
                with patch.dict(os.environ, {'PATHOGAIT_ALLOW_DEGENERATE':'1'}):
                    empty = writer.guarded_dump({},root/'results/empty.json')
            self.assertEqual(frozen.read_bytes(),b'archived content')
            self.assertEqual(saved,root/'outputs/recomputed/example.npz')
            np.testing.assert_array_equal(np.load(saved)['values'],[1.,2.])
            self.assertEqual(empty,root/'outputs/recomputed/empty.json')
            self.assertFalse((root/'results/empty.json').exists())

    def test_degenerate_result_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(writer,'ROOT',root), patch.dict(os.environ,{},clear=True):
                with self.assertRaises(SystemExit):
                    writer.guarded_dump({'per_subject':{},'PCC':None},root/'results/a.json')
            self.assertFalse((root/'outputs').exists())


class BaselineInputGuard(unittest.TestCase):
    def fixture(self, folder):
        root=Path(folder)
        (root/'cp').mkdir()
        split={'cv5':[]}
        severity={'mappings':{'cp':{}}}
        for i in range(5):
            stem=f'cp_fixture_{i}'
            split['cv5'].append({'val':[['cp',stem]]})
            severity['mappings']['cp'][stem]={'cohort_id':0,'severity_id':0}
            np.save(root/'cp'/f'{stem}.npy',np.zeros((1,101,54),np.float32))
            np.save(root/'cp'/f'{stem}_mask.npy',np.ones((1,54),bool))
        return root,split,severity

    def test_missing_later_fold_prevents_any_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root,split,severity=self.fixture(folder)
            (root/'cp/cp_fixture_4_mask.npy').unlink()
            with (patch.object(baseline,'GRFFIX',root),patch.object(baseline,'SPLIT',split),
                  patch.object(baseline,'train_model') as train):
                with self.assertRaises(FileNotFoundError):
                    baseline.run(Mock(),True,Mock(),severity,in_mask=True)
                train.assert_not_called()

    def test_exclusion_preserved_but_empty_fold_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,split,severity=self.fixture(folder)
            split['cv5'][0]['val'].append(['cp','cp_excluded'])
            with (patch.object(baseline,'GRFFIX',root),patch.object(baseline,'SPLIT',split),
                  patch.object(baseline,'EXCLUDE',{'cp_excluded'})):
                baseline.preflight_inputs(severity)
                split['cv5'][0]['val']=[['cp','cp_excluded']]
                with self.assertRaisesRegex(ValueError,'no validation'):
                    baseline.preflight_inputs(severity)
                split['cv5'][0]['val']=[['cp','cp_fixture_0']]
                np.save(root/'cp/cp_fixture_4_mask.npy',np.ones((2,54),bool))
                with self.assertRaisesRegex(ValueError,'shape mismatch'):
                    baseline.preflight_inputs(severity)

    def test_transform_failure_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root,split,severity=self.fixture(folder)
            with ExitStack() as context:
                context.enter_context(patch.object(baseline,'GRFFIX',root))
                context.enter_context(patch.object(baseline,'SPLIT',split))
                with self.assertRaisesRegex(RuntimeError,'Cannot load comparator input'):
                    baseline.load_cycles(Mock(side_effect=ValueError('bad input')),severity,['cp_fixture_0'])


if __name__ == '__main__':
    unittest.main()
