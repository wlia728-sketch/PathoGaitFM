"""BMClab representation mismatches fail at training and evaluation boundaries."""
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "scripts/lib")]
from pathogait.data import dataset
from pathogait.data.pelvis_contract import ANGLES, LEGACY
import eval_common


class PelvisLoadingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.raw = self.root / "bmclab_pd" / "SUB01_on.npy"
        self.raw.parent.mkdir()
        self.raw.touch()
        env = patch.dict(os.environ, {"PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION": LEGACY})
        env.start()
        self.addCleanup(env.stop)

    def mark(self, representation):
        self.raw.with_name(self.raw.stem + "_meta.csv").write_text(
            "pelvis_representation\n" + representation + "\n", encoding="utf-8")

    def make_dataset(self, representation, **kwargs):
        with (patch.object(dataset, "_discover_files", return_value={"bmclab_pd": [str(self.raw)]}),
              patch.object(dataset, "V4SubjectMetadata", return_value=object())):
            return dataset.V4BalancedDatasetV3(
                bmclab_pelvis_representation=representation, **kwargs)

    def test_dataset_rejects_new_arrays_before_sampling(self):
        self.mark(ANGLES)
        with self.assertRaisesRegex(ValueError, "representation"):
            self.make_dataset(LEGACY)

    def test_explicit_mode_reaches_default_transform(self):
        self.mark(ANGLES)
        selected = self.make_dataset(ANGLES)
        self.assertEqual(selected.transform.bmclab_pelvis_representation, ANGLES)

    def test_dataset_rejects_conflicting_transform_override(self):
        transform = SimpleNamespace(bmclab_pelvis_representation=ANGLES)
        with self.assertRaisesRegex(ValueError, "Dataset and transform"):
            self.make_dataset(LEGACY, transform_override=transform)

    def test_cached_recording_cannot_bypass_changed_metadata(self):
        self.mark(LEGACY)
        with patch.object(dataset, "validate_bmclab_input", wraps=dataset.validate_bmclab_input) as check:
            selected = self.make_dataset(LEGACY)
            cached = (object(), object())
            selected._cache[str(self.raw)] = cached
            for _ in range(2):
                self.assertIs(selected._load_transformed(str(self.raw), "bmclab_pd"), cached)
            self.assertEqual(check.call_count, 1)
            self.mark(ANGLES)
            with self.assertRaisesRegex(ValueError, "representation"):
                selected._load_transformed(str(self.raw), "bmclab_pd")
            self.assertEqual(check.call_count, 2)

    def test_mask_loader_checks_actual_recording_path(self):
        self.mark(ANGLES)
        np.save(self.raw.with_name(self.raw.stem + "_mask.npy"), np.ones((1, 54), bool))
        with self.assertRaisesRegex(ValueError, "representation"):
            eval_common.load_valid_mask_40(self.raw, 1)
        with patch.dict(os.environ, {"PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION": ANGLES}):
            self.assertEqual(eval_common.load_valid_mask_40(self.raw, 1).shape, (1, 40))

    def test_transform_factory_does_not_check_unrelated_default_arrays(self):
        self.mark(LEGACY)
        with (patch.dict(os.environ, {"PATHOGAIT_COHORT_RAW_DIR": str(self.root),
                                     "PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION": ANGLES}),
              patch.object(eval_common, "require_local", return_value=Path("table.json")),
              patch.object(eval_common, "V4LazyTransformV3PerSubject") as build):
            eval_common.per_subject_transform(object(), require_tables=True)
            self.assertEqual(build.call_args.kwargs["bmclab_pelvis_representation"], ANGLES)

    def test_checkpoint_mismatch_rejects_before_model_build(self):
        with (patch.dict(os.environ, {"PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION": ANGLES}),
              patch.object(eval_common.torch, "load", return_value={"model": {}, "args": {}}),
              patch.object(eval_common, "DiT_B_1D") as build):
            with self.assertRaisesRegex(RuntimeError, "pelvis representation"):
                eval_common.load_model(Path("synthetic.pt"), "cpu")
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
