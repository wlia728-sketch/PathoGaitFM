"""The fixed measured-reference window must not depend on signal zero crossings."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "opencap_reference_eval", ROOT / "scripts/eval/eval_opencap_moments.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class OpenCapReferenceContract(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        summary = []
        for subject in range(10):
            # 59 windows across 10 synthetic participants. True zero-valued
            # measured samples remain in the fixed 64-sample reference window.
            count = 5 if subject == 9 else 6
            raw = np.zeros((count, 101, 54), dtype=np.float32)
            mask = np.zeros((count, 54), dtype=bool)
            for _, left40 in evaluation.MOM.values():
                left54 = evaluation.INV40to54[left40]
                raw[:, :64, left54] = np.sin(np.linspace(0, 2 * np.pi, 64))
                raw[:, 10:20, left54] = 0
                mask[:, left54] = True
            stem = f"subject{subject}__momeval"
            np.save(self.root / f"{stem}.npy", raw)
            np.save(self.root / f"{stem}_mask.npy", mask)
            summary.append({"subj": f"subject{subject}", "mass": 70.0})
        (self.root / "_summary_momeval.json").write_text(json.dumps(summary))

    def tearDown(self):
        self.folder.cleanup()

    def test_true_zero_samples_do_not_shrink_reference_window(self):
        files, report = evaluation.validate_inputs(self.root)
        self.assertEqual(len(files), 10)
        self.assertEqual(report["reference_samples"], {"start": 0, "stop_exclusive": 64})
        self.assertEqual(report["counts_by_joint"], {"Hip": 59, "Knee": 59, "Ankle": 59})

    def test_different_padding_layout_is_rejected(self):
        file = self.root / "subject0__momeval.npy"
        raw = np.load(file)
        raw[0, 70, evaluation.INV40to54[13]] = 1.0
        np.save(file, raw)
        with self.assertRaisesRegex(ValueError, "outside the fixed reference block"):
            evaluation.validate_inputs(self.root)

    def test_shape_mismatch_is_not_silently_truncated(self):
        file = self.root / "subject0__momeval_mask.npy"
        np.save(file, np.load(file)[:-1])
        with self.assertRaisesRegex(ValueError, "array/mask layout"):
            evaluation.validate_inputs(self.root)

    def test_nonpositive_body_mass_is_rejected(self):
        file = self.root / "_summary_momeval.json"
        summary = json.loads(file.read_text())
        summary[0]["mass"] = 0
        file.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            evaluation.validate_inputs(self.root)

    def test_validation_does_not_load_model_and_frozen_output_is_protected(self):
        output = self.root / "validated.json"
        args = ["eval", "--processed", str(self.root), "--validate-only", "--out", str(output)]
        with patch.object(sys, "argv", args), patch.object(evaluation, "load_model") as load:
            evaluation.main()
            load.assert_not_called()
        self.assertTrue(output.is_file())
        args[-1] = str(ROOT / "results/opencap_moments.json")
        with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "protected"):
            evaluation.main()


if __name__ == "__main__":
    unittest.main()
