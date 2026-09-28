"""Check that the public producer runs only the manuscript's matched comparison."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_baseline"))
import published_comparators as producer


class ComparatorScope(unittest.TestCase):
    def test_entry_point_runs_three_matched_architectures(self):
        names = [name for name, _ in producer.ARCHS]
        self.assertEqual(names, ["groundlink_cnn", "sugai_lstm", "ozates_cnn"])
        calls, artifacts = [], []
        transform, severity = object(), {"fixture": True}

        def fake_run(build, seq, received_transform, received_severity, in_mask=False):
            calls.append(build())
            self.assertTrue(seq)
            self.assertTrue(in_mask, "Every reported comparator receives input-validity flags")
            self.assertIs(received_transform, transform)
            self.assertEqual(received_severity, severity)
            return {"subject_macro_PCC": 0.5, "n_params": 1, "elapsed_s": 0}

        def capture(artifact, destination, writer):
            self.assertEqual(destination, output)
            self.assertEqual(writer, "published_comparators.py")
            artifacts.append(json.loads(json.dumps(artifact)))

        builders = [(name, lambda cin, n=name: (n, cin)) for name in names]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "comparison.json"
            sev = Path(folder) / "severity.json"
            sev.write_text(json.dumps(severity), encoding="utf-8")
            with (patch.object(sys, "argv", ["published_comparators.py", "--out", str(output)]),
                  patch.object(producer, "ARCHS", builders),
                  patch.object(producer, "build_transform", return_value=transform),
                  patch.object(producer, "SEV", sev),
                  patch.object(producer, "run", side_effect=fake_run),
                  patch.object(producer, "guarded_dump", side_effect=capture),
                  patch.object(producer, "diffusion_reference", return_value=0.5),
                  contextlib.redirect_stdout(io.StringIO())):
                producer.main()

        self.assertEqual(calls, [(name, 32) for name in names])
        final = artifacts[-1]
        self.assertEqual(set(final), {f"{name}__matched" for name in names} | {"reference", "arms"})
        self.assertEqual(set(final["arms"]), {"matched"})
        for name in names:
            config = final[f"{name}__matched"]["config"]
            self.assertTrue(config["input_validity_channels"])
            self.assertEqual(config["n_input_channels"], 32)
            self.assertEqual(config["epochs"], 80)
            self.assertEqual(config["n_seed_ensemble"], 3)
            self.assertTrue(config["pd_moment_loss_excluded"])

    def test_results_folder_cannot_be_selected_as_output(self):
        destination = ROOT / "results/comparison.json"
        with (patch.object(sys, "argv", ["published_comparators.py", "--out", str(destination)]),
              patch.object(producer, "build_transform") as build,
              contextlib.redirect_stderr(io.StringIO()),
              self.assertRaises(SystemExit) as error):
            producer.main()
        self.assertEqual(error.exception.code, 2)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
