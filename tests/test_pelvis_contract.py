"""Synthetic BMClab representation checks; no data or checkpoints required."""
from argparse import Namespace
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from pathogait.data.pelvis_contract import (
    ANGLES, LEGACY, REPRESENTATIONS, configured_representation,
    validate_bmclab_input, validate_representation, verify_checkpoint_representation,
)
from pathogait.data.transform import V4LazyTransformV3
from pathogait.data.transform_per_subject import V4LazyTransformV3PerSubject


class PelvisContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw_path = self.directory / "bmclab_pd" / "synthetic.npy"
        self.raw_path.parent.mkdir()
        self.meta_path = self.raw_path.with_name("synthetic_meta.csv")

    def write_metadata(self, values):
        with self.meta_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["cycle", "pelvis_representation"])
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow({"cycle": index, "pelvis_representation": value})

    def test_environment_default_selection_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_representation(), LEGACY)
            for value in REPRESENTATIONS:
                os.environ["PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION"] = value
                self.assertEqual(configured_representation(), value)
            os.environ["PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION"] = ""
            with self.assertRaises(ValueError):
                configured_representation()
        for value in (None, "angles", "", [], 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_representation(value)

    def test_unlabelled_input_is_legacy(self):
        for metadata in (None, "cycle,side\n0,r\n"):
            if metadata is not None:
                self.meta_path.write_text(metadata, encoding="utf-8")
            validate_bmclab_input(self.raw_path, LEGACY)
            with self.assertRaises(ValueError):
                validate_bmclab_input(self.raw_path, ANGLES)

    def test_declared_inputs_require_matching_representation(self):
        for value in REPRESENTATIONS:
            self.write_metadata([value, value])
            validate_bmclab_input(self.raw_path, value)
            other = ANGLES if value == LEGACY else LEGACY
            with self.assertRaises(ValueError):
                validate_bmclab_input(self.raw_path, other)

    def test_empty_mixed_unknown_and_missing_row_values_are_rejected(self):
        for values in ([], [""], [ANGLES, ""], ["unknown"], [LEGACY, ANGLES], [None]):
            with self.subTest(values=values):
                self.write_metadata(values)
                for expected in REPRESENTATIONS:
                    with self.assertRaises(ValueError):
                        validate_bmclab_input(self.raw_path, expected)

    def test_other_cohorts_are_not_constrained(self):
        validate_bmclab_input(self.directory / "normal" / "synthetic.npy", ANGLES)

    def test_old_checkpoints_are_legacy(self):
        for checkpoint in ({}, {"args": {}}, {"args": Namespace()}):
            verify_checkpoint_representation(checkpoint, LEGACY, "synthetic")
            with self.assertRaises(RuntimeError):
                verify_checkpoint_representation(checkpoint, ANGLES, "synthetic")

    def test_new_checkpoints_require_matching_representation(self):
        for value in REPRESENTATIONS:
            for args in ({"bmclab_pelvis_representation": value},
                         Namespace(bmclab_pelvis_representation=value)):
                checkpoint = {"args": args}
                verify_checkpoint_representation(checkpoint, value, "synthetic")
                other = ANGLES if value == LEGACY else LEGACY
                with self.assertRaises(RuntimeError):
                    verify_checkpoint_representation(checkpoint, other, "synthetic")
        with self.assertRaises(ValueError):
            verify_checkpoint_representation(
                {"args": {"bmclab_pelvis_representation": "unknown"}}, LEGACY, "synthetic")

    def test_base_and_subject_transforms_preserve_distinct_pelvis_conventions(self):
        metadata = SimpleNamespace(get_bw=lambda *args: 70.0)
        raw = np.tile(np.arange(54, dtype=np.float32), (1, 101, 1))
        table_path = self.directory / "flips.json"
        table_path.write_text(json.dumps({"bmclab_pd": {"synthetic": {
            "hip_flex_r": -1, "ankle_dorsi_l": 1}}}), encoding="utf-8")
        cases = ((V4LazyTransformV3, {}),
                 (V4LazyTransformV3PerSubject, {}),
                 (V4LazyTransformV3PerSubject, {"per_subject_table_paper": str(table_path)}))
        for cls, kwargs in cases:
            with self.subTest(transform=cls.__name__, subject_table=bool(kwargs)):
                legacy = cls(metadata, bmclab_pelvis_representation=LEGACY, **kwargs)
                angular = cls(metadata, bmclab_pelvis_representation=ANGLES, **kwargs)
                old = legacy.preprocess_before_zscore(raw, "bmclab_pd", "cohort", "synthetic")
                new = angular.preprocess_before_zscore(raw, "bmclab_pd", "cohort", "synthetic")
                np.testing.assert_array_equal(new[..., 48:54], raw[..., 48:54])
                np.testing.assert_array_equal(old[..., [50, 53]], -raw[..., [50, 53]])
                unchanged = [ch for ch in range(54) if ch not in (50, 53)]
                np.testing.assert_array_equal(new[..., unchanged], old[..., unchanged])
                for source in ("normal", "cp", "vdk_stroke"):
                    np.testing.assert_array_equal(
                        legacy.preprocess_before_zscore(raw, source, "cohort", "synthetic"),
                        angular.preprocess_before_zscore(raw, source, "cohort", "synthetic"))

    def test_transform_configuration_and_override_lists_are_not_mutated(self):
        metadata = SimpleNamespace(get_bw=lambda *args: 70.0)
        override = {"bmclab_pd": [12, 50, 53], "normal": [50]}
        with patch.dict(os.environ, {"PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION": ANGLES}):
            selected = V4LazyTransformV3(metadata, cohort_flip_override=override)
            explicit = V4LazyTransformV3(metadata, cohort_flip_override=override,
                                         bmclab_pelvis_representation=LEGACY)
        self.assertEqual(selected.bmclab_pelvis_representation, ANGLES)
        self.assertEqual(selected.cohort_flip["bmclab_pd"], [12])
        self.assertEqual(selected.cohort_flip["normal"], [50])
        self.assertEqual(explicit.cohort_flip["bmclab_pd"], [12, 50, 53])
        self.assertEqual(override, {"bmclab_pd": [12, 50, 53], "normal": [50]})
        self.assertIsNot(selected.cohort_flip["normal"], override["normal"])


if __name__ == "__main__":
    unittest.main()
