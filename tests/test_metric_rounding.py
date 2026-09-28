"""Cohort aggregation keeps full precision until the headline is rounded."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from eval_kinetics_all_cycles import cohort_subject_means


class MetricRounding(unittest.TestCase):
    def test_pcc_cohort_values_keep_precision_until_headline(self):
        scores = {'cohort_a': {'subject_a': [0.80049]}, 'cohort_b': {'subject_b': [0.80149]}}
        means = cohort_subject_means(scores)
        self.assertEqual(round(sum(means.values()) / 2, 3), 0.801)
        self.assertEqual(means['cohort_a'], 0.80049)


if __name__ == '__main__':
    unittest.main()
