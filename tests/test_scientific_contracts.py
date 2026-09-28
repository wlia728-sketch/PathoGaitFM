import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/lib'),str(ROOT/'model')]
from evaluation_contract import preflight_cv5, validate_coverage, safe_result_path
from patient_statistics import resample_units, paired_test, cluster_means
from figure_io import output_dir
from pathogait.data.run_provenance import capture_settings, save_training_manifest
from display_filter import display_curve


class ScientificContracts(unittest.TestCase):
    def test_display_filter_preserves_raw_and_constant(self):
        raw=np.array([0.,0.,1.,0.,0.]);before=raw.copy()
        x,y,smooth=display_curve(raw,sigma=1,radius=3)
        np.testing.assert_array_equal(raw,before)
        self.assertEqual(len(smooth),len(raw));self.assertEqual(len(x),1001)
        self.assertTrue(np.isfinite(y).all())
        _,y,_=display_curve(np.full(100,2.),sigma=2,radius=6)
        np.testing.assert_allclose(y,2.)

    def test_pd_resampling_keeps_states_together(self):
        stems=['SUB01_on','SUB01_off','SUB02_on']
        rng=np.random.default_rng(12)
        for _ in range(100):
            sample=resample_units('bmclab_pd',stems,rng)
            self.assertEqual(sample.count('SUB01_on'),sample.count('SUB01_off'))
        result=paired_test({'bmclab_pd':{'SUB01_on':1.,'SUB01_off':-1.,'SUB02_on':0.}})
        self.assertEqual(result['n_patients_paired'],2)
        self.assertEqual(result['wilcoxon_p_two_sided'],1.)

    def test_single_patient_cluster_preserves_record_mean(self):
        values={'SUB01_on':1.,'SUB01_off':0.}
        samples=cluster_means('bmclab_pd',values,np.random.default_rng(5),100)
        np.testing.assert_array_equal(samples,np.full(100,.5))

    def test_frozen_output_and_nested_paths_rejected(self):
        for path in [ROOT/'figures',ROOT/'figures/nested',ROOT/'outputs/../figures']:
            with self.assertRaises(ValueError):output_dir(path)
        with self.assertRaises(ValueError):safe_result_path(ROOT/'results/kinetics.json',ROOT)
        self.assertEqual(safe_result_path(ROOT/'results/recomputed/a.json',ROOT),ROOT/'results/recomputed/a.json')

    def test_cv5_requires_every_weight_and_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'cp').mkdir()
            split={'cv5':[]};severity={'mappings':{'cp':{}}};checkpoints=[]
            for i in range(5):
                stem=f'cp_{i}';split['cv5'].append({'train':[],'val':[['cp',stem]]})
                severity['mappings']['cp'][stem]={'cohort_id':0,'severity_id':0}
                np.save(root/'cp'/f'{stem}.npy',np.ones((1,101,54)))
                np.save(root/'cp'/f'{stem}_mask.npy',np.ones((1,54),bool))
                cp=root/f'fold{i}.pt';cp.write_bytes(b'synthetic weight');checkpoints.append(cp)
            expected,report=preflight_cv5(split,root,severity,checkpoints)
            self.assertEqual(report['eligible_units'],5)
            with self.assertRaises(RuntimeError):validate_coverage(set.union(*expected.values()),{('cp','cp_0')})
            checkpoints[-1].unlink()
            with self.assertRaises(FileNotFoundError):preflight_cv5(split,root,severity,checkpoints)
            checkpoints[-1].write_bytes(b'synthetic weight')
            (root/'cp/cp_0_mask.npy').unlink()
            with self.assertRaises(FileNotFoundError):preflight_cv5(split,root,severity,checkpoints)

    def test_effective_settings_and_data_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stats=root/'stats.json';stats.write_text('{}')
            args=SimpleNamespace(zstats=str(stats))
            with patch.dict(os.environ,{'PATHOGAIT_INPUT_DROP_CH':'26,27','PATHOGAIT_LOSS_TARGET_CH':'10,22'}):
                capture_settings(args,root)
            p=root/'sample.npy';np.save(p,np.ones((1,101,54)))
            np.save(root/'sample_mask.npy',np.ones((1,54),bool))
            ds=SimpleNamespace(source_files={'cp':[str(p)]})
            save_training_manifest(args,ds,ds,root)
            saved=json.loads((root/'args.json').read_text())
            self.assertEqual(saved['run_provenance']['environment']['PATHOGAIT_INPUT_DROP_CH'],'26,27')
            self.assertEqual(len(saved['run_provenance']['data_manifest_sha256']),64)
            manifest=json.loads((root/'training_manifest.json').read_text())
            self.assertEqual(len(manifest['data_files']['train'][0]['array_sha256']),64)
            with patch.dict(os.environ,{'PATHOGAIT_INPUT_DROP_CH':'40'}):
                with self.assertRaises(ValueError):capture_settings(args,root)

if __name__=='__main__':unittest.main()
