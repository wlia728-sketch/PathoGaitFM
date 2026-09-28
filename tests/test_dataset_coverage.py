"""A declared training split must be present at file granularity."""
from contextlib import ExitStack, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'model'), str(ROOT / 'scripts/eval')]
from pathogait.data import dataset


class DatasetCoverage(unittest.TestCase):
    def create(self, folder, pairs, discovered, **options):
        root = Path(folder)
        manifest = root/'split.json'
        manifest.write_text(json.dumps({'splits':{'train':pairs}}),encoding='utf-8')
        with (patch.object(dataset,'_discover_files',return_value=discovered),
              patch.object(dataset,'V4SubjectMetadata',return_value=object())):
            return dataset.V4BalancedDatasetV3.from_split(
                manifest,'train',transform_override=object(),**options)

    def test_missing_file_and_entire_source_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            found={'cp':[str(Path(folder)/'cp_01.npy')]}
            for missing in [('cp','cp_02'),('vdk_stroke','vdk_stroke_001')]:
                with self.subTest(missing=missing),self.assertRaises(FileNotFoundError):
                    self.create(folder,[['cp','cp_01'],list(missing)],found)

    def test_stage1_parts_are_not_collapsed_to_subjects(self):
        with tempfile.TemporaryDirectory() as folder:
            name='addbio_Camargo2021'
            found={name:[str(Path(folder)/'Camargo2021_AB06_split0.npy')]}
            with self.assertRaisesRegex(FileNotFoundError,'split1'):
                self.create(folder,[[name,'Camargo2021_AB06_split0'],
                                    [name,'Camargo2021_AB06_split1']],found)

    def test_explicit_source_and_study_exclusions_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            found={'cp':[str(Path(folder)/'cp_01.npy')]}
            pairs=[['cp','cp_01'],['vdk_stroke','vdk_stroke_001']]
            selected=self.create(folder,pairs,found,sources=['cp'])
            self.assertEqual(selected.source_files,found)
            pairs=[['cp','cp_01'],['addbio_Han2023','Han2023_subject_split0']]
            selected=self.create(folder,pairs,found)
            self.assertEqual(selected.source_files,found)

    def test_normal_loading_keeps_discovery_order_and_manifest_cp_exclusions(self):
        with tempfile.TemporaryDirectory() as folder:
            found={'cp':[str(Path(folder)/'cp_02.npy'),str(Path(folder)/'cp_01.npy')]}
            selected=self.create(folder,[['cp','cp_01'],['cp','cp_02']],found)
            self.assertEqual(selected.source_files,found)


class MissingPairedCheckpoint(unittest.TestCase):
    def test_missing_weights_raise_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['eval_imu_zeroshot','eval_opencap_zeroshot']:
                with self.subTest(module=name),ExitStack() as context:
                    module=importlib.import_module(name)
                    context.enter_context(patch.object(module,'CKPT',Path(folder)/'missing.pt'))
                    load=context.enter_context(patch.object(module,'load_model'))
                    context.enter_context(redirect_stdout(io.StringIO()))
                    with self.assertRaises(FileNotFoundError):module.main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
