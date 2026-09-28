"""CSV and local server contracts, without GPU or loading a model checkpoint.

The two real local examples are optional and are deliberately not distributed in
the code release. Set PATHOGAIT_DEMO_EXAMPLES to test an authorised local overlay.
"""
from pathlib import Path
from contextlib import ExitStack
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'demo'))
import input_csv
from server import Demo, TARGET_NAMES

LOCAL_EXAMPLES = Path(os.environ['PATHOGAIT_DEMO_EXAMPLES']) if os.environ.get('PATHOGAIT_DEMO_EXAMPLES') else None


def fixture_csv(channels=None, cycles=1, value=None):
    """Parser fixture only: constant numeric values are not a scientific example."""
    channels = channels or input_csv.ANGLE_COLUMNS
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(['cycle_id', 'gait_percent', *channels])
    for c in range(cycles):
        for phase in range(100):
            writer.writerow([f'cycle_{c}', phase, *[(value(c, phase, name) if value else 10 + input_csv.ANGLE_COLUMNS.index(name)) for name in channels]])
    return out.getvalue()


def replace_cell(text, row, column, value):
    rows = list(csv.reader(io.StringIO(text)))
    rows[row][column] = str(value)
    out = io.StringIO(newline='')
    csv.writer(out).writerows(rows)
    return out.getvalue()


class CSVContract(unittest.TestCase):
    def test_column_reordering_maps_to_canonical_order(self):
        text = fixture_csv(list(reversed(input_csv.ANGLE_COLUMNS)), cycles=2)
        angles, ids = input_csv.parse_angles_csv('\ufeff' + text)
        self.assertEqual(ids, ['cycle_0', 'cycle_1'])
        self.assertEqual(angles.shape, (2, 100, 16))
        np.testing.assert_array_equal(angles[1, 50], np.arange(10, 26, dtype=np.float32))

    def test_whole_missing_angle_and_omitted_columns_are_nan(self):
        text = fixture_csv(['R_Hip_X', 'R_Knee_X'], value=lambda c,p,n: '' if n=='R_Knee_X' else 20)
        angles, _ = input_csv.parse_angles_csv(text)
        self.assertTrue(np.isfinite(angles[:, :, 0]).all())
        self.assertTrue(np.isnan(angles[:, :, 1:]).all())

    def test_reject_unknown_and_kinetic_headers(self):
        base = fixture_csv()
        for header in ['Unknown_X', 'R_Hip_Mom_X', 'R_GRF_Z', 'vGRF', 'reference']:
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, 'Unexpected columns'):
                    input_csv.parse_angles_csv(base.replace('R_Hip_X', header, 1))

    def test_reject_duplicate_column(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            input_csv.parse_angles_csv(fixture_csv().replace('L_Hip_X','R_Hip_X',1))

    def test_reject_missing_phase_or_angles(self):
        for text in ['cycle_id,gait_percent\na,0\n', 'cycle_id,R_Hip_X\na,0\n', '', None, ' \n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                input_csv.parse_angles_csv(text)

    def test_reject_partial_missing_channels(self):
        for missing in ['', 'NaN', 'nan']:
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError,'entire cycle'):
                    input_csv.parse_angles_csv(replace_cell(fixture_csv(), 51, 2, missing))

    def test_reject_no_available_angles(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            input_csv.parse_angles_csv(fixture_csv(value=lambda c,p,n: ''))
        with self.assertRaisesRegex(ValueError, 'at least one'):
            input_csv.parse_angles_csv(input_csv.template_csv())

    def test_reject_wrong_phase_duplicate_and_101st_endpoint(self):
        for phase in [0, 49, 51, 100, 'NaN', 'inf', 'zero']:
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                input_csv.parse_angles_csv(replace_cell(fixture_csv(), 51, 1, phase))
        extra = fixture_csv() + 'cycle_0,100,' + ','.join(['10'] * 16) + '\n'
        with self.assertRaisesRegex(ValueError, 'more than 100'):
            input_csv.parse_angles_csv(extra)
        lines = fixture_csv().splitlines()
        with self.assertRaisesRegex(ValueError, '99 rows'):
            input_csv.parse_angles_csv('\n'.join(lines[:-1]))

    def test_reject_invalid_numeric_cells(self):
        for value in ['inf','-inf','1e999','361','-361','x','True','1+2j']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                input_csv.parse_angles_csv(replace_cell(fixture_csv(), 1, 2, value))

    def test_reject_wrong_row_width(self):
        base = fixture_csv()
        for text in [base.replace('cycle_0,0,','cycle_0,0,extra,',1), replace_cell(base,1,2,'').replace('cycle_0,0,,','cycle_0,0,',1)]:
            with self.assertRaisesRegex(ValueError, 'number of cells'):
                input_csv.parse_angles_csv(text)

    def test_reject_oversize_bytes_and_cycle_count(self):
        with self.assertRaisesRegex(ValueError, 'larger than 2 MB'):
            input_csv.parse_angles_csv('x' * (input_csv.MAX_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'at most'):
            input_csv.parse_angles_csv(fixture_csv(cycles=input_csv.MAX_CYCLES + 1))


class CapturingModel:
    def __init__(self):
        self.calls = []

    def predict(self, angles, **options):
        self.calls.append((angles.copy(), options.copy()))
        values = np.broadcast_to(np.arange(8, dtype=np.float32)[None,None,:], (len(angles),100,8)).copy()
        return None, values

    def conditions(self, cohort, source_id, severity_id):
        return source_id, {'cp':0,'normal':1,'vdk_stroke':2,'bmclab_pd':3}[cohort], severity_id

    def metadata(self):
        return {'checkpoint':'private/local/checkpoint.pt','checkpoint_sha256':'test-checkpoint-hash',
                'zstats_sha256':'test-stats-hash','ddim_steps':50,'cfg':1.0,'input_units':'degrees'}


class ServerContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.checkpoint = self.path / 'checkpoint.pt'
        self.checkpoint.write_bytes(b'not loaded; model is mocked')
        self.demo = Demo(self.checkpoint, 'cpu', seeds=3)
        self.demo.model = CapturingModel()

    def tearDown(self):
        self.temp.cleanup()

    def request(self, **extra):
        return {'csv':fixture_csv(), 'cohort':'cp', 'input_set':'full', 'conventions_confirmed':True, **extra}

    def test_only_angles_reach_model_no_upload_written_metadata_retained(self):
        request = self.request()
        before = sorted(p.name for p in self.path.iterdir())
        # The checkpoint is already instantiated. Inference must not write the
        # uploaded CSV or temporary data files through either common open API.
        with ExitStack() as stack:
            stack.enter_context(patch('builtins.open',side_effect=AssertionError('unexpected file I/O')))
            stack.enter_context(patch('io.open',side_effect=AssertionError('unexpected file I/O')))
            result = self.demo.predict(request)
        self.assertEqual(before, sorted(p.name for p in self.path.iterdir()))
        self.assertEqual(len(self.demo.model.calls),1)
        angles, options = self.demo.model.calls[0]
        self.assertEqual(angles.shape,(1,100,16))
        np.testing.assert_array_equal(angles[0,0],np.arange(10,26))
        self.assertEqual(options,{'cohort':'cp','input_set':'full','source_id':12,'severity_id':6,'seeds':3,'seed':42})
        self.assertEqual(result['metadata']['conditions'],[12,0,6])
        self.assertEqual(result['metadata']['seed_values'],[42,7961,15880])
        self.assertTrue(result['metadata']['no_reference_kinetics_used'])
        self.assertEqual(result['metadata']['input_csv_sha256'],hashlib.sha256(request['csv'].encode()).hexdigest())
        self.assertNotIn('checkpoint',result['metadata'])
        self.assertNotIn('csv',result)
        self.assertNotIn('pcc',json.dumps(result).lower())
        self.assertEqual(result['output_names'],TARGET_NAMES)
        self.assertEqual(result['output_units'],['N\u00b7m/kg']*6+['BW']*2)
        self.assertEqual(np.asarray(result['predictions']).shape,(1,100,8))
        for channel, name in enumerate(TARGET_NAMES):
            np.testing.assert_array_equal(result['mean_predictions'][name],np.full(100,channel))
        json.dumps(result,allow_nan=False)

    def test_reject_invalid_request_before_model(self):
        cases = [None, [], self.request(conventions_confirmed=False), self.request(conventions_confirmed=1),
                 self.request(cohort='auto'), self.request(input_set='generation'),
                 self.request(reference=[1,2,3]), self.request(csv=None), self.request(csv='bad header')]
        for request in cases:
            with self.subTest(request_type=type(request).__name__), self.assertRaises(ValueError):
                self.demo.predict(request)
        self.assertEqual(self.demo.model.calls,[])

    def test_configurations_forward_without_adding_kinetic_inputs(self):
        for value in input_csv.INPUT_SETS:
            result = self.demo.predict(self.request(input_set=value))
            self.assertEqual(result['metadata']['input_set'],value)
            self.assertEqual(self.demo.model.calls[-1][1]['input_set'],value)
            self.assertEqual(self.demo.model.calls[-1][0].shape[-1],16)

    def test_busy_and_missing_checkpoint_are_explicit(self):
        self.demo.lock.acquire()
        try:
            with self.assertRaises(BlockingIOError):self.demo.predict(self.request())
        finally:self.demo.lock.release()
        self.demo.checkpoint = self.path / 'missing.pt'
        with self.assertRaises(FileNotFoundError):self.demo.predict(self.request())
        self.assertEqual(self.demo.model.calls,[])


@unittest.skipUnless(LOCAL_EXAMPLES is not None and (LOCAL_EXAMPLES/'examples.json').is_file(),'Set PATHOGAIT_DEMO_EXAMPLES to an authorised local example folder; participant data are not distributed.')
class LocalExampleContract(unittest.TestCase):
    def test_two_real_csvs_and_payload_match_server_and_ui(self):
        examples = json.loads((LOCAL_EXAMPLES/'examples.json').read_text(encoding='utf8'))['examples']
        self.assertEqual([x['id'] for x in examples],['example_td','example_cp'])
        demo = Demo(ROOT/'checkpoints/not-loaded.pt','cpu',LOCAL_EXAMPLES)
        self.assertEqual(demo.examples(),examples)
        for example in examples:
            with self.subTest(example=example['id']):
                text = (LOCAL_EXAMPLES/example['input']['csv']).read_text(encoding='utf8')
                angles, ids = input_csv.parse_angles_csv(text)
                self.assertEqual(angles.shape,(example['nCycles'],100,16))
                self.assertEqual(len(ids),example['nCycles'])
                self.assertEqual(example['side'],'R')
                self.assertEqual(example['input']['channelNames'],input_csv.ANGLE_COLUMNS)
                np.testing.assert_allclose(angles.mean(0),example['input']['mean'],rtol=2e-6,atol=2e-5)
                self.assertTrue(example['provenance']['localOnly'])
                self.assertFalse(example['provenance']['redistributionCleared'])
                target=example['targets']['Hip_Mom']
                self.assertEqual(target['unit'],'N\u00b7m/kg')
                self.assertEqual(set(target['configurations']),set(input_csv.INPUT_SETS))
                reference=np.asarray(target['reference'])
                self.assertEqual(reference.shape,(100,))
                for prediction in target['configurations'].values():
                    mean=np.asarray(prediction['mean'])
                    self.assertEqual(mean.shape,(100,))
                    self.assertAlmostEqual(float(np.corrcoef(reference,mean)[0,1]),prediction['metrics']['pcc'],places=12)
                    self.assertAlmostEqual(float(np.sqrt(np.mean((reference-mean)**2))),prediction['metrics']['rmse'],places=12)
                self.assertNotIn('C:\\',json.dumps(example))
                self.assertNotIn('normal_06',json.dumps(example))
                self.assertNotIn('cp_11',json.dumps(example))


if __name__ == '__main__': unittest.main()
