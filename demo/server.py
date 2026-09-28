"""Local upload-to-inference demo. Run from the submission root: python demo/server.py."""
from __future__ import annotations
import argparse
import hashlib
import json
import mimetypes
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from input_csv import ANGLE_COLUMNS, COHORTS, INPUT_SETS, MAX_BYTES, parse_angles_csv, template_csv

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / 'static'
TARGET_NAMES = ['R_Hip_Mom_X', 'L_Hip_Mom_X', 'R_Knee_Mom_X', 'L_Knee_Mom_X',
                'R_Ankle_Mom_X', 'L_Ankle_Mom_X', 'R_GRF_Z', 'L_GRF_Z']


class Demo:
    def __init__(self, checkpoint, device, examples=None, seeds=3):
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.seeds = seeds
        self.examples_dir = examples.resolve() if examples else None
        self.token = secrets.token_urlsafe(32)
        self.model = None
        self.lock = threading.Lock()

    def examples(self):
        path = self.examples_dir / 'examples.json' if self.examples_dir else None
        if path and path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            return data['examples'] if isinstance(data, dict) else data
        return []

    def predict(self, request):
        if not isinstance(request, dict):
            raise ValueError('Expected a JSON object.')
        if set(request) - {'csv', 'cohort', 'input_set', 'conventions_confirmed'}:
            raise ValueError('Unknown prediction options.')
        if request.get('conventions_confirmed') is not True:
            raise ValueError('Confirm that the uploaded angles use the documented units, gait-cycle alignment and sign conventions.')
        cohort = request.get('cohort')
        input_set = request.get('input_set')
        if cohort not in COHORTS or input_set not in INPUT_SETS:
            raise ValueError('Choose a model cohort and an available input configuration.')
        angles, cycle_ids = parse_angles_csv(request.get('csv'))
        if not self.checkpoint.is_file():
            raise FileNotFoundError('The trained checkpoint is not installed. See the setup instructions.')
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('A prediction is already running. Wait for it to finish.')
        try:
            if self.model is None:
                sys.path.insert(0, str(ROOT / 'scripts/lib'))
                from pathogait_api import PathoGait
                self.model = PathoGait(self.checkpoint, self.device)
            # Unknown source and severity, never inferred from the uploaded file name.
            _, kinetics = self.model.predict(angles, cohort=cohort, input_set=input_set,
                                              source_id=12, severity_id=6, seeds=self.seeds, seed=42)
            result = {
                'kind': 'live_inference', 'cohort': cohort, 'input_set': input_set,
                'cycle_count': len(angles), 'cycle_ids': cycle_ids,
                'gait_percent': list(range(100)),
                'output_names': TARGET_NAMES, 'output_units': ['N·m/kg'] * 6 + ['BW'] * 2,
                'predictions': kinetics.tolist(),
                'mean_predictions': {name: kinetics[:, :, i].mean(0).tolist() for i, name in enumerate(TARGET_NAMES)},
                'mean_angles': {},
                'metadata': {**self.model.metadata(), 'conditions': list(self.model.conditions(cohort, 12, 6)),
                             'seed_values': [42 + i * 7919 for i in range(self.seeds)],
                             'input_csv_sha256': hashlib.sha256(request['csv'].encode('utf-8')).hexdigest(),
                             'input_set': input_set, 'no_reference_kinetics_used': True,
                             'evaluation': 'Uploaded data have no reference scores. Inference uses the checkpoint identified by checkpoint_sha256.'},
            }
            # Do not expose installation paths through the browser API.
            result['metadata'].pop('checkpoint', None)
            for i, name in enumerate(ANGLE_COLUMNS):
                available = np.isfinite(angles[:, :, i]).all(1)
                if available.any():
                    result['mean_angles'][name] = angles[available, :, i].mean(0).tolist()
            return result
        finally:
            self.lock.release()


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'PathoGaitLocal/1.0'

        def send(self, code, content, content_type='application/json; charset=utf-8'):
            if not isinstance(content, bytes):
                content = (content if isinstance(content, str) else json.dumps(content, allow_nan=False)).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def local_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.local_host():
                self.send(403, {'error': 'Use the local demo address.'}); return
            path = urlsplit(self.path).path
            if path == '/api/status':
                self.send(200, {'checkpoint_available': demo.checkpoint.is_file(), 'device': demo.device,
                                'seeds': demo.seeds, 'token': demo.token}); return
            if path == '/api/examples':
                self.send(200, {'examples': demo.examples()}); return
            if path == '/api/template.csv':
                self.send(200, template_csv(), 'text/csv; charset=utf-8'); return
            if path == '/api/sample.csv':
                sample = ROOT / 'demo' / 'examples' / 'synthetic_healthy_angles.csv'
                if sample.is_file():
                    self.send(200, sample.read_bytes(), 'text/csv; charset=utf-8'); return
                self.send(404, {'error': 'The synthetic example is not installed.'}); return
            if path.startswith('/api/examples/') and path.endswith('.csv'):
                example_id = path.rsplit('/', 1)[-1][:-4]
                row = next((v for v in demo.examples() if v['id'] == example_id), None)
                if row:
                    file = demo.examples_dir / (example_id + '_angles.csv')
                    if file.is_file():
                        self.send(200, file.read_bytes(), 'text/csv; charset=utf-8'); return
                self.send(404, {'error': 'Example angles are not installed.'}); return
            file = (STATIC / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
            if STATIC.resolve() not in file.parents or not file.is_file():
                self.send(404, {'error': 'Not found.'}); return
            mime = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
            self.send(200, file.read_bytes(), mime + ('; charset=utf-8' if mime.startswith('text/') else ''))

        def do_POST(self):
            if not self.local_host() or self.headers.get('X-PathoGait-Token') != demo.token:
                self.send(403, {'error': 'Open the local demo page before running inference.'}); return
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}:
                self.send(403, {'error': 'Requests must come from the local demo page.'}); return
            if urlsplit(self.path).path != '/api/predict':
                self.send(404, {'error': 'Not found.'}); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > MAX_BYTES * 2:
                    self.send(413, {'error': 'Request size is invalid or exceeds 4 MB.'}); return
                if not self.headers.get('Content-Type', '').startswith('application/json'):
                    self.send(415, {'error': 'Use a JSON prediction request.'}); return
                request = json.loads(self.rfile.read(size).decode('utf-8'))
                self.send(200, demo.predict(request))
            except (ValueError, UnicodeError) as exc:
                self.send(400, {'error': str(exc)})
            except FileNotFoundError as exc:
                self.send(503, {'error': str(exc)})
            except BlockingIOError as exc:
                self.send(409, {'error': str(exc)})
            except Exception as exc:
                print(f'Inference failed: {type(exc).__name__}: {exc}', file=sys.stderr)
                self.send(500, {'error': 'Inference failed. Check the local terminal for the cause.'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/v4_stage2_final8ch_ALLDATA/final.pt')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--examples', type=Path, help='Optional local folder containing two authorised examples.')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(Demo(args.checkpoint, args.device, args.examples)))
    print(f'PathoGaitFM demo: http://127.0.0.1:{args.port}', flush=True)
    print('Uploaded angles stay in memory on this computer. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
