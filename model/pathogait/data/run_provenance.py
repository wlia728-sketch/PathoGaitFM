"""Persist effective training settings and identities of the actual input files."""
import json
import math
import os
from pathlib import Path
from .zstats_provenance import sha256_file

DEFAULTS = {
    'PATHOGAIT_DYNCYCLE_POOL': '', 'PATHOGAIT_INPUT_DROP_CH': '',
    'PATHOGAIT_LOSS_TARGET_CH': '', 'PATHOGAIT_EMG_DROPOUT_CH': '',
    'PATHOGAIT_EMG_DROPOUT_P': '0.5', 'PATHOGAIT_AUX_LOSS_CH': '',
    'PATHOGAIT_AUX_LOSS_W': '0.3',
}


def validate_training_scope():
    """Reject retired training variants before they can change a reported run."""
    unsupported = [key for key in
                   ('PATHOGAIT_EMG_DROPOUT_CH', 'PATHOGAIT_AUX_LOSS_CH')
                   if os.environ.get(key, '')]
    if unsupported:
        raise ValueError(
            'This manuscript package excludes EMG modality-dropout and '
            'auxiliary-channel loss experiments. Unset: ' + ', '.join(unsupported))


def capture_settings(args, root):
    validate_training_scope()
    root = Path(root)
    env = {k: os.environ.get(k, v) for k, v in DEFAULTS.items()}
    for key, value in env.items():
        if key.endswith('_CH') and value:
            channels = [int(c) for c in value.split(',') if c.strip()]
            if len(set(channels)) != len(channels) or any(c < 0 or c >= 40 for c in channels):
                raise ValueError(f'{key}: expected unique channel indices in [0, 39]')
    if env['PATHOGAIT_DYNCYCLE_POOL'] not in ('', '0', '1'):
        raise ValueError('PATHOGAIT_DYNCYCLE_POOL must be 0 or 1')
    if not 0 <= float(env['PATHOGAIT_EMG_DROPOUT_P']) <= 1:
        raise ValueError('EMG dropout probability must be in [0,1]')
    weight=float(env['PATHOGAIT_AUX_LOSS_W'])
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Auxiliary loss weight must be nonnegative')
    env['PATHOGAIT_COHORT_RAW_DIR'] = str(Path(os.environ.get(
        'PATHOGAIT_COHORT_RAW_DIR', root/'data/cohorts_raw')).resolve())
    for key, default in [('PATHOGAIT_METADATA_DIR',root/'data/metadata'),
                         ('PATHOGAIT_INHOUSE_WORKBOOK',root/'data/inhouse/基本信息.xlsx'),
                         ('PATHOGAIT_PD_WORKBOOK',Path(os.environ.get('PATHOGAIT_METADATA_DIR',root/'data/metadata'))/'PDGinfo.xlsx')]:
        env[key]=str(Path(os.environ.get(key,default)).resolve())
    refs = {}
    for key in ('split','zstats','severity_label','per_subject_flip_paper',
                'per_subject_flip_addbio','emg_condition_json','resume'):
        value = getattr(args,key,None)
        if value:
            p = Path(value).resolve()
            if not p.is_file():
                raise FileNotFoundError(f'{key}: {p}')
            refs[key] = {'path':str(p),'sha256':sha256_file(p)}
    metadata_files=list(Path(env['PATHOGAIT_METADATA_DIR']).glob('*.json'))
    metadata_files += [Path(env['PATHOGAIT_INHOUSE_WORKBOOK']),Path(env['PATHOGAIT_PD_WORKBOOK'])]
    for p in metadata_files:
        if p.is_file():refs['metadata:'+p.name]={'path':str(p.resolve()),'sha256':sha256_file(p)}
    args.run_provenance = {'schema_version':1,'environment':env,'reference_files':refs}
    return args.run_provenance


def save_training_manifest(args, train_ds, val_ds, out_dir):
    """Hash loaded train/validation array and mask files once before training."""
    cache = {}
    files = {}
    for label, ds in (('train',train_ds),('validation',val_ds)):
        rows = []
        for source, paths in sorted(ds.source_files.items()):
            for filename in sorted(paths):
                p = Path(filename).resolve()
                entry = {'source':source,'array':str(p)}
                for kind, f in (('array',p),('mask',p.with_name(p.stem+'_mask.npy'))):
                    if not f.is_file():
                        raise FileNotFoundError(f'Missing training {kind}: {f}')
                    if str(f) not in cache:
                        cache[str(f)] = sha256_file(f)
                    entry[kind+'_sha256'] = cache[str(f)]
                rows.append(entry)
        files[label] = rows
    out = Path(out_dir)
    payload = {'settings':args.run_provenance,'data_files':files}
    p = out/'training_manifest.json'
    p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    args.run_provenance['data_manifest_sha256'] = sha256_file(p)
    args.run_provenance['data_manifest_file'] = p.name
    (out/'args.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
