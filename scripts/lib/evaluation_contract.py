"""Fail before inference when a CV5 run cannot cover its declared population."""
import hashlib
from pathlib import Path
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def preflight_cv5(split, raw_root, severity, checkpoints, excluded=()):
    if len(split['cv5']) != 5:
        raise ValueError('CV5 requires exactly five folds')
    missing=[str(p) for p in checkpoints if not Path(p).is_file()]
    if missing:raise FileNotFoundError('Missing required checkpoints: '+', '.join(missing))
    seen=set(); expected={}; excluded_no_valid=[]; identities={}
    for i, fold in enumerate(split['cv5']):
        train={tuple(p) for p in fold['train']}; val=[tuple(p) for p in fold['val']]
        if len(set(val)) != len(val) or train.intersection(val) or seen.intersection(val):
            raise ValueError(f'Duplicate/overlapping recording IDs in CV5 fold {i}')
        seen.update(val)
        admitted=set()
        for source,stem in val:
            if stem in excluded:continue
            p=Path(raw_root)/source/(stem+'.npy'); mask=p.with_name(stem+'_mask.npy')
            for f in (p,mask):
                if not f.is_file():raise FileNotFoundError(f'Missing evaluation input: {f}')
                identities[str(f.resolve())]=sha256(f)
            label=severity['mappings'].get(source,{}).get(stem)
            if label is None:raise ValueError(f'Missing severity label: {source}/{stem}')
            cohort={0:'cp',1:'normal',2:'vdk_stroke',3:'bmclab_pd'}.get(int(label['cohort_id']))
            if cohort!=source:raise ValueError(f'Cohort label mismatch: {source}/{stem}')
            x=np.load(p,mmap_mode='r');m=np.load(mask).astype(bool)
            if x.ndim!=3 or x.shape[1]<100 or x.shape[2]!=54:
                raise ValueError(f'Invalid array shape at {p}: {x.shape}')
            if m.ndim==1:m=np.tile(m,(len(x),1))
            if m.shape!=(len(x),54):raise ValueError(f'Mask/array shape mismatch: {p}')
            eligible=False
            for side in ([18,24,26,36],[21,25,27,39]):
                if source=='bmclab_pd':side=side[-1:]
                keep=m[:,side].all(axis=1)
                if not np.isfinite(x[keep,:100,:][:,:,side]).all():
                    raise ValueError(f'Nonfinite target marked valid: {p}')
                eligible |= bool(keep.any())
            if eligible:admitted.add((source,stem))
            else:excluded_no_valid.append([source,stem])
        expected[f'cv5_{i}']=admitted
    if not all(expected.values()):raise ValueError('At least one CV5 fold has no eligible evaluation units')
    report={'folds':list(expected),'eligible_units':sum(map(len,expected.values())),
            'excluded_no_valid_targets':excluded_no_valid,'input_sha256':identities,
            'checkpoint_sha256':{str(Path(p).resolve()):sha256(p) for p in checkpoints}}
    return expected,report


def validate_coverage(expected, actual):
    if set(actual)!=set(expected):
        raise RuntimeError('Evaluation coverage differs: missing=%s extra=%s' %
                           (sorted(set(expected)-set(actual)),sorted(set(actual)-set(expected))))


def safe_result_path(path, root):
    p=Path(path).resolve(); root=Path(root).resolve()
    # Never replace an existing archived result, including through an alias/symlink.
    frozen=root/'results'
    if p==frozen or (frozen in p.parents and (frozen/'recomputed') not in p.parents):
        raise ValueError('Choose outputs/ or results/recomputed/; archived results are protected')
    if p.suffix.lower() != '.json':raise ValueError('Result output must be a JSON file')
    allowed=[(root/'outputs').resolve(),(root/'results/recomputed').resolve()]
    if root in p.parents and not any(folder in p.parents for folder in allowed):
        raise ValueError('Choose outputs/ or results/recomputed/ inside the repository')
    return p
