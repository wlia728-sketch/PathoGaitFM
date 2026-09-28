"""Verify and export the five exact manuscript images. This does not rerun inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/figures_snapshot')
    args=parser.parse_args()
    out=args.out.resolve()
    for folder in ('figures','results','data','checkpoints','model','scripts','tests','demo'):
        frozen=(ROOT/folder).resolve()
        if out==frozen or frozen in out.parents:
            raise ValueError('Choose an output outside the supplied source folders.')
    source=ROOT/'figures'
    manifest=json.loads((source/'manifest.json').read_text())
    assert [v['figure'] for v in manifest['figures']]==list(range(1,6))
    for row in manifest['figures']:
        path=source/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f'Figure checksum mismatch: {path.name}')
    out.mkdir(parents=True,exist_ok=True)
    for row in manifest['figures']:
        shutil.copy2(source/row['file'],out/row['file'])
        caption=f'Fig{row["figure"]}_caption.txt'
        shutil.copy2(source/caption,out/caption)
    shutil.copy2(source/'manifest.json',out/'manifest.json')
    print(f'Verified and exported five manuscript figures to {out}')


if __name__=='__main__':main()
