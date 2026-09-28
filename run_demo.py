"""Start the local demo with the installed checkpoint. No data are sent externally."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/v4_stage2_final8ch_ALLDATA/final.pt')
    parser.add_argument('--examples',type=Path,default=ROOT/'demo/local_examples')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    try:
        import torch
    except ImportError:
        parser.error('Install dependencies first: python -m pip install -r requirements.txt')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    command=[sys.executable,str(ROOT/'demo/server.py'),'--checkpoint',str(args.checkpoint),'--device',device,'--port',str(args.port)]
    if (args.examples/'examples.json').is_file():command+=['--examples',str(args.examples)]
    raise SystemExit(subprocess.call(command,cwd=ROOT))


if __name__=='__main__':main()
