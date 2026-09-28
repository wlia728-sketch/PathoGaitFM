"""Check checkpoint sizes and SHA-256 against the release manifest, without loading pickle."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--run", action="append", help="verify only this run; repeatable")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checkpoints/weights.json").read_text(encoding="utf-8"))
    runs = args.run or list(manifest["files"])
    failed = False
    for run in runs:
        if run not in manifest["files"]:
            parser.error(f"unknown run {run}")
        info, path = manifest["files"][run], args.root / run / "final.pt"
        if not path.is_file():
            print("MISSING", path)
            failed = True
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        valid = path.stat().st_size == info["bytes"] and digest.hexdigest() == info["sha256"]
        print("PASS" if valid else "FAIL", run)
        failed |= not valid
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
