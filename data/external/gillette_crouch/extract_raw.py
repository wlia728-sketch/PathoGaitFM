"""Extract C1..C10 _markerdata.trc + _GRF.mot + _IK.mot + _model2392.osim from the nested
CrouchGaitSimulations-latest.zip into raw/. ASCII-only stdout."""
import zipfile, io, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
outer = RAW / "CrouchGaitSimulations-latest.zip"

WANT_SUFFIX = ("_markerdata.trc", "_GRF.mot", "_IK.mot", "_model2392.osim")


def main():
    only = sys.argv[1:]  # e.g. C2 ; empty = all
    zf = zipfile.ZipFile(outer)
    inner = [n for n in zf.namelist() if n.lower().endswith(".zip")]
    n_written = 0
    for izname in inner:
        subj = Path(izname).stem  # C1..C10
        if only and subj not in only:
            continue
        iz = zipfile.ZipFile(io.BytesIO(zf.read(izname)))
        for name in iz.namelist():
            base = Path(name).name
            if base.endswith(WANT_SUFFIX):
                out = RAW / base
                out.write_bytes(iz.read(name))
                n_written += 1
                print("  wrote", base, "(%.1f KB)" % (out.stat().st_size / 1024))
    print("extracted", n_written, "files to", RAW)


if __name__ == "__main__":
    main()
