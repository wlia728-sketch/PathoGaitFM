"""Extract ONLY processed overground angle (ang) + kinetics (knt) txt from WBDSascii.zip.
These are cycle-normalized (101 pt) per subject-speed. Serial, no shell globbing."""
import zipfile, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP = HERE / "raw" / "WBDSascii.zip"
OUT = HERE / "raw" / "ascii_proc"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    counts = {"ang": 0, "knt": 0}
    for n in z.namelist():
        bn = n.split("/")[-1]
        if "walkO" in bn and (bn.endswith("ang.txt") or bn.endswith("knt.txt")):
            kind = "ang" if bn.endswith("ang.txt") else "knt"
            with z.open(n) as src, open(OUT / bn, "wb") as dst:
                dst.write(src.read())
            counts[kind] += 1
    json.dump(counts, open(HERE / "_extract_ascii_counts.json", "w"), indent=2)
    print("extracted overground processed:", counts)


if __name__ == "__main__":
    main()
