"""Figure output directory and the local dump directory used by the figure scripts."""
import csv
import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def output_dir(path=None):
    folder = Path(path if path is not None else os.environ.get("PATHOGAIT_FIG_DIR", str(ROOT / "outputs/figures"))).resolve()
    frozen = (ROOT / "figures").resolve()
    if folder == frozen or frozen in folder.parents:
        raise ValueError("Choose a new figure directory; published renderings are frozen")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "panel_data").mkdir(exist_ok=True)
    return folder


def dumps_dir():
    return Path(os.environ.get("PATHOGAIT_DUMPS", str(ROOT / "outputs/dumps"))).resolve()


def first_csv_table(path):
    text = Path(path).read_text(encoding="utf-8").split("\n\n", 1)[0]
    return list(csv.DictReader(io.StringIO(text)))
