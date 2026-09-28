"""Reading a per-site harmonisation summary.

The summaries under data/external/*/processed/ record what the harmonisation produced: the source release's own
pseudonym, the cycle count, the per-channel validity counts and, for the Parkinson site, the polarity
correlations. They do not carry body mass. Mass is a participant measurement, so this package ships
none; the harmonisation script writes it when a licensed user rebuilds the site.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_site_summary(path):
    """The record list, whether the file is a bare list or a {_note, subjects} object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            "%s is not present. Rebuild this site with its harmonisation script from the source "
            "release named in the site README; see README, Dataset layout." % p)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        if isinstance(doc.get("subjects"), list):
            return doc["subjects"]
        return [v for k, v in doc.items() if not k.startswith("_") and isinstance(v, dict)]
    raise ValueError("%s is neither a list nor an object of records" % path)


def site_mass(record, path):
    """Body mass for one record, or a message naming what to rebuild."""
    mass = record.get("mass")
    if mass is None:
        sid = record.get("sid") or record.get("subj") or "?"
        raise KeyError(
            "%s carries no body mass for %s. Mass is a participant measurement and is not shipped. "
            "Rebuild this site with its harmonisation script, which writes the summary including mass; "
            "see the site README and README, Dataset layout." % (Path(path).name, sid))
    return float(mass)
