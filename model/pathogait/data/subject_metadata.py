"""Unified subject-level metadata loader for V4 unified BW normalize.

Returns body weight (kg) and optionally body height (cm) for any
(source, subject_id) tuple across the 12 V4 sources:

  AddBio (7 studies): per-subject _meta.json (`mass_kg`)
  vdk_healthy / vdk_stroke: `vdk_demographics.json` (`weight_kg`, sub_idx)
  cp / normal: in-house paediatric workbook; these two sources are absent from
             the release and get_bw raises for them, naming the reason
  bmclab_pd: `data/metadata/bmclab_metadata.json` (de-identified mass + height derived from the
             public BMClab participant file)

`subject_id` follows the naming used in cohort .npy files and AddBio
meta.json `subject_path`:

  AddBio: `{study}_{subject_split}` (e.g. `Camargo2021_AB06_split0`)
  vdk_healthy / vdk_stroke: `{source}_{NNN}` zero-padded (e.g. `vdk_healthy_000`)
  cp / normal: `{source}_{NN}` zero-padded 1-25 / 1-10 (e.g. `cp_01`)
  bmclab_pd: `SUB{NN}_{state}` (e.g. `SUB01_off`, `SUB01_on`)
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import openpyxl

ROOT = Path(__file__).resolve().parents[3]
FORENSICS_DIR = Path(os.environ.get("PATHOGAIT_METADATA_DIR", ROOT / "data" / "metadata"))
ADDBIO_DIR = ROOT / "data" / "pretraining" / "addbio"
CPNORM_XLSX = Path(os.environ.get("PATHOGAIT_INHOUSE_WORKBOOK", ROOT / "data" / "inhouse" / "基本信息.xlsx"))
BMCLAB_XLSX = Path(os.environ.get("PATHOGAIT_PD_WORKBOOK", FORENSICS_DIR / "PDGinfo.xlsx"))
BMCLAB_JSON = FORENSICS_DIR / "bmclab_metadata.json"
VDK_JSON = FORENSICS_DIR / "vdk_demographics.json"
ADDBIO_JSON = FORENSICS_DIR / "addbio_meta_summary.json"

GRAVITY_MS2 = 9.80665   # standard gravity, exact for force unit conversion


class V4SubjectMetadata:
    """Unified BW/BH metadata across 12 V4 sources.

    Use `get_bw(source, subject_id)` -> kg (raises KeyError if missing).
    Use `coverage_report()` -> per-source coverage breakdown.
    """

    def __init__(self):
        self.unavailable: list = []   # human-readable note per source that could not be loaded
        self.bw_table: Dict[Tuple[str, str], float] = {}
        self.bh_table: Dict[Tuple[str, str], float] = {}    # cm
        # v4.1 — Phase 3 extended demographics + clinical scores
        self.age_table: Dict[Tuple[str, str], float] = {}            # years
        self.sex_table: Dict[Tuple[str, str], str] = {}              # 'M' / 'F'
        self.bmi_table: Dict[Tuple[str, str], float] = {}            # kg/m^2
        self.clinical_table: Dict[Tuple[str, str], Dict] = {}        # PD only (bmclab_pd)

        self._load_addbio()
        self._load_vdk()
        self._load_cohort_cp_normal()
        self._load_bmclab()

        # Extended loaders (Phase 3). AddBio has no age/sex/height — not extended.
        self._load_vdk_extended()
        self._load_cohort_cp_normal_extended()
        self._load_bmclab_extended()

    def _note_unavailable(self, msg: str) -> None:
        """Record, once, that a source could not be loaded, so get_bw can say why."""
        if msg not in self.unavailable:
            self.unavailable.append(msg)

    # ---------- AddBio (7 studies) ----------

    def _load_addbio(self) -> None:
        if not ADDBIO_JSON.exists():
            self._note_unavailable(f"pretraining-corpus anthropometry: {ADDBIO_JSON.name} not present")
            return
        with open(ADDBIO_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for study, entries in data.items():
            src = f"addbio_{study}"
            for e in entries:
                sid = e["subject_id"]
                self.bw_table[(src, sid)] = float(e["mass_kg"])
                if e.get("height_m") is not None:
                    self.bh_table[(src, sid)] = float(e["height_m"]) * 100.0

    # ---------- vdk_healthy + vdk_stroke ----------

    def _load_vdk(self) -> None:
        if not VDK_JSON.exists():
            self._note_unavailable(f"vdk_healthy / vdk_stroke anthropometry: {VDK_JSON.name} not present")
            return
        with open(VDK_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for source, entries in [("vdk_healthy", data["vdk_healthy"]),
                                  ("vdk_stroke",  data["vdk_stroke"])]:
            for e in entries:
                idx0 = int(e["sub_idx"]) - 1     # 1-indexed in JSON -> 0-indexed in filename
                sid = f"{source}_{idx0:03d}"
                self.bw_table[(source, sid)] = float(e["weight_kg"])
                # height in MAT is mm -> cm
                self.bh_table[(source, sid)] = float(e["height_cm"]) / 10.0

    # ---------- cp + normal ----------

    def _load_cohort_cp_normal(self) -> None:
        try:
            wb = openpyxl.load_workbook(CPNORM_XLSX, data_only=True)
        except (FileNotFoundError, OSError):
            self._note_unavailable(
                "cp / normal body mass and demographics: in-house paediatric workbook not present")
            return
        for sheet, source in [("cp", "cp"), ("normal", "normal")]:
            if sheet not in wb.sheetnames:
                continue
            ws = wb[sheet]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] is None:
                    continue
                try:
                    subj_num = int(row[0])
                except (TypeError, ValueError):
                    continue
                # Schema: col1=ID, col5=Height(cm), col6=Weight(kg) per inventory
                height = row[4] if len(row) > 4 else None
                weight = row[5] if len(row) > 5 else None
                if weight is None:
                    continue
                sid = f"{source}_{subj_num:02d}"
                self.bw_table[(source, sid)] = float(weight)
                if height is not None:
                    self.bh_table[(source, sid)] = float(height)

    # ---------- bmclab_pd ----------

    def _load_bmclab(self) -> None:
        """Body mass and height for bmclab_pd.

        Prefers the shipped de-identified derivative, which carries only subject id, mass and height.
        The source workbook it was derived from also carries age, sex, disease duration, medication dose,
        family and surgical history and the full ON/OFF clinical batteries, none of which this work needs
        or reproduces. Falls back to that workbook when a user has placed it, and otherwise fails with the
        filename and the rebuild recipe rather than leaving the table silently short.
        """
        if BMCLAB_JSON.exists():
            doc = json.loads(BMCLAB_JSON.read_text(encoding="utf-8"))
            for sid_root, rec in doc.get("subjects", {}).items():
                for state in ("off", "on"):
                    sid = f"{sid_root}_{state}"
                    self.bw_table[("bmclab_pd", sid)] = float(rec["mass_kg"])
                    if rec.get("height_cm") is not None:
                        self.bh_table[("bmclab_pd", sid)] = float(rec["height_cm"])
            return
        if not BMCLAB_XLSX.exists():
            self._note_unavailable(
                f"bmclab_pd anthropometry: neither {BMCLAB_JSON.name} nor {BMCLAB_XLSX.name} is present "
                f"under {BMCLAB_JSON.parent.name}/ (BMClab release, doi:10.6084/m9.figshare.14896881)")
            return
        wb = openpyxl.load_workbook(BMCLAB_XLSX, data_only=True)
        ws = wb["PDGinfo"] if "PDGinfo" in wb.sheetnames else wb.active
        # Schema: ID (col 1), Gender, Age, Height (cm) (col 4), Weight (kg) (col 5)
        header = [c.value for c in ws[1]]
        try:
            ix_id = header.index("ID")
            ix_h  = header.index("Height (cm)")
            ix_w  = header.index("Weight (kg)")
        except ValueError:
            ix_id, ix_h, ix_w = 0, 3, 4
        for row in ws.iter_rows(min_row=2, values_only=True):
            sid_root = row[ix_id]
            if sid_root is None:
                continue
            sid_root = str(sid_root).strip()
            weight = row[ix_w]
            height = row[ix_h]
            if weight is None:
                continue
            # bmclab_pd has off + on states per subject -> same BW
            for state in ("off", "on"):
                sid = f"{sid_root}_{state}"
                self.bw_table[("bmclab_pd", sid)] = float(weight)
                if height is not None:
                    self.bh_table[("bmclab_pd", sid)] = float(height)

    # ---------- Phase 3 extended loaders (age / sex / BMI / clinical) ----------

    @staticmethod
    def _coerce_num(v) -> Optional[float]:
        """Coerce '-' / '' / 'N/A' / None -> None; numeric strings -> float."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if v in ("-", "", "N/A", "n/a", "NA"):
                return None
            try:
                return float(v)
            except ValueError:
                return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _set_bmi_from_bw_bh(self, source: str, sid: str) -> None:
        """Compute BMI from BW (kg) and BH (cm) already in tables."""
        bw = self.bw_table.get((source, sid))
        bh = self.bh_table.get((source, sid))
        if bw and bh and bh > 0:
            self.bmi_table[(source, sid)] = float(bw / ((bh / 100.0) ** 2))

    def _load_vdk_extended(self) -> None:
        """Add age + sex to vdk_healthy / vdk_stroke (BW + BH loaded earlier)."""
        if not VDK_JSON.exists():
            return
        with open(VDK_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for source, entries in [("vdk_healthy", data["vdk_healthy"]),
                                  ("vdk_stroke",  data["vdk_stroke"])]:
            for e in entries:
                idx0 = int(e["sub_idx"]) - 1
                sid = f"{source}_{idx0:03d}"
                if e.get("age") is not None:
                    self.age_table[(source, sid)] = float(e["age"])
                if e.get("male") is not None:
                    self.sex_table[(source, sid)] = "M" if float(e["male"]) >= 0.5 else "F"
                self._set_bmi_from_bw_bh(source, sid)

    def _load_cohort_cp_normal_extended(self) -> None:
        """Add age + sex to cp / normal (in-house demographics workbook)."""
        SEX_MAP = {"M": "M", "F": "F", "m": "M", "f": "F"}
        try:
            wb = openpyxl.load_workbook(CPNORM_XLSX, data_only=True)
        except (FileNotFoundError, OSError):
            self._note_unavailable(
                "cp / normal body mass and demographics: in-house paediatric workbook not present")
            return
        for sheet, source in [("cp", "cp"), ("normal", "normal")]:
            if sheet not in wb.sheetnames:
                continue
            ws = wb[sheet]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] is None:
                    continue
                try:
                    subj_num = int(row[0])
                except (TypeError, ValueError):
                    continue
                sid = f"{source}_{subj_num:02d}"
                sex_raw = row[2] if len(row) > 2 else None
                age_raw = row[3] if len(row) > 3 else None
                if sex_raw is not None:
                    key = str(sex_raw).strip()
                    if key in SEX_MAP:
                        self.sex_table[(source, sid)] = SEX_MAP[key]
                if age_raw is not None:
                    try:
                        self.age_table[(source, sid)] = float(age_raw)
                    except (TypeError, ValueError):
                        pass
                self._set_bmi_from_bw_bh(source, sid)

    def _load_bmclab_extended(self) -> None:
        """Add age + sex + BMI + Hoehn-Yahr + UPDRS-III for bmclab_pd.

        Note: bmclab missing values are dash strings ('-') — coerced to None.
        Header strings are taken verbatim from the source workbook (some have
        trailing spaces, e.g. 'ON - Hoehn & Yahr ').
        """
        if not BMCLAB_XLSX.exists():
            self._note_unavailable(
                "bmclab_pd age, sex and clinical scores: source participant workbook not present")
            return
        wb = openpyxl.load_workbook(BMCLAB_XLSX, data_only=True)
        ws = wb["PDGinfo"] if "PDGinfo" in wb.sheetnames else wb.active
        header = [c.value for c in ws[1]]

        def _ix(*candidates) -> Optional[int]:
            for name in candidates:
                if name in header:
                    return header.index(name)
            return None

        ix_id     = _ix("ID")
        ix_gender = _ix("Gender")
        ix_age    = _ix("Age")
        ix_bmi    = _ix("BMI (kg/m2)", "BMI")
        # Trailing space in workbook is real — keep it as the primary key.
        ix_hy_off   = _ix("OFF - Hoehn & Yahr ", "OFF - Hoehn & Yahr")
        ix_hy_on    = _ix("ON - Hoehn & Yahr ",  "ON - Hoehn & Yahr")
        ix_upd3_off = _ix("OFF - UPDRS-III")
        ix_upd3_on  = _ix("ON - UPDRS-III")

        for row in ws.iter_rows(min_row=2, values_only=True):
            sid_root = row[ix_id] if ix_id is not None else row[0]
            if sid_root is None:
                continue
            sid_root = str(sid_root).strip()

            age_val = self._coerce_num(row[ix_age]) if ix_age is not None else None
            bmi_val = self._coerce_num(row[ix_bmi]) if ix_bmi is not None else None
            hy_off  = self._coerce_num(row[ix_hy_off])   if ix_hy_off   is not None else None
            hy_on   = self._coerce_num(row[ix_hy_on])    if ix_hy_on    is not None else None
            upd3_off = self._coerce_num(row[ix_upd3_off]) if ix_upd3_off is not None else None
            upd3_on  = self._coerce_num(row[ix_upd3_on])  if ix_upd3_on  is not None else None

            sex_str = None
            if ix_gender is not None and row[ix_gender] is not None:
                g = str(row[ix_gender]).strip().upper()
                if g.startswith("M"):
                    sex_str = "M"
                elif g.startswith("F"):
                    sex_str = "F"

            for state in ("off", "on"):
                sid = f"{sid_root}_{state}"
                if sex_str:
                    self.sex_table[("bmclab_pd", sid)] = sex_str
                if age_val is not None:
                    self.age_table[("bmclab_pd", sid)] = age_val
                if bmi_val is not None:
                    self.bmi_table[("bmclab_pd", sid)] = bmi_val
                clinical: Dict[str, float] = {}
                if state == "off":
                    if hy_off   is not None: clinical["hoehn_yahr"] = hy_off
                    if upd3_off is not None: clinical["updrs_iii"]  = upd3_off
                else:
                    if hy_on   is not None: clinical["hoehn_yahr"] = hy_on
                    if upd3_on is not None: clinical["updrs_iii"]  = upd3_on
                if clinical:
                    self.clinical_table[("bmclab_pd", sid)] = clinical

    # ---------- API ----------

    def get_bw(self, source: str, subject_id: str) -> float:
        key = (source, subject_id)
        if key not in self.bw_table:
            why = "; ".join(self.unavailable) if self.unavailable else "no source was skipped at load time"
            raise KeyError(f"No BW metadata for {source}/{subject_id} ({why})")
        return self.bw_table[key]

    def get_bh(self, source: str, subject_id: str) -> Optional[float]:
        return self.bh_table.get((source, subject_id))

    def get_age(self, source: str, subject_id: str) -> Optional[float]:
        return self.age_table.get((source, subject_id))

    def get_sex(self, source: str, subject_id: str) -> Optional[str]:
        return self.sex_table.get((source, subject_id))

    def get_bmi(self, source: str, subject_id: str) -> Optional[float]:
        return self.bmi_table.get((source, subject_id))

    def get_clinical(self, source: str, subject_id: str) -> Optional[Dict]:
        return self.clinical_table.get((source, subject_id))

    def has(self, source: str, subject_id: str) -> bool:
        return (source, subject_id) in self.bw_table

    def coverage_report(self) -> Dict[str, Dict[str, int]]:
        """Group subjects by source for quick coverage tally."""
        by_source: Dict[str, set] = {}
        for src, sid in self.bw_table:
            by_source.setdefault(src, set()).add(sid)
        return {src: {"n_subjects": len(sids),
                       "n_with_bh": sum(1 for s in sids
                                          if (src, s) in self.bh_table)}
                for src, sids in sorted(by_source.items())}

    def coverage_report_extended(self) -> Dict[str, Dict[str, int]]:
        """Extended coverage incl. age / sex / BMI / clinical scores."""
        by_source: Dict[str, set] = {}
        for src, sid in self.bw_table:
            by_source.setdefault(src, set()).add(sid)
        out: Dict[str, Dict[str, int]] = {}
        for src, sids in sorted(by_source.items()):
            out[src] = {
                "n_subjects":      len(sids),
                "n_with_bh":       sum(1 for s in sids if (src, s) in self.bh_table),
                "n_with_age":      sum(1 for s in sids if (src, s) in self.age_table),
                "n_with_sex":      sum(1 for s in sids if (src, s) in self.sex_table),
                "n_with_bmi":      sum(1 for s in sids if (src, s) in self.bmi_table),
                "n_with_clinical": sum(1 for s in sids if (src, s) in self.clinical_table),
            }
        return out


# ---------- Subject ID derivation from .npy filename ----------

_VDK_RE  = re.compile(r"^(vdk_healthy|vdk_stroke)_(\d{3})$")
_CP_RE   = re.compile(r"^(cp|normal)_(\d{2})$")
_BMCL_RE = re.compile(r"^(SUB\d{2})_(off|on)$")


def subject_id_from_filename(source: str, filename_stem: str) -> str:
    """Derive metadata subject_id key from .npy filename stem.

    For AddBio, the filename stem already matches `Camargo2021_AB06_split0`.
    For cohort, returns the same stem unchanged.
    """
    return filename_stem
