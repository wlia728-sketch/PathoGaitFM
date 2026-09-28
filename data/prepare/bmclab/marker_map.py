"""Authoritative BMClab → Rajagopal2015 marker mapping.

The mapping decisions are recorded in the comments below.

`BMCLAB_TO_RAJAGOPAL`:
    dict[bmclab_label_in_c3d, rajagopal_name_or_None].
    Keys are the literal POINT-label strings inside the BMClab c3d (including
    the dotted style like 'R.ASIS', 'R.Knee.Medial' on the lower body and the
    plain style like 'RSHO', 'CLAV' on the trunk/upper body).
    Values are Rajagopal2015 marker names verbatim; or None for markers that
    have no Rajagopal equivalent and should not be written to TRC.

`RAJAGOPAL_UNUSED`:
    Rajagopal markers that receive no BMClab input and therefore need IK
    weight = 0 in Step 3.3 setup XML. Populated by `_verify_mapping.py` as
    `set(model_markers) - set(non-None map values)`. Initialised empty here
    and overwritten by the verification script.

`BMCLAB_DUPLICATE_LABELS`:
    POINT labels that appear more than once inside the c3d (Visual3D
    pipeline artefact). Anything in this set MUST be read by integer
    index, never by label-name lookup.
"""

BMCLAB_TO_RAJAGOPAL: dict = {
    # ===== Pelvis (4) =====
    "R.ASIS":           "RASI",
    "L.ASIS":           "LASI",
    "R.PSIS":           "RPSI",
    "L.PSIS":           "LPSI",

    # ===== Thigh — knee landmarks (2 dynamic + 2 static-only medial) =====
    "R.Knee":           "RLFC",
    "L.Knee":           "LLFC",
    "R.Knee.Medial":    "RMFC",
    "L.Knee.Medial":    "LMFC",

    # ===== Thigh — greater trochanter (no Rajagopal equivalent) =====
    "R.GTR":            None,
    "L.GTR":            None,

    # ===== Shank — tibial plateau / tuberosity =====
    # Investigation A resolved → case (i): R_tibial_plateau exists as a single
    # named marker in Rajagopal2015 MarkerSet, mapped one-to-one.
    "R.TT":             "R_tibial_plateau",
    "L.TT":             "L_tibial_plateau",

    # ===== Shank — head of fibula (no Rajagopal equivalent) =====
    "R.HF":             None,
    "L.HF":             None,

    # ===== Shank — malleoli =====
    "R.Ankle":          "RLMAL",
    "L.Ankle":          "LLMAL",
    "R.Ankle.Medial":   "RMMAL",
    "L.Ankle.Medial":   "LMMAL",

    # ===== Foot =====
    # Investigation B resolved: Rajagopal has RCAL (heel), RMT5 (5th MT, lateral)
    # and RTOE (toe tip, medial — ~5 cm distal of the true 1st-MT head). We map
    # R.MT1 → RTOE for medial-foot tracking; the Scale Tool absorbs the offset.
    # R.MT2 has no Rajagopal equivalent.
    "R.Heel":           "RCAL",
    "L.Heel":           "LCAL",
    "R.MT5":            "RMT5",
    "L.MT5":            "LMT5",
    "R.MT1":            "RTOE",
    "L.MT1":            "LTOE",
    "R.MT2":            None,
    "L.MT2":            None,

    # ===== Trunk =====
    "C7":               "C7",
    "CLAV":             "CLAV",
    "STRN":             None,
    "T10":              None,

    # ===== Shoulder (acromion) =====
    "RSHO":             "RACR",
    "LSHO":             "LACR",

    # ===== Upper-arm cluster (no single-marker Rajagopal equivalent) =====
    "RUPA":             None,
    "LUPA":             None,

    # ===== Elbow =====
    "REL":              "RLEL",
    "LEL":              "LLEL",
    "REM":              "RMEL",
    "LEM":              "LMEL",

    # ===== Forearm cluster (no single-marker Rajagopal equivalent) =====
    "RFRA":             None,
    "LFRA":             None,

    # ===== Wrist (no Rajagopal wrist markers — only forearm cluster) =====
    "RWL":              None,
    "LWL":              None,
    "RWM":              None,
    "LWM":              None,
}

# Populated by _verify_mapping.py after diffing against the model XML.
RAJAGOPAL_UNUSED: list = [
    'RASH',
    'RPSH',
    'LASH',
    'LPSH',
    'RSJC',
    'RUA1',
    'RUA2',
    'RUA3',
    'RFAsuperior',
    'RFAradius',
    'RFAulna',
    'LSJC',
    'LUA1',
    'LUA2',
    'LUA3',
    'LFAsuperior',
    'LFAradius',
    'LFAulna',
    'LHJC',
    'RHJC',
    'RTH1',
    'RTH2',
    'RTH3',
    'RKJC',
    'RTB1',
    'RTB2',
    'RTB3',
    'RAJC',
    'LTH1',
    'LTH2',
    'LTH3',
    'LKJC',
    'LTB1',
    'LTB2',
    'LTB3',
    'LAJC',
    'REJC',
    'LEJC',
]

# Duplicate POINT labels (must use index, not name).
BMCLAB_DUPLICATE_LABELS: set = {"V_R.TT"}


def mapped_target_names() -> list:
    """Return the non-None Rajagopal target names, in insertion order, deduplicated."""
    seen = set()
    out = []
    for v in BMCLAB_TO_RAJAGOPAL.values():
        if v is not None and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def n_mapped() -> int:
    return sum(1 for v in BMCLAB_TO_RAJAGOPAL.values() if v is not None)


if __name__ == "__main__":
    print(f"BMCLAB_TO_RAJAGOPAL entries: {len(BMCLAB_TO_RAJAGOPAL)}")
    print(f"  mapped (non-None):  {n_mapped()}")
    print(f"  dropped (None):     {len(BMCLAB_TO_RAJAGOPAL) - n_mapped()}")
    print(f"  unique Rajagopal targets: {len(mapped_target_names())}")
    print(f"BMCLAB_DUPLICATE_LABELS: {BMCLAB_DUPLICATE_LABELS}")
