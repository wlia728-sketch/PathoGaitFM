"""Strict, non-executable CSV contract for already harmonised angle cycles."""
from __future__ import annotations
import csv
import io
import math
from collections import OrderedDict
from pathlib import Path
import sys
import numpy as np

LIB = Path(__file__).resolve().parents[1] / 'scripts/lib'
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
from input_configurations import INPUT_SETS as PAPER_INPUT_SETS

ANGLE_COLUMNS = [
    'R_Hip_X', 'R_Hip_Y', 'R_Hip_Z', 'L_Hip_X', 'L_Hip_Y', 'L_Hip_Z',
    'R_Knee_X', 'L_Knee_X', 'R_Ankle_X', 'L_Ankle_X',
    'R_Pelvis_X', 'R_Pelvis_Y', 'R_Pelvis_Z',
    'L_Pelvis_X', 'L_Pelvis_Y', 'L_Pelvis_Z',
]
COHORTS = ('normal', 'cp', 'vdk_stroke', 'bmclab_pd')
INPUT_SETS = PAPER_INPUT_SETS
MAX_CYCLES = 16
MAX_BYTES = 2 * 1024 * 1024


def parse_angles_csv(text: str):
    """No unit conversion, interpolation, segmentation or polarity guessing."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Choose a non-empty CSV file.')
    if len(text.encode('utf-8')) > MAX_BYTES:
        raise ValueError('The CSV is larger than 2 MB.')
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        raise ValueError('CSV column names must be unique.')
    allowed = set(ANGLE_COLUMNS) | {'cycle_id', 'gait_percent'}
    extra = set(headers) - allowed
    if extra:
        raise ValueError('Unexpected columns: ' + ', '.join(sorted(extra)) + '. Upload joint angles only.')
    if 'gait_percent' not in headers:
        raise ValueError('Add gait_percent with exactly 0, 1, ..., 99 for each cycle.')
    present = [name for name in ANGLE_COLUMNS if name in headers]
    if not present:
        raise ValueError('No recognised joint-angle columns. Download the CSV template for column names.')
    groups = OrderedDict()
    for index, row in enumerate(reader, 2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'Row {index} has a different number of cells from the header.')
        cycle = row.get('cycle_id', 'cycle_1').strip()
        if not cycle or len(cycle) > 80:
            raise ValueError(f'Row {index}: use a short, non-empty cycle_id.')
        if cycle not in groups and len(groups) >= MAX_CYCLES:
            raise ValueError(f'The local demo accepts at most {MAX_CYCLES} cycles per run.')
        rows = groups.setdefault(cycle, [])
        if len(rows) >= 100:
            raise ValueError(f'Cycle {cycle} has more than 100 rows; remove the duplicate 100% endpoint.')
        try:
            phase = float(row['gait_percent'])
        except (TypeError, ValueError):
            raise ValueError(f'Row {index}: gait_percent must be numeric.') from None
        if not math.isfinite(phase) or phase != len(rows):
            raise ValueError(f'Cycle {cycle}: expected gait_percent {len(rows)}, received {row["gait_percent"]}.')
        values = []
        for name in ANGLE_COLUMNS:
            value = row.get(name, '').strip()
            if value == '' or value.lower() == 'nan':
                values.append(float('nan'))
                continue
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f'Row {index}, {name}: enter degrees or leave a whole missing channel blank.') from None
            if not math.isfinite(number):
                raise ValueError(f'Row {index}, {name}: infinity is not an angle input.')
            if abs(number) > 360:
                raise ValueError(f'Row {index}, {name}: outside ±360 degrees; check units and conventions.')
            values.append(number)
        rows.append(values)
    if not groups:
        raise ValueError('The CSV contains no data rows.')
    for cycle, rows in groups.items():
        if len(rows) != 100:
            raise ValueError(f'Cycle {cycle} has {len(rows)} rows; exactly 100 are required.')
    angles = np.asarray(list(groups.values()), dtype=np.float32)
    finite = np.isfinite(angles)
    if (finite.any(1) & ~finite.all(1)).any():
        raise ValueError('A missing angle must be absent for its entire cycle; isolated missing samples are not supported.')
    if not finite.all(1).any(1).all():
        raise ValueError('Each cycle needs at least one available joint-angle channel.')
    return angles, list(groups)


def template_csv():
    """Blank template, deliberately not a synthetic participant or a prediction."""
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(['cycle_id', 'gait_percent', *ANGLE_COLUMNS])
    for i in range(100):
        writer.writerow(['cycle_1', i, *([''] * 16)])
    return out.getvalue()
