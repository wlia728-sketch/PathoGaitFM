"""Write demo/examples/synthetic_healthy_angles.csv: three synthetic gait cycles of harmonised joint angles.

The curves are smooth parametric approximations of textbook sagittal, frontal and transverse joint-angle profiles
of healthy adult walking (hip and knee flexion positive, ankle dorsiflexion positive, pelvic tilt anterior positive),
in the model's channel convention and units (degrees, 100 samples per cycle, heel strike to heel strike of the same
limb). They come from no participant and no dataset; they exist so that the local demo can be tried without any
recording. Left-limb channels are the right-limb curves shifted by half a cycle.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "examples" / "synthetic_healthy_angles.csv"
COLUMNS = ["cycle_id", "gait_percent",
           "R_Hip_X", "R_Hip_Y", "R_Hip_Z", "L_Hip_X", "L_Hip_Y", "L_Hip_Z",
           "R_Knee_X", "L_Knee_X", "R_Ankle_X", "L_Ankle_X",
           "R_Pelvis_X", "R_Pelvis_Y", "R_Pelvis_Z", "L_Pelvis_X", "L_Pelvis_Y", "L_Pelvis_Z"]


def bump(t: float, centre: float, width: float, height: float) -> float:
    return height * math.exp(-0.5 * ((t - centre) / width) ** 2)


def right_limb(t: float) -> dict:
    """Angles in degrees at cycle fraction t (0 = heel strike, 1 = next heel strike of the same limb)."""
    hip_x = 10.0 + 22.0 * math.cos(2 * math.pi * (t + 0.02)) + 4.0 * math.sin(4 * math.pi * t)
    hip_y = 4.0 * math.cos(2 * math.pi * t) - 1.0
    hip_z = -4.0 + 4.0 * math.sin(2 * math.pi * (t - 0.1))
    knee_x = 5.0 + bump(t, 0.14, 0.07, 14.0) + bump(t, 0.72, 0.10, 58.0) + bump(t, 0.95, 0.06, 5.0)
    ankle_x = (-4.0 * bump(t, 0.06, 0.05, 1.0) + bump(t, 0.42, 0.14, 12.0) - bump(t, 0.63, 0.06, 18.0)
               + 2.0 * bump(t, 0.85, 0.08, 1.0) - 1.0)
    pelvis_x = 3.0 * math.sin(2 * math.pi * t + 0.4)
    pelvis_y = -0.7 + 4.0 * math.sin(2 * math.pi * t - 0.3)
    pelvis_z = 12.0 + 2.0 * math.cos(4 * math.pi * t)
    return {"Hip_X": hip_x, "Hip_Y": hip_y, "Hip_Z": hip_z, "Knee_X": knee_x, "Ankle_X": ankle_x,
            "Pelvis_X": pelvis_x, "Pelvis_Y": pelvis_y, "Pelvis_Z": pelvis_z}


def cycle(cycle_id: str, scale: float, shift: float) -> list:
    rows = []
    for i in range(100):
        t = i / 100.0
        r = right_limb(t)
        l = right_limb((t + 0.5 + shift) % 1.0)
        row = {"cycle_id": cycle_id, "gait_percent": i}
        for side, a in (("R", r), ("L", l)):
            for k, v in a.items():
                base = {"Hip_Y": -1.0, "Hip_Z": -4.0, "Ankle_X": -1.0, "Pelvis_Y": -0.7, "Pelvis_Z": 12.0}.get(k, 5.0 if k == "Knee_X" else 0.0)
                row[f"{side}_{k}"] = round(base + (v - base) * scale, 3)
        rows.append(row)
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = cycle("synthetic_1", 1.00, 0.000) + cycle("synthetic_2", 0.94, 0.006) + cycle("synthetic_3", 1.06, -0.006)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("written", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
