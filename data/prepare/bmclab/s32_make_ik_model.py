"""Create per-subject 'IK model' = scaled model with ForceSet cleared.

IK doesn't need muscle forces. The default Rajagopal2015 has ~80 muscle force
elements per side; their cached state-realization is what's slowing IK down to
~14x real-time. Stripping the ForceSet drops it to ~1-2x real-time.

We do NOT mutate the original SUB{NN}.osim — instead we write SUB{NN}_ik.osim
beside it, which the batch driver uses as the IK input.
"""

from pathlib import Path
import opensim
opensim.Logger.setLevelString("Warn")

SCALED_DIR = Path(__file__).resolve().parents[3] / "data/prepare/bmclab/work/scaled_models"

def make_ik_model(scaled_path: Path, ik_path: Path) -> dict:
    m = opensim.Model(str(scaled_path))
    n_forces_before = m.getForceSet().getSize()
    n_muscles_before = m.getMuscles().getSize()
    n_constraints = m.getConstraintSet().getSize()
    n_markers = m.getMarkerSet().getSize()
    # Clear all forces (muscles, ligaments, contact forces, etc.)
    m.updForceSet().clearAndDestroy()
    n_forces_after = m.getForceSet().getSize()
    # Re-initialize and write
    m.initSystem()
    m.printToXML(str(ik_path))
    return {
        "n_forces_before": n_forces_before,
        "n_muscles_before": n_muscles_before,
        "n_forces_after": n_forces_after,
        "n_constraints_kept": n_constraints,
        "n_markers_kept": n_markers,
        "size_in_bytes": scaled_path.stat().st_size,
        "size_out_bytes": ik_path.stat().st_size,
    }


def main():
    osims = sorted(p for p in SCALED_DIR.iterdir()
                   if p.name.startswith("SUB") and p.name.endswith(".osim")
                   and "_ms" not in p.name and "_ik" not in p.name)
    print(f"Found {len(osims)} scaled models to strip.")
    for p in osims:
        ik_path = p.with_name(p.stem + "_ik.osim")
        try:
            s = make_ik_model(p, ik_path)
            print(f"  {p.name:<14} -> {ik_path.name}  "
                  f"forces {s['n_forces_before']}→{s['n_forces_after']}, "
                  f"muscles {s['n_muscles_before']} stripped, "
                  f"markers kept = {s['n_markers_kept']}, "
                  f"size {s['size_in_bytes']:,}→{s['size_out_bytes']:,} bytes")
        except Exception as e:
            print(f"  {p.name}: FAIL — {e}")


if __name__ == "__main__":
    main()
