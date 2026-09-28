"""Apply Issue A — GRF cyclic swap to all bmclab_pd .npy files.

Per Wenqi's decision:
  BMClab vx (forward) → 4-cohort ch 35 (Y, forward)
  BMClab vy (up)      → 4-cohort ch 36 (Z, up)
  BMClab vz (lateral) → 4-cohort ch 34 (X, lateral)

Lateral sign verified positive on R stance peak (matches cohort consensus,
no extra sign flip needed).

Also re-write the per-cycle masks so that the new channel's mask reflects
the old source channel's mask:
  new ch 34 mask ← old ch 36 mask
  new ch 35 mask ← old ch 34 mask
  new ch 36 mask ← old ch 35 mask
Same logic for ch 37, 38, 39 (L side).

The mask values today are identical for ch 34/35/36 within a cycle (all
three R-GRF channels are gated by the same R-side + grf_presence rule).
But we still update them defensively in case a future change makes them
diverge.
"""
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
OUT = PROJ / "data/cohorts_raw/bmclab_pd"

def cyclic_swap_grf_block(arr_3d, base):
    """In-place cyclic swap of channels (base, base+1, base+2) along axis 2.
    new[base]   = old[base+2]   (vz → lateral)
    new[base+1] = old[base]     (vx → forward)
    new[base+2] = old[base+1]   (vy → vertical)
    """
    new_lat  = arr_3d[..., base + 2].copy()  # vz → lateral
    new_fwd  = arr_3d[..., base].copy()      # vx → forward
    new_vert = arr_3d[..., base + 1].copy()  # vy → vertical
    arr_3d[..., base]     = new_lat
    arr_3d[..., base + 1] = new_fwd
    arr_3d[..., base + 2] = new_vert
    return arr_3d


def cyclic_swap_mask_block(arr_2d, base):
    """Same cyclic swap on the mask (N, 54) for the 3 GRF channels starting at base."""
    new_lat  = arr_2d[..., base + 2].copy()
    new_fwd  = arr_2d[..., base].copy()
    new_vert = arr_2d[..., base + 1].copy()
    arr_2d[..., base]     = new_lat
    arr_2d[..., base + 1] = new_fwd
    arr_2d[..., base + 2] = new_vert
    return arr_2d


def main():
    data_files = sorted(p for p in OUT.glob("*.npy") if not p.name.endswith("_mask.npy"))
    print(f"Applying GRF cyclic swap to {len(data_files)} subject-state files...")
    n_total_cyc = 0
    for d_path in data_files:
        m_path = d_path.with_name(d_path.stem + "_mask.npy")
        d = np.load(d_path)
        m = np.load(m_path)
        if d.shape[1:] != (101, 54) or m.shape[1] != 54:
            print(f"  skip shape {d.shape}/{m.shape}: {d_path.name}")
            continue
        # R-side block: ch 34-36
        d = cyclic_swap_grf_block(d, 34)
        m = cyclic_swap_mask_block(m, 34)
        # L-side block: ch 37-39
        d = cyclic_swap_grf_block(d, 37)
        m = cyclic_swap_mask_block(m, 37)
        np.save(d_path, d)
        np.save(m_path, m)
        n_total_cyc += d.shape[0]
    print(f"Done. Swapped {n_total_cyc} cycles across {len(data_files)} files.")


if __name__ == "__main__":
    main()
