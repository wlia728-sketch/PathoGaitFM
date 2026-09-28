"""Joint power computation for Step 5.

For each cycle, compute 6 joint powers (W/kg = N·m/kg × rad/s):

  hip_flex_power_r  = hip_flexion_r_moment  × d(hip_flexion_r) /dt
  hip_flex_power_l  = hip_flexion_l_moment  × d(hip_flexion_l) /dt
  knee_power_r      = knee_angle_r_moment   × d(knee_angle_r)  /dt
  knee_power_l      = knee_angle_l_moment   × d(knee_angle_l)  /dt
  ankle_power_r     = ankle_angle_r_moment  × d(ankle_angle_r) /dt
  ankle_power_l     = ankle_angle_l_moment  × d(ankle_angle_l) /dt

Input is already time-normalized to 101 points (per cycle), so
dt = cycle_duration_sec / 100 (we go from index 0 to 100 = 100 intervals).

Angle is in degrees → convert to radians before differentiating.
Apply a 6-Hz low-pass filter to ω matching the OpenSim ID LP cutoff to
keep power smooth. With 101 points spanning ~1 s, effective sample rate
~100 Hz, so 6-Hz LP is reasonable.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def _lowpass_filter(sig: np.ndarray, fs_hz: float, cutoff_hz: float = 6.0, order: int = 4):
    """Zero-phase Butterworth LP. Returns sig unmodified if signal is too short."""
    if sig.size < 3 * order: return sig.copy()
    nyq = 0.5 * fs_hz
    if cutoff_hz >= nyq * 0.99:  # signal too short to filter sensibly
        return sig.copy()
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    pad = min(3 * order, sig.size - 1)
    return filtfilt(b, a, sig, padlen=pad)


def compute_powers(cycle_data: np.ndarray, cycle_duration_sec: float) -> np.ndarray:
    """Compute 6 power channels from the (34, 101) BMClab cycle array.

    Layout in cycle_data (per `s4_cycle_extraction.ALL_CHANNELS`):
      [ 3] hip_flexion_r          [11] ankle_angle_r       [18] hip_flexion_r_moment
      [ 4] hip_flexion_l          [12] ankle_angle_l       [19] hip_flexion_l_moment
      [ 9] knee_angle_r                                    [22] knee_angle_r_moment
      [10] knee_angle_l                                    [23] knee_angle_l_moment
                                                            [24] ankle_angle_r_moment
                                                            [25] ankle_angle_l_moment

    NOTE on indices: Cycle .npy uses the ALL_CHANNELS list
    [angles 0-17][moments 18-27][grf 28-33]. Within angles, the order is:
      [0] pelvis_tilt, [1] pelvis_list, [2] pelvis_rotation,
      [3] hip_flexion_r, [4] hip_flexion_l,
      [5] hip_adduction_r, [6] hip_adduction_l,
      [7] hip_rotation_r, [8] hip_rotation_l,
      [9] knee_angle_r, [10] knee_angle_l,
      [11] ankle_angle_r, [12] ankle_angle_l,
      [13] subtalar_angle_r, [14] subtalar_angle_l,
      [15] lumbar_extension, [16] lumbar_bending, [17] lumbar_rotation.
    Moments [18-27]:
      [18] hip_flexion_r_moment, [19] hip_flexion_l_moment,
      [20] hip_adduction_r_moment, [21] hip_adduction_l_moment,
      [22] knee_angle_r_moment, [23] knee_angle_l_moment,
      [24] ankle_angle_r_moment, [25] ankle_angle_l_moment,
      [26] subtalar_angle_r_moment, [27] subtalar_angle_l_moment.

    Returns (6, 101) array:
      [0] hip_flex_power_r,  [1] hip_flex_power_l,
      [2] knee_power_r,      [3] knee_power_l,
      [4] ankle_power_r,     [5] ankle_power_l.

    If `cycle_duration_sec` is not finite or <=0, returns all-NaN.
    """
    if not np.isfinite(cycle_duration_sec) or cycle_duration_sec <= 0:
        return np.full((6, 101), np.nan, dtype=np.float32)
    dt = cycle_duration_sec / 100.0  # 101 samples → 100 intervals
    fs = 1.0 / dt

    # Angles in deg → convert to rad before differentiating
    pairs = [
        # (angle_idx, moment_idx)
        (3,  18),  # hip_flex_r
        (4,  19),  # hip_flex_l
        (9,  22),  # knee_r
        (10, 23),  # knee_l
        (11, 24),  # ankle_r
        (12, 25),  # ankle_l
    ]
    out = np.zeros((6, 101), dtype=np.float32)
    for k, (a_idx, m_idx) in enumerate(pairs):
        angle_deg = cycle_data[a_idx, :].astype(np.float64)
        moment = cycle_data[m_idx, :].astype(np.float64)
        if not (np.isfinite(angle_deg).all() and np.isfinite(moment).all()):
            out[k, :] = np.nan
            continue
        angle_rad = np.deg2rad(angle_deg)
        # numerical derivative (central differences) → angular velocity (rad/s)
        omega = np.gradient(angle_rad, dt)
        # 6-Hz LP to smooth differentiation noise
        omega_f = _lowpass_filter(omega, fs, cutoff_hz=6.0)
        out[k, :] = (moment * omega_f).astype(np.float32)
    return out


if __name__ == "__main__":
    # Smoke test on a real cycle .npy
    import sys
    from pathlib import Path
    PROJ = Path(__file__).resolve().parents[3]
    sample = PROJ / "data/prepare/bmclab/work/cycles/SUB02/off_walk_8/cycle_0_r.npy"
    cyc = np.load(sample)
    # Look up duration from metadata
    import pandas as pd
    meta = pd.read_parquet(PROJ / "data/prepare/bmclab/work/cycles/_cycle_metadata.parquet")
    row = meta[(meta["trial"] == "SUB02_off_walk_8") & (meta["side"] == "r") & (meta["cycle_idx_in_trial_side"] == 0)]
    if row.empty:
        print("could not find metadata for sample; skipping smoke test")
        sys.exit(0)
    dur = float(row["cycle_duration_sec"].iloc[0])
    p = compute_powers(cyc, dur)
    print(f"sample: {sample.name}  duration = {dur:.3f}s")
    labels = ["hip_flex_power_r", "hip_flex_power_l", "knee_power_r", "knee_power_l", "ankle_power_r", "ankle_power_l"]
    for i, L in enumerate(labels):
        print(f"  {L:>20}  range=[{p[i].min():+.3f}, {p[i].max():+.3f}]  W/kg")
