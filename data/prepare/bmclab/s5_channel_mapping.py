"""54-channel mapping: BMClab cycle (34, 101) → 4-cohort (101, 54) + mask (54,).

Per-side cycle convention (Q1 confirmed): each BMClab cycle .npy is one side's
gait cycle. R cycles populate R-side channels, L cycles populate L-side
channels, opposite side is masked.

Mapping table (after Wenqi's Q2-Q5 decisions):

  Target ch     Target name           BMClab source                      Side
  ─────────────────────────────────────────────────────────────────────────────
    0  R_Hip_Angle_X       ←  hip_flexion_r       (deg)                  R
    1  R_Hip_Angle_Y       ←  hip_adduction_r     (deg)                  R
    2  R_Hip_Angle_Z       ←  hip_rotation_r      (deg)                  R
    3  L_Hip_Angle_X       ←  hip_flexion_l       (deg)                  L
    4  L_Hip_Angle_Y       ←  hip_adduction_l     (deg)                  L
    5  L_Hip_Angle_Z       ←  hip_rotation_l      (deg)                  L
    6  R_Knee_Angle_X      ←  knee_angle_r        (deg)                  R
    7  R_Knee_Angle_Y      ←  (none)              mask=0
    8  R_Knee_Angle_Z      ←  (none)              mask=0
    9  L_Knee_Angle_X      ←  knee_angle_l        (deg)                  L
   10  L_Knee_Angle_Y      ←  (none)              mask=0
   11  L_Knee_Angle_Z      ←  (none)              mask=0
   12  R_Ankle_Angle_X     ←  ankle_angle_r       (deg)                  R
   13  R_Ankle_Angle_Y     ←  (none)              mask=0
   14  R_Ankle_Angle_Z     ←  subtalar_angle_r    (deg)  (Q2)            R
   15  L_Ankle_Angle_X     ←  ankle_angle_l       (deg)                  L
   16  L_Ankle_Angle_Y     ←  (none)              mask=0
   17  L_Ankle_Angle_Z     ←  subtalar_angle_l    (deg)  (Q2)            L
   18  R_Hip_Moment_X      ←  hip_flexion_r_moment    (N·m/kg)           R, kinetic
   19  R_Hip_Moment_Y      ←  hip_adduction_r_moment  (N·m/kg)           R, kinetic
   20  R_Hip_Moment_Z      ←  (none) (Q3)         mask=0
   21  L_Hip_Moment_X      ←  hip_flexion_l_moment                       L, kinetic
   22  L_Hip_Moment_Y      ←  hip_adduction_l_moment                     L, kinetic
   23  L_Hip_Moment_Z      ←  (none) (Q3)         mask=0
   24  R_Knee_Moment_X     ←  knee_angle_r_moment                        R, kinetic
   25  L_Knee_Moment_X     ←  knee_angle_l_moment                        L, kinetic
   26  R_Ankle_Moment_X    ←  ankle_angle_r_moment                       R, kinetic
   27  L_Ankle_Moment_X    ←  ankle_angle_l_moment                       L, kinetic
   28  R_Hip_power         ←  computed (Q4)                              R, kinetic
   29  L_hip_power         ←  computed                                   L, kinetic
   30  R_Knee_power        ←  computed                                   R, kinetic
   31  L_Knee_power        ←  computed                                   L, kinetic
   32  R_Ankle_power       ←  computed                                   R, kinetic
   33  L_Ankle_power       ←  computed                                   L, kinetic
   34  R_GRF_V_X           ←  r_ground_force_vx (BW)                     R, kinetic
   35  R_GRF_V_Y           ←  r_ground_force_vy (BW)                     R, kinetic
   36  R_GRF_V_Z           ←  r_ground_force_vz (BW)                     R, kinetic
   37  L_GRF_V_X           ←  l_ground_force_vx (BW)                     L, kinetic
   38  L_GRF_V_Y           ←  l_ground_force_vy (BW)                     L, kinetic
   39  L_GRF_V_Z           ←  l_ground_force_vz (BW)                     L, kinetic
   40  R Tibialis Ant      ←  (none) BMClab has no EMG  mask=0
   41  R Gastrocnemius     ←  (none)              mask=0
   42  R Hamstring         ←  (none)              mask=0
   43  R Rectus            ←  (none)              mask=0
   44  L Tibialis Ant      ←  (none)              mask=0
   45  L Gastrocnemius     ←  (none)              mask=0
   46  L Hamstring         ←  (none)              mask=0
   47  L Rectus            ←  (none)              mask=0
   48  R_Pelvic_Lab_X      ←  pelvis_tx           (Q5, replicated)       both
   49  R_Pelvic_Lab_Y      ←  pelvis_ty                                  both
   50  R_Pelvic_Lab_Z      ←  pelvis_tz                                  both
   51  L_Pelvic_Lab_X      ←  pelvis_tx (same value as ch 48)            both
   52  L_Pelvic_Lab_Y      ←  pelvis_ty                                  both
   53  L_Pelvic_Lab_Z      ←  pelvis_tz                                  both

Mask logic (final mask combines side gating + per-cycle kinetics availability):

  - Side gating: if this cycle is R-side, mask all L-side channels (and v/v).
    Pelvic_Lab (48-53) always on.
  - Kinetics availability: if cycle's `grf_presence_in_cycle <= 0.5`, mask
    all moment + power + GRF channels (18-39).
  - Kinetics side carve-out: if subject-state is in `_grf_mask.json` as
    `r_missing` / `l_missing` / `both_missing`, mask the missing side's
    moment/power/GRF regardless (overrides cycle-level).
  - EMG (40-47): always mask=0 for BMClab.
"""
from __future__ import annotations

import numpy as np

# Indices into BMClab cycle .npy (`s4_cycle_extraction.ALL_CHANNELS`):
SRC = {
    "pelvis_tilt":  0, "pelvis_list":  1, "pelvis_rotation": 2,
    "hip_flexion_r":  3, "hip_flexion_l":  4,
    "hip_adduction_r":  5, "hip_adduction_l":  6,
    "hip_rotation_r":  7, "hip_rotation_l":  8,
    "knee_angle_r":  9, "knee_angle_l": 10,
    "ankle_angle_r": 11, "ankle_angle_l": 12,
    "subtalar_angle_r": 13, "subtalar_angle_l": 14,
    "lumbar_extension": 15, "lumbar_bending": 16, "lumbar_rotation": 17,
    "hip_flexion_r_moment": 18, "hip_flexion_l_moment": 19,
    "hip_adduction_r_moment": 20, "hip_adduction_l_moment": 21,
    "knee_angle_r_moment": 22, "knee_angle_l_moment": 23,
    "ankle_angle_r_moment": 24, "ankle_angle_l_moment": 25,
    "subtalar_angle_r_moment": 26, "subtalar_angle_l_moment": 27,
    "r_ground_force_vx": 28, "r_ground_force_vy": 29, "r_ground_force_vz": 30,
    "l_ground_force_vx": 31, "l_ground_force_vy": 32, "l_ground_force_vz": 33,
}

# Channel groups by side (1-based to mask logic)
R_SIDE_CHANNELS_KIN = [0, 1, 2, 6, 7, 8, 12, 13, 14]        # R angles (incl. masked Y/Z)
L_SIDE_CHANNELS_KIN = [3, 4, 5, 9, 10, 11, 15, 16, 17]      # L angles (incl. masked Y/Z)
R_SIDE_CHANNELS_KINETIC = [18, 19, 20, 24, 26, 28, 30, 32, 34, 35, 36]   # R moment + power + GRF
L_SIDE_CHANNELS_KINETIC = [21, 22, 23, 25, 27, 29, 31, 33, 37, 38, 39]   # L moment + power + GRF
EMG_CHANNELS = list(range(40, 48))      # always masked for BMClab
PELVIC_CHANNELS = list(range(48, 54))    # always available
KINETIC_CHANNELS = list(range(18, 40))   # moments + powers + GRF
KINEMATIC_CHANNELS = list(range(0, 18))  # angles
# Channels with NO BMClab source — always masked
ALWAYS_MASKED = [
    # Knee Y/Z (Rajagopal 1-DOF):
    7, 8, 10, 11,
    # Ankle Y (Rajagopal 1-DOF; Ankle_Z is subtalar):
    13, 16,
    # Hip rotation moment (not in BMClab ID extract per Q3):
    20, 23,
    # EMG:
    *EMG_CHANNELS,
]


def build_54ch(cycle_data: np.ndarray,
                cycle_side: str,
                grf_presence: float,
                grf_mask_category: str,
                computed_powers: np.ndarray,
                pelvis_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (101, 54) data + (54,) mask for one cycle.

    Args:
      cycle_data    -- BMClab (34, 101) cycle array.
      cycle_side    -- 'r' or 'l'.
      grf_presence  -- per-cycle GRF presence fraction (0-1).
      grf_mask_category -- subject-state mask category from
                       _grf_mask.json ('pass', 'r_strike_only', etc.).
      computed_powers -- (6, 101) array from s5_joint_power.compute_powers.
      pelvis_xyz    -- (3, 101) array of pelvis_tx/ty/tz time-normalized to
                       this cycle.

    Returns:
      data54: (101, 54) float32 with NaN for masked positions
      mask54: (54,)     bool   True where channel is reliable
    """
    data54 = np.full((101, 54), np.nan, dtype=np.float32)
    mask54 = np.zeros(54, dtype=bool)

    is_r = (cycle_side == "r")

    # ── Angles ───────────────────────────────────────────────────────────
    # R-side angles: ch 0,1,2 = hip flex/add/rot R; ch 6 = knee_r; ch 12,14 = ankle_r, subtalar_r
    # L-side angles: ch 3,4,5; ch 9; ch 15,17
    angle_map_r = {0: "hip_flexion_r", 1: "hip_adduction_r", 2: "hip_rotation_r",
                   6: "knee_angle_r", 12: "ankle_angle_r", 14: "subtalar_angle_r"}
    angle_map_l = {3: "hip_flexion_l", 4: "hip_adduction_l", 5: "hip_rotation_l",
                   9: "knee_angle_l", 15: "ankle_angle_l", 17: "subtalar_angle_l"}
    # Populate the cycle's own side
    src_map = angle_map_r if is_r else angle_map_l
    for tgt_ch, src_name in src_map.items():
        data54[:, tgt_ch] = cycle_data[SRC[src_name], :]
        mask54[tgt_ch] = True
    # Opposite-side angles remain NaN with mask=False (already initialized).

    # ── Kinetics (moment + power + GRF, ch 18-39) ────────────────────────
    # Kinetics map (R-side):  ch 18 hip_flex_R_moment, 19 hip_add_R_moment,
    #                         ch 24 knee_R_moment, ch 26 ankle_R_moment
    #                         ch 28 hip_power_R, ch 30 knee_power_R, ch 32 ankle_power_R
    #                         ch 34/35/36 R_GRF_vx/vy/vz
    kin_map_r = {
        18: ("source",   "hip_flexion_r_moment"),
        19: ("source",   "hip_adduction_r_moment"),
        24: ("source",   "knee_angle_r_moment"),
        26: ("source",   "ankle_angle_r_moment"),
        28: ("computed", 0),  # hip_flex_power_r in computed_powers[0]
        30: ("computed", 2),  # knee_power_r     in computed_powers[2]
        32: ("computed", 4),  # ankle_power_r    in computed_powers[4]
        34: ("source",   "r_ground_force_vx"),
        35: ("source",   "r_ground_force_vy"),
        36: ("source",   "r_ground_force_vz"),
    }
    kin_map_l = {
        21: ("source",   "hip_flexion_l_moment"),
        22: ("source",   "hip_adduction_l_moment"),
        25: ("source",   "knee_angle_l_moment"),
        27: ("source",   "ankle_angle_l_moment"),
        29: ("computed", 1),  # hip_flex_power_l
        31: ("computed", 3),  # knee_power_l
        33: ("computed", 5),  # ankle_power_l
        37: ("source",   "l_ground_force_vx"),
        38: ("source",   "l_ground_force_vy"),
        39: ("source",   "l_ground_force_vz"),
    }
    kin_map = kin_map_r if is_r else kin_map_l

    # Cycle-level kinetics gate
    kinetics_available = (grf_presence > 0.5)
    # Subject-state carve-outs: if subject is r_missing or l_missing or both,
    # the respective side's kinetics are unreliable regardless of cycle presence.
    side_blocked = False
    if grf_mask_category == "both_missing":
        side_blocked = True
    elif grf_mask_category == "r_missing" and is_r:
        side_blocked = True
    elif grf_mask_category == "l_missing" and not is_r:
        side_blocked = True

    if kinetics_available and not side_blocked:
        for tgt_ch, (kind, key) in kin_map.items():
            if kind == "source":
                data54[:, tgt_ch] = cycle_data[SRC[key], :]
            else:  # computed
                data54[:, tgt_ch] = computed_powers[key, :]
            mask54[tgt_ch] = True

    # ── Pelvic_Lab (48-53) — always present, replicated to both R and L slots ────
    # pelvis_xyz is (3, 101): row 0 = tx, row 1 = ty, row 2 = tz
    data54[:, 48] = pelvis_xyz[0, :]   # R_Pelvic_Lab_X
    data54[:, 49] = pelvis_xyz[1, :]   # R_Pelvic_Lab_Y
    data54[:, 50] = pelvis_xyz[2, :]   # R_Pelvic_Lab_Z
    data54[:, 51] = pelvis_xyz[0, :]   # L_Pelvic_Lab_X (replicated)
    data54[:, 52] = pelvis_xyz[1, :]
    data54[:, 53] = pelvis_xyz[2, :]
    mask54[48:54] = True

    # ── EMG (40-47): always masked off ──────────────────────────────────
    # (already False; data left as NaN)

    # ── Knee Y/Z, Ankle Y, hip_rotation moments: always masked off ────
    # (already False; data left as NaN — Step 4 produced NaN for absent sources anyway)
    return data54, mask54
