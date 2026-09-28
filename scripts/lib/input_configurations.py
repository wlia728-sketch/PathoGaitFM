"""The seven manuscript input configurations, in the model's 40-channel index space.

Only angle channels are listed. The sampler's fixed DROP_16CH is applied separately.
The regression test checks these against partial_input_masking.CONDITIONS.
"""

INPUT_MASKS = {
    "full": (),
    "no_hip_ang": (0, 1, 2, 3, 4, 5),
    "no_knee_ang": (6, 7),
    "no_ankle_ang": (8, 9),
    "no_pelvis": (34, 35, 36, 37, 38, 39),
    "ladder_no_ankle_pelvis": (8, 9, 34, 35, 36, 37, 38, 39),
    "ladder_hip_only": (8, 9, 34, 35, 36, 37, 38, 39, 6, 7),
}
INPUT_SETS = tuple(INPUT_MASKS)
