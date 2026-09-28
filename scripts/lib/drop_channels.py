"""Input-channel regime shared by the force-plate-free evaluators.

DROP_16CH lists the 40-channel positions token-filled at input so the model runs on
the kinematic input regime the estimators report: the EMG channels (26-33) and the
non-sagittal hip-moment channels (11, 12, 14, 15). Every frozen-estimation evaluator
imports this constant to reproduce that input regime.
"""
DROP_16CH = [26, 27, 28, 29, 30, 31, 32, 33, 11, 12, 14, 15]
