# Third-party notices

This package builds on the work below. The code in this repository is MIT-licensed (see `LICENSE`);
nothing here relicenses any of the following.

## Reimplemented architectures

The three comparators in Figure 2 are our own reimplementations, written against the published
descriptions on this work's own inputs and targets. Each deviates from its source in ways recorded in the
class docstring of `model_baseline/published_comparators.py`. No source code from any of them was copied.

**GroundLinkNet** — Han F, Zhang Y, Ceylan D, et al. *GroundLink: a dataset unifying human body movement
and ground reaction dynamics.* SIGGRAPH Asia 2023 Conference Papers, article 55.
doi:10.1145/3610548.3618247

While writing `CondGroundLinkNet` we consulted the **UnderPressure** reference implementation, which the
GroundLink authors state their network inherits from. UnderPressure is distributed by InterDigital under a
research-only Limited Software Evaluation License Agreement. As that agreement requires:

> UnderPressure is a product of InterDigital.

and its publication is cited: Mourot L, Hoyet L, Le Clerc F, Hellier P. *UnderPressure: deep learning for
foot contact detection, ground reaction force estimation and footskate cleanup.* Computer Graphics Forum
(SCA) 2022;41(8). doi:10.1111/cgf.14635

No UnderPressure source was copied into this package, and the shared hyperparameters are those the
GroundLink paper publishes.

**Sugai LSTM** — Sugai K, et al. *LSTM network-based estimation of ground reaction forces during walking in
stroke patients using markerless motion capture system.* IEEE Transactions on Medical Robotics and Bionics
2023;5(4):831-841.

**Ozates CNN** — Ozates ME, Karabulut D, Salami F, Wolf SI, Arslan YZ. *Machine-learning-based prediction of
joint moments based on kinematics in patients with cerebral palsy.* Journal of Biomechanics 2023; and
Ozates ME, Salami F, Wolf SI, Arslan YZ. *Estimating ground reaction forces from gait kinematics in
cerebral palsy: a convolutional neural network approach.* Annals of Biomedical Engineering
2025;53(3):634-. The dense widths are scaled, as the class docstring records.

## Scientific software

**OpenSim** — Delp SL, Anderson FC, Arnold AS, et al. *OpenSim: open-source software to create and analyze
dynamic simulations of movement.* IEEE Trans Biomed Eng 2007;54(11):1940-1950. And Seth A, Hicks JL,
Uchida TK, et al. *OpenSim: simulating musculoskeletal dynamics and neuromuscular control.* PLoS Comput
Biol 2018;14(7):e1006223. Apache 2.0. Used for inverse kinematics and inverse dynamics at the external
sites.

**Rajagopal full-body model** — Rajagopal A, Dembia CL, DeMers MS, et al. *Full-body musculoskeletal model
for muscle-driven simulation of human gait.* IEEE Trans Biomed Eng 2016;63(10):2068-2079. Scaled per
subject for the Leuven inverse kinematics.

**Madgwick orientation filter** — Madgwick SOH, Harrison AJL, Vaidyanathan R. *Estimation of IMU and MARG
orientation using a gradient descent algorithm.* IEEE ICORR 2011. Reached through the `ahrs` package (MIT)
for the wearable-inertial branch.

**nimblephysics / AddBiomechanics** — Werling K, et al. *AddBiomechanics: automating model scaling, inverse
kinematics, and inverse dynamics from human motion data through optimization.* PLOS ONE
2023;18(11):e0295152. Apache 2.0.

**ezc3d** — Michaud B, Begon M. *ezc3d: an easy C3D file I/O cross-platform solution for C++, Python and
MATLAB.* Journal of Open Source Software 2021;6(58):2911. MIT.

**PyTorch** — Paszke A, et al. *PyTorch: an imperative style, high-performance deep learning library.*
NeurIPS 2019. BSD-3-Clause.

**NumPy, SciPy, scikit-learn, Matplotlib, openpyxl** — retain their respective package licences and notices.

## Datasets

Every dataset used in this work is listed with its citation and link in `data/README.md`,
and each carries its own terms. The two SimTK cerebral-palsy projects are additionally documented in
`data/external/leuven_cp/PROVENANCE.md`.
