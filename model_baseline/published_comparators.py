"""Published state-of-the-art comparators for Table 1, on the IDENTICAL fair pipeline as
baseline_nondiffusion.py: same CV5 split, valid-cycle set, cell-level target validity mask INSIDE the loss
(including the bmclab_pd moment exclusion the diffusion trainer applies), 16 kinematic inputs, the SAME
source/cohort/severity conditioning the diffusion model receives, three-seed prediction ensembling, and
subject-macro Method-B scoring.

BASELINE SELECTION follows the criterion GaitDynamics states for its own comparison: compare against
published models "specifically designed to use joint kinematics for force estimation", choosing
"representative architectures distinct from" the proposed one, trained on the same data, optimizer, batch
size and learning rate. GaitDynamics names two, and both are reimplemented here:

  CondGroundLinkNet   Han et al., SIGGRAPH Asia 2023 (GaitDynamics ref 21), convolutional
  CondSugaiLSTM       Sugai et al., IEEE TMRB 2023 (GaitDynamics ref 22), recurrent, published on GRF
                      estimation in POST-STROKE walking and therefore the closest published state of the
                      art to this paper's own cohorts
  CondOzatesCNN       Ozates et al., one-dimensional convolutional, published on joint-kinetics prediction
                      from kinematics in a large single-site CEREBRAL-PALSY cohort, the closest published
                      state of the art to this paper's own primary cohort

Generic architectures are excluded: they are not published methods for this task, so under this
selection criterion they do not belong in the comparison. Each
reimplementation's deviations from its source are documented in its own class docstring. Params and the
diffusion reference are written by the producer (no hardcoded numbers).

ONE MATCHED PROTOCOL per architecture, as reported in the manuscript.
  matched     The 16 per-cycle input-validity flags are concatenated to the 16 kinematic
              channels, so the comparator is told which of its inputs are absent, which is the information
              the diffusion model gets from its learned mask token and explicit unknown flag. Without this
              the comparison hands one model a signal the other cannot see, on the 5 of 16 input channels
              that are absent on every post-stroke and Parkinson cycle.

Comparators are reported for the internal cross-validation only. They are not run at the external sites,
because the Leuven inverse-kinematics solution expresses the pelvis in the global laboratory frame and five
of its twenty trials exceed 5 cm mean marker error, so an external comparison would index the input
quality at that site rather than the architectures.
Writes outputs/evaluation/published_comparators.json by default.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # model_baseline -> the package root
for _p in [ROOT, ROOT / "model", ROOT / "model_baseline"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from guarded_write import guarded_dump
from evaluation_contract import safe_result_path
import torch
import torch.nn as nn
# reuse the EXACT fair pipeline (masked loss + conditioning + CV5 + subject-macro)
from baseline_nondiffusion import (
    run, build_transform, diffusion_reference, CondEmbed, SEV,
    NIN, NT, NOUT, DCOND, DEVICE,
)




class CondSugaiLSTM(nn.Module):
    """The recurrent comparator GaitDynamics benchmarks against (its reference 22, Sugai et al., IEEE Trans
    Med Robot Bionics 2023), reimplemented here on our inputs and targets. That work estimates ground
    reaction force during walking in STROKE patients from markerless motion capture, so it is the published
    state of the art closest to this paper's own cohorts.

    Architecture exactly as the paper's Table IV and Figure 3, the configuration its Optuna search selected
    over 70 iterations: two UNIDIRECTIONAL LSTM hidden layers of 216 and 76 units, with dropout 0.20 after
    the first and 0.25 after the second, then a linear read-out.

    THREE DOCUMENTED DEVIATIONS, all forced by our pipeline rather than chosen:
      1. The original slides a 27-frame window over a continuous recording. Our pipeline scores whole
         gait-cycle sequences of 100 points, so the network is run over the full cycle. The layer stack and
         its causal, unidirectional receptive field are unchanged; only the sequence the loss sees differs.
      2. Source, cohort and severity conditioning is concatenated to the input at every timestep, as for
         every other baseline here, so all comparators see the conditioning the diffusion model sees.
      3. Inputs are our sixteen kinematic channels and outputs our eight kinetic channels, instead of the
         original 52 Kinect joint-position features and six ground-reaction-force components.
    """
    def __init__(self, cin=NIN, cout=NOUT, hid1=216, hid2=76, p1=0.20, p2=0.25, dcond=DCOND):
        super().__init__()
        self.cond = CondEmbed(dcond)
        self.l1 = nn.LSTM(cin + dcond, hid1, num_layers=1, batch_first=True)   # unidirectional, as published
        self.d1 = nn.Dropout(p=p1)
        self.l2 = nn.LSTM(hid1, hid2, num_layers=1, batch_first=True)
        self.d2 = nn.Dropout(p=p2)
        self.head = nn.Linear(hid2, cout)

    def forward(self, x, cond):                                       # x (B,NIN,NT)
        h = x.transpose(1, 2)                                         # (B,NT,NIN)
        c = self.cond(cond)[:, None, :].expand(-1, h.shape[1], -1)
        h, _ = self.l1(torch.cat([h, c], 2)); h = self.d1(h)
        h, _ = self.l2(h); h = self.d2(h)
        return self.head(h).transpose(1, 2)                           # (B,NOUT,NT)


class CondGroundLinkNet(nn.Module):
    """GroundLinkNet, the convolutional comparator GaitDynamics benchmarks against (its reference 21,
    Han et al., SIGGRAPH Asia 2023), reimplemented here on our inputs and targets.

    Architecture as published, cross-checked against the UnderPressure implementation the authors state
    GroundLinkNet inherits from: four 1D temporal convolutions with 7-frame kernels and replicate padding,
    channel progression 128, 128, 256, 256, each followed by ELU, then three fully connected layers applied
    independently at each frame (two 256-wide with ELU and dropout 0.2, then a bias-free read-out).

    THREE DOCUMENTED DEVIATIONS, all forced by the target space rather than chosen:
      1. No Softplus read-out. The published network ends in Softplus because it predicts ground reaction
         force, which is non-negative in physical units. Our targets live in the frozen z-scored space,
         where even vertical GRF is signed (physical zero maps to about -0.34), and half of them are joint
         moments, which are signed in any units. A non-negativity constraint is therefore not applicable.
      2. Source, cohort and severity conditioning is concatenated to the input channels, as for every other
         baseline here, so that all comparators receive the same conditioning the diffusion model receives.
      3. Inputs are our sixteen kinematic channels and outputs our eight kinetic channels, instead of the
         original joint angles plus pelvis trajectory to force and centre of pressure.
    """
    def __init__(self, cin=NIN, cout=NOUT, k=7, widths=(128, 128, 256, 256),
                 fc_depth=3, fc_dropout=0.2, dcond=DCOND):
        super().__init__()
        self.cond = CondEmbed(dcond)
        layers = []; d = cin + dcond
        for w in widths:
            layers += [nn.Conv1d(d, w, k, padding=k // 2, padding_mode="replicate"), nn.ELU()]
            d = w
        self.body = nn.Sequential(*layers)
        fc = []
        for _ in range(max(0, fc_depth - 1)):
            fc += [nn.Conv1d(d, d, 1), nn.ELU(), nn.Dropout(p=fc_dropout)]   # kernel 1 == per-frame Linear
        fc += [nn.Conv1d(d, cout, 1, bias=False)]
        self.head = nn.Sequential(*fc)

    def forward(self, x, cond):
        c = self.cond(cond)[:, :, None].expand(-1, -1, x.shape[-1])
        return self.head(self.body(torch.cat([x, c], 1)))


OZ_CONV_W = (128, 128, 512, 1024, 2048)          # published filter counts
OZ_CONV_K = (30, 15, 10, 5, 3)                   # published kernel sizes
OZ_FC = (10000, 8000, 6000, 4000, 3000, 2000, 1000, 500, 250, 100)   # published dense widths
OZ_SCALE, OZ_FLOOR = 1 / 16, 128


class CondOzatesCNN(nn.Module):
    """The one-dimensional convolutional network of Ozates et al., reimplemented on our inputs and targets.

    That work predicts lower-limb joint kinetics from kinematics in a large single-site cerebral-palsy
    cohort, which makes it the published architecture closest to this paper's own primary cohort. Both of
    its papers specify the identical network: five convolutions with 128, 128, 512, 1024 and 2048 filters
    and kernels of 30, 15, 10, 5 and 3, ReLU throughout, then ten fully connected layers of 10000, 8000,
    6000, 4000, 3000, 2000, 1000, 500, 250 and 100 units over the flattened convolutional output.

    THREE DOCUMENTED DEVIATIONS.
      1. WIDTH. The published dense widths are defined against that work's own flattened representation.
         Against ours, which is 2048 channels over a 100-point cycle and so 204,800 wide, the first dense
         layer alone would carry 2.05 billion weights and the whole network 2.23 billion, sixty-seven times
         the diffusion model it is meant to bound. The ten widths are therefore scaled by one sixteenth.
         They are additionally floored at 128 units, because the published funnel narrows to 100 units
         before its read-out and an unfloored one-sixteenth scaling would narrow to 6, which cannot carry
         the 800 values of our eight-channel hundred-point target. The convolutional stack, the layer
         count and the funnel shape are unchanged.
      2. Source, cohort and severity conditioning is concatenated to the input channels, as for every other
         comparator here.
      3. Inputs are our sixteen kinematic channels and outputs our eight kinetic channels over the cycle.

    Optimiser, batch size, learning rate and epoch budget are this comparison's shared settings rather than
    the paper's own 500 epochs at batch 32, which is what holding the training protocol fixed across
    architectures requires.
    """
    def __init__(self, cin=NIN, cout=NOUT, nt=NT, dcond=DCOND,
                 conv_w=OZ_CONV_W, conv_k=OZ_CONV_K, fc=OZ_FC, scale=OZ_SCALE, floor=OZ_FLOOR):
        super().__init__()
        self.cond = CondEmbed(dcond)
        self.cout, self.nt = cout, nt
        layers = []; d = cin + dcond
        for w, k in zip(conv_w, conv_k):
            layers += [nn.Conv1d(d, w, k, padding=k // 2), nn.ReLU()]
            d = w
        self.body = nn.Sequential(*layers)
        self.widths = [max(int(round(w * scale)), floor) for w in fc]
        dense = []; prev = d * nt
        for w in self.widths:
            dense += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        dense += [nn.Linear(prev, cout * nt)]
        self.head = nn.Sequential(*dense)

    def forward(self, x, cond):
        c = self.cond(cond)[:, :, None].expand(-1, -1, x.shape[-1])
        h = self.body(torch.cat([x, c], 1))
        h = h[..., :self.nt].flatten(1)          # even kernels shift length by one, so clip to the cycle
        return self.head(h).view(-1, self.cout, self.nt)


ARCHS = [
    ("groundlink_cnn", lambda cin: CondGroundLinkNet(cin=cin)),
    ("sugai_lstm", lambda cin: CondSugaiLSTM(cin=cin)),
    ("ozates_cnn", lambda cin: CondOzatesCNN(cin=cin)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation/published_comparators.json")
    parser.add_argument("--diffusion", type=Path, default=ROOT / "outputs/evaluation/kinetics_cv5.json",
                        help="eval_kinetics_all_cycles.py result, used for the reference headline when present")
    args = parser.parse_args()
    try:
        args.out = safe_result_path(args.out, ROOT)
    except ValueError as error:
        parser.error(str(error))
    print(f"Table 1 comparators (masked loss + source/cohort/severity cond) on {DEVICE}", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    transform = build_transform()
    with open(SEV, encoding="utf-8") as handle:
        sev = json.load(handle)
    dst = args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    cin = NIN * 2
    for name, builder in ARCHS:
        key = f"{name}__matched"
        res = run(lambda b=builder, c=cin: b(c), True, transform, sev, in_mask=True)
        res["config"] = {"arch": name, "arm": "matched", "epochs": 80, "masked_loss": True,
                         "conditioned": True, "n_seed_ensemble": 3, "pd_moment_loss_excluded": True,
                         "input_validity_channels": True, "n_input_channels": cin,
                         "inputs": "16 sagittal kinematics + 16 validity flags",
                         "targets": "8 (vGRF + sagittal moments)"}
        out[key] = res
        guarded_dump(out, dst, "published_comparators.py")
        print(f"  {key:28} PCC {res['subject_macro_PCC']}  ({res['n_params']} params, {res['elapsed_s']}s)", flush=True)
    out["reference"] = {"diffusion_CV5": diffusion_reference(args.diffusion),
                        "source": "eval_kinetics_all_cycles.py result, None when not produced"}
    out["arms"] = {
        "matched": ("information-matched, MAIN Table 1. The 16 per-cycle input-validity flags are "
                    "concatenated to the 16 kinematic channels, so the comparator is told which inputs "
                    "are absent, as the diffusion model is by its learned mask token."),
    }
    saved_path = guarded_dump(out, dst, "published_comparators.py")
    print("\nsaved", saved_path, flush=True)
    print(f"  {'model':22} {'matched':>10}")
    for name, _ in ARCHS:
        a = out[f"{name}__matched"]["subject_macro_PCC"]
        print(f"  {name:22} {a:>10}", flush=True)
    print(f"  {'diffusion (this work)':22} {out['reference']['diffusion_CV5']:>10}", flush=True)


if __name__ == "__main__":
    main()
