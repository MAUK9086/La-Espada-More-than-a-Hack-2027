"""GPU experiments on the dilated TCN of E57: seed noise, seed ensembles, decoder weight, configurations.

File structure
--------------
- CONFIGS: named network / training configurations. "base" is E57's boundary TCN unchanged (tcn.py:
  two stages, 64 / 32 channels, receptive field 255 frames, Adam 5e-4, 20 epochs, 1500-frame windows,
  batch 8). A GPU-trained network is a new experiment, never a reproduction of the CPU result.
- build_model(): the two-stage dilated TCN, sized by a configuration.
- load_seqs(): the cached 75-column sequences of tcn.py (cache/tcn/<profile>/<match>.npz).
- fit(): one model on the r1 sequences of the given matches, on CUDA, deterministic for a seed.
- predict(): team-symmetric H/A/D probabilities of one match (kinematic inputs).
- lomo(): for every seed and held-out match: fit on the other 7, predict the held-out one; saved to
  cache/tcn_gpu/<config>/s<seed>/probs_<match>.npy (resumable).
- final(): one model per seed on all 8 calibration matches, cache/tcn_gpu/<config>/s<seed>/model_final.pt.
- ensemble(): arithmetic mean of the seeds' probabilities.
- Decoding and statistics: decode_scores() (run.decode_evidence through review_ablation.decode_variant,
  as tcn.score() does), sign_flip_p(), paired_bootstrap() (possessions resampled within matches, the
  organisers' bootstrap_spell_f1 applied to a difference of 8-match means).
- report_e59(): the declared tables of E59 -> reports/tcn_gpu_seeds.md.
- load_realisation(), seq_aug(): further degradation realisations of the calibration matches (other
  seeds for the ball noise, player dropout and ball dropout; optionally mirrored across the pitch's long
  axis), written as sequences to cache/tcn_aug/<profile>_a<r>/<match>.npz for training only.

Declared before running. Baseline p3-refined 0.7077 / 0.6889, which must reproduce
inside the run; bar 0.7177 / 0.6939. No held-out match and no BAR-ATH data are touched by this file.

    .venv/Scripts/python solution/external/tcn_gpu.py lomo base 0 1 2 3 4
    .venv/Scripts/python solution/external/tcn_gpu.py e59
"""

import itertools
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common  # noqa: E402

common.FLAGS_ENABLED = False

import player_tokens  # noqa: E402
import possession  # noqa: E402
import possession_metrics as pm  # noqa: E402
from common import A, CACHE_DIR, D, H, REPORT_DIR, X  # noqa: E402
from external import diagnose_errors, lomo as lomo_mod, min_duration, review_ablation  # noqa: E402

# Params
SEQ_DIR = os.path.join(CACHE_DIR, "tcn")          # the sequences written by tcn.py seq
OUT_DIR = os.path.join(CACHE_DIR, "tcn_gpu")
AUG_DIR = os.path.join(CACHE_DIR, "tcn_aug")     # further degradation realisations (seq_aug)
P1_NAMES = ("p1_h", "p1_a", "p1_d")
SWAP_LABEL = np.array([A, H, D, X], dtype=np.int64)
FLOOR = 1e-3
BASELINE = (0.7077, 0.6889)                        # p3-refined, LOMO r1 -> k
BAR = (0.7177, 0.6939)
R1 = ("steady", 50, 10.0)                          # the R1 decoder of E55 / candidate A
CATS = ("short", "split_dead", "split_team", "merged")
BOOT_N, BOOT_SEED = 2000, 0
CONFIGS = {
    "base": dict(hidden=64, dilations=(1, 2, 4, 8, 16, 32, 64), hidden2=32, dilations2=(1, 2, 4, 8, 16, 32),
                 dropout=0.2, lam_smooth=0.15, smooth_clamp=16.0, lam_boundary=0.1, boundary_radius=2,
                 lr=5e-4, epochs=20, window=1500, batch=8, stages=2, weight_decay=0.0),
}
# E60: the same network, trained on several degradation realisations per match.
CONFIGS["aug4"] = dict(CONFIGS["base"], sources=("r1", "r1_a1", "r1_a2", "r1_a3"), epochs=5)
CONFIGS["aug4_e10"] = dict(CONFIGS["base"], sources=("r1", "r1_a1", "r1_a2", "r1_a3"), epochs=10)
CONFIGS["aug4k"] = dict(CONFIGS["base"], sources=("r1", "r1_a1", "r1_a2", "k_a4"), epochs=5)
# E61: training profile, and an auxiliary touch-level target.
CONFIGS["k4"] = dict(CONFIGS["base"], sources=("k_a4", "k_a5", "k_a6", "k_a7"), epochs=5)
CONFIGS["mix8"] = dict(CONFIGS["base"], sources=("r1", "r1_a1", "r1_a2", "r1_a3", "k_a4", "k_a5", "k_a6", "k_a7"),
                       epochs=5)
CONFIGS["aug4_touch0.3"] = dict(CONFIGS["aug4"], lam_touch=0.3)
CONFIGS["aug4_touch1"] = dict(CONFIGS["aug4"], lam_touch=1.0)
# E62: control for E60 (longer training without realisations) and scaling.
R1_SOURCES4 = ("r1", "r1_a1", "r1_a2", "r1_a3")
R1_SOURCES8 = R1_SOURCES4 + ("r1_a8", "r1_a9", "r1_a10", "r1_a11")
CONFIGS["base_e40"] = dict(CONFIGS["base"], epochs=40)
CONFIGS["aug4_e20"] = dict(CONFIGS["base"], sources=R1_SOURCES4, epochs=20)
CONFIGS["aug8_e10"] = dict(CONFIGS["base"], sources=R1_SOURCES8, epochs=10)
# E64: the 20 SkillCorner matches as extra training sequences in every fold.
CONFIGS["aug4_e10_skc"] = dict(CONFIGS["base"], sources=("r1", "r1_a1", "r1_a2", "r1_a3"), epochs=10, extra="skc")
CONFIGS["k4_e10"] = dict(CONFIGS["base"], sources=("k_a4", "k_a5", "k_a6", "k_a7"), epochs=10)
# E65: a per-player encoder in front of the TCN ("set": embedding width).
CONFIGS["set_k4_e10"] = dict(CONFIGS["k4_e10"], set=32)
CONFIGS["set_aug4_e10"] = dict(CONFIGS["base"], sources=("r1", "r1_a1", "r1_a2", "r1_a3"), epochs=10, set=32)
# E66: 8 realisations of profile k, and 8 + 8 of both profiles.
K_SOURCES8 = tuple("k_a{}".format(i) for i in range(4, 12))
CONFIGS["k8_e10"] = dict(CONFIGS["base"], sources=K_SOURCES8, epochs=10)
CONFIGS["mix16"] = dict(CONFIGS["base"], sources=R1_SOURCES8 + K_SOURCES8, epochs=5)
# E67: the k recipe trained on 1.5 s labels.
CONFIGS["k4_e10_c1.5"] = dict(CONFIGS["k4_e10"], control_s=1.5)
# E69: ceiling test -- trained and tested on tracking without (part of) the degradation.
CONFIGS["ceil_clean"] = dict(CONFIGS["base"], sources=("clean_a0", "clean_a1"), epochs=10, test="clean_a0")
CONFIGS["ceil_nodrop"] = dict(CONFIGS["base"], sources=("nodrop_a0", "nodrop_a1", "nodrop_a2", "nodrop_a3"), epochs=10,
                              test="nodrop_a0")
# E70: architecture and training screen on the k4_e10 recipe; every variant also
# stores its prediction on the mirrored test match ("tta").
_K4 = dict(CONFIGS["k4_e10"], tta=True)
SCREEN = {
    "x_ref": {},                                                       # k4_e10 again (determinism check, mirror)
    "x_w128": dict(hidden=128, hidden2=64),
    "x_deep": dict(dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256), dilations2=(1, 2, 4, 8, 16, 32, 64)),
    "x_ema": dict(ema=0.999),
    "x_cos": dict(opt="adamw_cos", lr=1e-3, weight_decay=1e-2),
    "x_st3": dict(stages=3),
    "x_gru": dict(gru=64),
    "x_ls": dict(label_smooth=0.1),
    "x_drop1": dict(dropout=0.1),
    "x_drop35": dict(dropout=0.35),
    "x_bnd3": dict(lam_boundary=0.3),
    "x_nosm": dict(lam_smooth=0.0),
}
for _name, _change in SCREEN.items():
    CONFIGS[_name] = dict(_K4, **_change)
# E71: the two training changes of E70 that were positive, together.
CONFIGS["y_cos_ema"] = dict(_K4, opt="adamw_cos", lr=1e-3, weight_decay=1e-2, ema=0.999)
# E63: the control rule of the training labels (seconds; 2.0 is aug4_e10).
CONFIGS["ctrl1.0"] = dict(CONFIGS["aug4_e10"], control_s=1.0)
CONFIGS["ctrl1.5"] = dict(CONFIGS["aug4_e10"], control_s=1.5)
CONFIGS["ctrl3.0"] = dict(CONFIGS["aug4_e10"], control_s=3.0)


def _torch():
    """torch with deterministic CUDA settings."""
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch


def build_model(n_in, p1_index, cfg):
    """The dilated TCN of tcn.py, sized by cfg; cfg["stages"] refinement stages in total (>= 2).

    Args:
        n_in: input columns.
        p1_index: positions of the three P1 columns (fed again to every refinement stage).
        cfg: a CONFIGS entry.

    Returns:
        torch.nn.Module: forward(x (B, F, T)) -> (list of per-stage logits (B, 3, T), boundary logits (B, T),
        touch-level logits (B, 3, T) or None). The touch head (cfg["lam_touch"] > 0) is an auxiliary
        training target on the stage-1 trunk and is not used at inference.
    """
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class Residual(nn.Module):
        """Dilated conv -> ReLU -> 1x1 conv -> dropout -> residual add."""

        def __init__(self, channels, dilation):
            super().__init__()
            self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.point = nn.Conv1d(channels, channels, 1)
            self.drop = nn.Dropout(cfg["dropout"])

        def forward(self, x):
            return x + self.drop(self.point(F.relu(self.dilated(x))))

    class Stage(nn.Module):
        """1x1 input conv, residual dilated blocks, 1x1 head to the 3 classes."""

        def __init__(self, n_inputs, channels, dilations):
            super().__init__()
            self.inp = nn.Conv1d(n_inputs, channels, 1)
            self.blocks = nn.ModuleList([Residual(channels, d) for d in dilations])
            self.out = nn.Conv1d(channels, 3, 1)

        def forward(self, x):
            h = self.inp(x)
            for block in self.blocks:
                h = block(h)
            return h, self.out(h)

    embed = cfg.get("set")
    n_set = player_tokens.N_COLS if embed else 0
    n_base = n_in - n_set
    per_team, n_ch = player_tokens.PER_TEAM, len(player_tokens.CHANNELS)

    class SetEncoder(nn.Module):
        """Every player as a token: shared layers over the 22 players, one exchange of team context
        (own-team and other-team mean), then max and mean pooling per team -> (B, 4 * embed, T)."""

        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_ch + 4 + len(player_tokens.BALL_CHANNELS) + 1, embed, 1)
            self.time1 = nn.Conv1d(embed, embed, 5, padding=2)
            self.mix = nn.Conv1d(3 * embed, embed, 1)
            self.time2 = nn.Conv1d(embed, embed, 5, padding=4, dilation=2)
            self.drop = nn.Dropout(cfg["dropout"])

        def forward(self, x):
            b, _, t = x.shape
            tok = x[:, :2 * per_team * n_ch].reshape(b, 2 * per_team, n_ch, t)
            ball = x[:, 2 * per_team * n_ch:]                                   # (B, 5, T): x, y, on, vx, vy
            on = tok[:, :, 4:5]                                                 # (B, 22, 1, T)
            seen = ball[:, None, 2:3]
            dx = (tok[:, :, 0:1] - ball[:, None, 0:1]) * (player_tokens.X_SCALE / 10.0) * seen * on
            dy = (tok[:, :, 1:2] - ball[:, None, 1:2]) * (player_tokens.Y_SCALE / 10.0) * seen * on
            dist = torch.sqrt(dx * dx + dy * dy + 1e-6) * seen * on             # units of 10 m
            near = torch.exp(-dist * (10.0 / 3.0)) * seen * on
            flag = torch.ones(b, 2 * per_team, 1, t, device=x.device)
            flag[:, per_team:] = -1.0
            feat = torch.cat((tok, dx, dy, dist, near, ball[:, None].expand(b, 2 * per_team, -1, t), flag * on), dim=2)
            h = F.relu(self.inp(feat.reshape(b * 2 * per_team, -1, t)))
            h = F.relu(self.time1(h)).reshape(b, 2 * per_team, embed, t) * on
            count = on.reshape(b, 2, per_team, 1, t).sum(2).clamp(min=1.0)       # (B, 2, 1, T)
            mean = h.reshape(b, 2, per_team, embed, t).sum(2) / count            # (B, 2, E, T)
            own = mean[:, :, None].expand(b, 2, per_team, embed, t).reshape(b, 2 * per_team, embed, t)
            other = mean.flip(1)[:, :, None].expand(b, 2, per_team, embed, t).reshape(b, 2 * per_team, embed, t)
            h = F.relu(self.mix(torch.cat((h, own, other), dim=2).reshape(b * 2 * per_team, 3 * embed, t)))
            h = F.relu(self.time2(h)).reshape(b, 2, per_team, embed, t) * on.reshape(b, 2, per_team, 1, t)
            pooled = torch.cat((h.amax(2), h.sum(2) / count), dim=2)             # (B, 2, 2E, T)
            return self.drop(pooled.reshape(b, 4 * embed, t))

    class DilatedTCN(nn.Module):
        """Stage 1 on the inputs; each later stage on softmax(previous) + the P1 columns."""

        def __init__(self):
            super().__init__()
            self.players = SetEncoder() if embed else None
            self.stage1 = Stage(n_base + (4 * embed if embed else 0), cfg["hidden"], cfg["dilations"])
            self.refine = nn.ModuleList([Stage(3 + len(p1_index), cfg["hidden2"], cfg["dilations2"])
                                         for _ in range(cfg["stages"] - 1)])
            self.boundary = nn.Conv1d(cfg["hidden"], 1, 1)
            if cfg.get("gru"):
                self.gru = nn.GRU(cfg["hidden"], cfg["gru"], batch_first=True, bidirectional=True)
                self.gru_out = nn.Conv1d(2 * cfg["gru"], 3, 1)
            else:
                self.gru = None
            self.touch = nn.Conv1d(cfg["hidden"], 3, 1) if cfg.get("lam_touch") else None
            self.register_buffer("p1_index", torch.tensor(p1_index, dtype=torch.long))

        def forward(self, x):
            if self.players is not None:
                trunk, logits = self.stage1(torch.cat((x[:, :n_base], self.players(x[:, n_base:])), dim=1))
            else:
                trunk, logits = self.stage1(x)
            if self.gru is not None:
                # in pieces of the training window: the GRU never saw a longer sequence, and cuDNN
                # refuses a whole half (about 70,000 frames) in one call
                step = cfg["window"]
                pieces = [self.gru(trunk[:, :, i:i + step].transpose(1, 2).contiguous())[0]
                          for i in range(0, trunk.shape[2], step)]
                logits = logits + self.gru_out(torch.cat(pieces, dim=1).transpose(1, 2))
            outs = [logits]
            p1 = x.index_select(1, self.p1_index)
            for stage in self.refine:
                _, logits = stage(torch.cat((F.softmax(logits, dim=1), p1), dim=1))
                outs.append(logits)
            touch = self.touch(trunk) if self.touch is not None else None
            return outs, self.boundary(trunk).squeeze(1), touch

    return DilatedTCN()


def load_seqs(profile, names=None, seq_dir=None):
    """{match: dict(x_home, x_swap, labels, blocks, names)} from the tcn.py sequence cache."""
    names = names or [n for n, _ in lomo_mod.calibration_matches()]
    out = {}
    for name in names:
        with np.load(os.path.join(seq_dir or SEQ_DIR, profile, name + ".npz")) as z:
            out[name] = {k: z[k] for k in z.files}
    return out


def boundary_target(labels, radius):
    """1 within +/-radius frames of a change between two non-X labels, else 0."""
    change = np.zeros(len(labels), bool)
    change[1:] = (labels[1:] != labels[:-1]) & (labels[1:] != X) & (labels[:-1] != X)
    out = np.zeros(len(labels), np.float32)
    for t in np.flatnonzero(change):
        out[max(t - radius, 0):t + radius + 1] = 1.0
    return out


def _scale(x, mean, std):
    return np.nan_to_num((x - mean) / std, nan=0.0).astype(np.float32)


def _windows(x, labels, bnd, blocks, window, touch=None):
    """Tile each block into fixed windows; padding has label -100 and mask 0 (as tcn._windows).

    Returns xs, ys, bs, ms and ts (the touch-level labels, all -100 when touch is None).
    """
    xs, ys, bs, ms, ts = [], [], [], [], []
    for _, first, last in blocks:
        for start in range(first, last + 1, window):
            stop = min(start + window, last + 1)
            m = stop - start
            xw = np.zeros((x.shape[1], window), np.float32)
            xw[:, :m] = x[start:stop].T
            yw = np.full(window, -100, np.int64)
            yw[:m] = np.where(labels[start:stop] == X, -100, labels[start:stop])
            bw = np.zeros(window, np.float32)
            bw[:m] = bnd[start:stop]
            mw = np.zeros(window, np.float32)
            mw[:m] = 1.0
            tw = np.full(window, -100, np.int64)
            if touch is not None:
                tw[:m] = np.where(touch[start:stop] == X, -100, touch[start:stop])
            xs.append(xw), ys.append(yw), bs.append(bw), ms.append(mw), ts.append(tw)
    return xs, ys, bs, ms, ts


def _loss(torch, outs, bnd, y, b, mask, pos_weight, cfg, touch_logits=None, t=None):
    """CE on every stage + lam_smooth * T-MSE on every stage + lam_boundary * BCE (as tcn.loss_fn).

    The cross-entropy is written out (gather) because nll_loss on (B, C, T) has no deterministic CUDA path.
    """
    F = torch.nn.functional
    valid = (y != -100).float()
    target = y.clamp(min=0).unsqueeze(1)
    pair = mask[:, 1:] * mask[:, :-1]
    total = 0.0
    for logits in outs:
        logp = F.log_softmax(logits, dim=1)
        picked = logp.gather(1, target).squeeze(1)
        eps = cfg.get("label_smooth", 0.0)
        if eps:
            picked = (1.0 - eps) * picked + eps * logp.mean(1)
        total = total - (picked * valid).sum() / valid.sum().clamp(min=1.0)
        diff = torch.clamp((logp[:, :, 1:] - logp[:, :, :-1].detach()) ** 2, max=cfg["smooth_clamp"]).mean(1)
        total = total + cfg["lam_smooth"] * (diff * pair).sum() / pair.sum().clamp(min=1.0)
    if cfg["lam_boundary"]:
        bce = F.binary_cross_entropy_with_logits(bnd, b, pos_weight=pos_weight, reduction="none")
        total = total + cfg["lam_boundary"] * (bce * valid).sum() / valid.sum().clamp(min=1.0)
    if touch_logits is not None:
        tv = (t != -100).float()
        logp = F.log_softmax(touch_logits, dim=1)
        ce = -(logp.gather(1, t.clamp(min=0).unsqueeze(1)).squeeze(1) * tv).sum() / tv.sum().clamp(min=1.0)
        total = total + cfg["lam_touch"] * ce
    return total


def fit(train, cfg, seed, tag=""):
    """Train one model on CUDA.

    Args:
        train: list of sequence dicts (load_seqs values), profile r1.
        cfg: a CONFIGS entry.
        seed: seeds torch (initialisation, dropout) and the window order.
        tag: printed with the progress line.

    Returns:
        dict: state (CPU tensors), mean, std, cols, p1_index, cfg, loss per epoch, seconds.
    """
    torch = _torch()
    dev = torch.device("cuda")
    started = time.time()
    cols = list(train[0]["names"])
    p1_index = [cols.index(c) for c in P1_NAMES]
    # z-score from the graded frames of both views, accumulated per sequence (low memory; since 11 Nov
    # evening -- earlier models used one stacked array, which differs only in floating-point rounding)
    count = np.zeros(len(cols))
    total = np.zeros(len(cols))
    square = np.zeros(len(cols))
    for d in train:
        graded = d["labels"] != X
        for key in ("x_home", "x_swap"):
            x = d[key][graded].astype(np.float64)
            seen = ~np.isnan(x)
            x = np.where(seen, x, 0.0)
            count += seen.sum(0)
            total += x.sum(0)
            square += (x * x).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean * mean, 0.0))
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    mean = np.nan_to_num(mean, nan=0.0)
    is_token = np.array([c.startswith("pl_") for c in cols])
    mean, std = np.where(is_token, 0.0, mean), np.where(is_token, 1.0, std)
    xs, ys, bs, ms, ts, pos, neg = [], [], [], [], [], 0.0, 0.0   # per-sequence tensors, already on CUDA
    use_touch = bool(cfg.get("lam_touch"))
    for d in train:
        labels = d["labels"].astype(np.int64)
        blocks = [tuple(int(v) for v in b) for b in d["blocks"]]
        bnd = boundary_target(labels, cfg["boundary_radius"])
        valid = labels != X
        pos += float(bnd[valid].sum()) * 2
        neg += float((1 - bnd[valid]).sum()) * 2
        touch = d["touch"].astype(np.int64) if use_touch else None
        for x, lab, tch in ((d["x_home"], labels, touch),
                            (d["x_swap"], SWAP_LABEL[labels], None if touch is None else SWAP_LABEL[touch])):
            parts = _windows(_scale(x, mean, std), lab, bnd, blocks, cfg["window"], tch)
            for store, part in zip((xs, ys, bs, ms, ts), parts):
                store.append(torch.from_numpy(np.stack(part)).to(dev))
            del parts
    x_all, y_all, b_all, m_all, t_all = (torch.cat(v) for v in (xs, ys, bs, ms, ts))
    del xs, ys, bs, ms, ts
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(len(cols), p1_index, cfg).to(dev)
    if cfg.get("opt") == "adamw_cos":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        n_steps = cfg["epochs"] * int(np.ceil(len(x_all) / cfg["batch"]))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=n_steps, pct_start=0.05,
                                                    anneal_strategy="cos")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = None
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg.get("ema") else None
    pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    log = []
    for _ in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(x_all))
        total, steps = 0.0, 0
        for i in range(0, len(order), cfg["batch"]):
            idx = torch.from_numpy(order[i:i + cfg["batch"]]).to(dev)
            y = y_all[idx]
            if not bool((y != -100).any()):
                continue
            outs, bnd, touch_logits = model(x_all[idx])
            loss = _loss(torch, outs, bnd, y, b_all[idx], m_all[idx], pos_weight, cfg, touch_logits, t_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(cfg["ema"]).add_(v.detach(), alpha=1.0 - cfg["ema"])
                        else:
                            ema[k].copy_(v)
            total += loss.item()
            steps += 1
        log.append(total / steps)
    print("  fit {} seed {}: final loss {:.4f} ({:.0f} s)".format(tag, seed, log[-1], time.time() - started),
          flush=True)
    state = ema if ema is not None else model.state_dict()
    return {"state": {k: v.cpu() for k, v in state.items()}, "mean": mean, "std": std,
            "cols": cols, "p1_index": p1_index, "cfg": cfg, "loss": log, "seconds": time.time() - started}


def predict(saved, d, with_touch=False):
    """Team-symmetric probabilities (N, 3) of one match: 0.5 * (P_home + swap(P_swap)); 1/3 outside blocks.

    With with_touch (a model trained with the touch head), returns (probabilities, touch-level
    probabilities), both team-symmetric.
    """
    torch = _torch()
    dev = torch.device("cuda")
    if list(d["names"]) != saved["cols"]:
        raise SystemExit("columns differ between training and test sequences")
    model = build_model(len(saved["cols"]), saved["p1_index"], saved["cfg"]).to(dev)
    model.load_state_dict(saved["state"])
    model.eval()
    blocks = [tuple(int(v) for v in b) for b in d["blocks"]]
    views, tviews = [], []
    for key in ("x_home", "x_swap"):
        x = _scale(d[key], saved["mean"], saved["std"])
        out = np.full((len(x), 3), 1.0 / 3.0)
        tout = np.full((len(x), 3), 1.0 / 3.0)
        with torch.no_grad():
            for _, first, last in blocks:
                xb = torch.from_numpy(np.ascontiguousarray(x[first:last + 1].T)[None]).to(dev)
                outs, _, touch = model(xb)
                out[first:last + 1] = torch.softmax(outs[-1], dim=1)[0].T.double().cpu().numpy()
                if with_touch:
                    tout[first:last + 1] = torch.softmax(touch, dim=1)[0].T.double().cpu().numpy()
        views.append(out)
        tviews.append(tout)

    def symmetric(v):
        p = np.empty_like(v[0])
        p[:, 0] = 0.5 * (v[0][:, 0] + v[1][:, 1])
        p[:, 1] = 0.5 * (v[0][:, 1] + v[1][:, 0])
        p[:, 2] = 0.5 * (v[0][:, 2] + v[1][:, 2])
        return p

    return (symmetric(views), symmetric(tviews)) if with_touch else symmetric(views)


def _seed_dir(config, seed):
    return os.path.join(OUT_DIR, config, "s{}".format(seed))


def rule_labels(name, control_s):
    """Labels of a calibration match under another control rule: controlled(touch_labels, control_s)."""
    from external import sportec_dfl
    return sportec_dfl.controlled(touch_labels(name), control_s)


SET_DIR = os.path.join(CACHE_DIR, "tcn_set")     # player tokens per source (seq_set)


def _source_parts(src):
    """(profile, realisation, flip) of a source name: "r1", "k" or "<profile>_a<r>" (odd r is mirrored)."""
    if src in ("r1", "k"):
        return src, 0, False
    profile, r = src.split("_a")
    return profile, int(r), int(r) % 2 == 1


def seq_set(sources):
    """Player tokens (player_tokens.tokens) of every calibration match for the given sources, written
    to cache/tcn_set/<source>/<match>.npz. The match is rebuilt with load_realisation, so the tokens
    belong to the same degradation realisation as the source's feature sequences."""
    for src in sources:
        profile, r, flip = _source_parts(src)
        out_dir = os.path.join(SET_DIR, src)
        os.makedirs(out_dir, exist_ok=True)
        for name, _ in lomo_mod.calibration_matches():
            target = os.path.join(out_dir, name + ".npz")
            if os.path.isfile(target):
                continue
            started = time.time()
            f, blocks, labels = load_realisation(name, profile, r, flip)
            names, t_home, t_swap = player_tokens.tokens(f, blocks)
            np.savez_compressed(target, t_home=t_home, t_swap=t_swap, names=np.array(names), n=len(labels))
            print("  tokens {} {} ({:.0f} s)".format(src, name, time.time() - started), flush=True)


def _attach_tokens(seqs, src):
    """Append the token columns of a source to its sequences (in place)."""
    for name, d in seqs.items():
        with np.load(os.path.join(SET_DIR, src, name + ".npz")) as z:
            if len(z["t_home"]) != len(d["labels"]):
                raise SystemExit("tokens of {} {} have another length".format(src, name))
            d["x_home"] = np.concatenate((d["x_home"].astype(np.float32), z["t_home"]), axis=1)
            d["x_swap"] = np.concatenate((d["x_swap"].astype(np.float32), z["t_swap"]), axis=1)
            d["names"] = np.concatenate((d["names"], z["names"]))
    return seqs


def _train_sets(sources, control_s=None, with_tokens=False):
    """[{match: seq}] for every training source: "r1" (the original cache/tcn/r1) or an augmentation
    folder name under cache/tcn_aug (e.g. "r1_a1"). With control_s, the labels are rebuilt with that
    control rule (rule_labels); at 2.0 they must equal the cached labels."""
    out = []
    touch = {}
    for src in sources:
        seqs = load_seqs(src) if src in ("r1", "k") else load_seqs(src, seq_dir=AUG_DIR)
        if with_tokens:
            _attach_tokens(seqs, src)
        for name, d in seqs.items():
            if name not in touch:
                touch[name] = touch_labels(name)
            if len(touch[name]) != len(d["labels"]):
                raise SystemExit("touch labels of {} have another length".format(name))
            d["touch"] = touch[name]
            if control_s is not None:
                from external import sportec_dfl
                relabelled = sportec_dfl.controlled(touch[name], control_s)
                if control_s == 2.0 and not np.array_equal(relabelled, d["labels"]):
                    raise SystemExit("rule_labels does not reproduce the 2 s labels of {}".format(name))
                d["labels"] = relabelled
        out.append(seqs)
    return out


def touch_labels(name):
    """Touch-level H/A/D/X labels of a calibration match: the labels before the 2 s control rule.

    DFL: BallStatus / BallPossession as delivered. Metrica: the event-derived labels before
    sportec_dfl.controlled(). The same for every degradation realisation of the match.
    """
    from external import metrica_sample
    if name.startswith("dfl_"):
        with np.load(os.path.join(CACHE_DIR, "dfl_{}.npz".format(name[4:]))) as z:
            status, poss = z["status"], z["possession"]
        return np.where(status == 0, D, np.where(poss == 1, H, np.where(poss == 2, A, X))).astype(np.int8)
    game = int(name[-1])
    periods = metrica_sample.load_tracking(game)["periods"]
    return metrica_sample.labels_from_events(metrica_sample.events_csv(game), len(periods), periods)


SKC_SEQ_DIR = os.path.join(CACHE_DIR, "tcn_skc")


def load_skc():
    """{skc_<id>: seq dict with the decoder features} from external/skillcorner.py seq."""
    out = {}
    for fn in sorted(os.listdir(SKC_SEQ_DIR)):
        with np.load(os.path.join(SKC_SEQ_DIR, fn)) as z:
            out[fn[:-4]] = {k: z[k] for k in z.files}
    return out


def _extras(cfg):
    """Extra training sequences of a configuration (cfg["extra"] == "skc": the SkillCorner matches)."""
    if cfg.get("extra") != "skc":
        return []
    return [{k: d[k] for k in ("x_home", "x_swap", "labels", "blocks", "names", "touch")} for d in load_skc().values()]


def lomo(config, seeds, test_profile="k"):
    """Leave-one-match-out probabilities for every seed (resumable).

    The training inputs are the configuration's "sources" (default: the original r1 sequences); every
    source contributes its realisation of the 7 training matches. The held-out match is always
    predicted on its original test sequence, so every configuration is scored on the same inputs.
    """
    cfg = CONFIGS[config]
    sets = _train_sets(cfg.get("sources", ("r1",)), cfg.get("control_s"), bool(cfg.get("set")))
    extras = _extras(cfg)
    test = load_seqs(cfg["test"], seq_dir=AUG_DIR) if cfg.get("test") else load_seqs(test_profile)
    if cfg.get("set"):
        _attach_tokens(test, test_profile)
    mirrored = load_seqs("k_a0", seq_dir=AUG_DIR) if cfg.get("tta") else None   # the test match mirrored (y -> W - y)
    names = list(test)
    for seed in seeds:
        os.makedirs(_seed_dir(config, seed), exist_ok=True)
        for held in names:
            target = os.path.join(_seed_dir(config, seed), "probs_{}.npy".format(held))
            if os.path.isfile(target):
                continue
            train = [s[n] for s in sets for n in names if n != held] + extras
            saved = fit(train, cfg, seed, "{} without {}".format(config, held))
            if cfg.get("lam_touch"):
                probs, touch = predict(saved, test[held], with_touch=True)
                np.save(os.path.join(_seed_dir(config, seed), "touch_{}.npy".format(held)), touch.astype(np.float32))
            else:
                probs = predict(saved, test[held])
            if mirrored is not None:
                np.save(os.path.join(_seed_dir(config, seed), "probsflip_{}.npy".format(held)),
                        predict(saved, mirrored[held]).astype(np.float32))
            np.save(target, probs.astype(np.float32))
            with open(os.path.join(_seed_dir(config, seed), "fit_{}.json".format(held)), "w") as handle:
                json.dump({"loss": saved["loss"], "seconds": saved["seconds"]}, handle)


def final(config, seeds):
    """One model per seed on all 8 calibration matches (every source of the configuration)."""
    torch = _torch()
    cfg = CONFIGS[config]
    sets = _train_sets(cfg.get("sources", ("r1",)), cfg.get("control_s"))
    for seed in seeds:
        os.makedirs(_seed_dir(config, seed), exist_ok=True)
        target = os.path.join(_seed_dir(config, seed), "model_final.pt")
        if not os.path.isfile(target):
            torch.save(fit([d for s in sets for d in s.values()] + _extras(cfg), cfg, seed, config + " final"),
                       target)


def seed_probs(config, seed, names):
    """{match: (N, 3) float64} of one seed."""
    return {n: np.load(os.path.join(_seed_dir(config, seed), "probs_{}.npy".format(n))).astype(np.float64)
            for n in names}


def ensemble(config, seeds, names):
    """Arithmetic mean of the seeds' probabilities, per match."""
    sets = [seed_probs(config, s, names) for s in seeds]
    return {n: np.mean([s[n] for s in sets], axis=0) for n in names}


def blend(refined, tcn, w):
    """exp(w * log p_refined + (1 - w) * log p_tcn), each clipped at FLOOR (tcn.score's blend)."""
    return np.exp(w * np.log(np.clip(refined, FLOOR, 1.0)) + (1 - w) * np.log(np.clip(tcn, FLOOR, 1.0)))


def decode_scores(folds, probs, alpha, c_switch=8.0, l_min=None, restart=None, cache=None, truth=None,
                  true_dead=False):
    """Decode every fold with the given probabilities and score it.

    Args:
        folds: min_duration.load_folds().
        probs: {match: (N, 3)}.
        alpha, c_switch: decoder weight of the log-probabilities and the H <-> A cost.
        l_min, restart: the R1 minimum-duration decoder (None = the 3-state decoder).
        cache: evidence cache shared between calls.
        truth: optional {match: labels} to score against instead of the folds' labels (E63).
        true_dead: impose the truth's dead-ball frames on the prediction before scoring (D where the
            truth is D; a diagnostic that isolates team identity and timing, E64a).

    Returns:
        list[dict]: per match: spell, n_pred, n_gt, cats, start, end, flags (5, n_gt) bool matched per
        IoU threshold (for the bootstrap).
    """
    cache = {} if cache is None else cache
    saved = possession.MIN_DURATION, possession.MIN_RESTART
    possession.MIN_DURATION, possession.MIN_RESTART = l_min, restart
    out = []
    try:
        for f in folds:
            labels = review_ablation.decode_variant(dict(f, probs=probs[f["name"]]),
                                                    {"alpha": alpha, "c_switch": c_switch}, cache)
            true = f["labels"] if truth is None else truth[f["name"]]
            if true_dead:
                labels = np.where(true == D, D, labels).astype(np.int8)
            r = diagnose_errors.breakdown(true, labels)
            runs = pm.valid_runs(true)
            gt = pm.possession_spells(true, runs)
            pr = pm.possession_spells(pm.fill_abstentions(labels, runs), runs)
            overlaps = pm.spell_overlaps(gt, pr)
            flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt), t) for t in pm.SPELL_IOUS])
            out.append({"spell": float(r["spell"]), "n_pred": int(r["n_pred"]), "n_gt": int(r["n_gt"]),
                        "cats": {c: int(r["cats"][c]) for c in CATS}, "start": float(r["start_abs_med"]),
                        "end": float(r["end_abs_med"]), "flags": flags})
    finally:
        possession.MIN_DURATION, possession.MIN_RESTART = saved
    return out


def sign_flip_p(delta):
    """Exact two-sided sign-flip p-value of the mean of per-match differences."""
    delta = np.asarray(delta, dtype=float)
    flips = np.array(list(itertools.product((1, -1), repeat=len(delta))))
    return float(np.mean(np.abs((flips * delta).mean(1)) >= abs(delta.mean()) - 1e-12))


def _f1_from_flags(flags, picks, n_pred):
    """Mean F1 over the IoU thresholds for resampled possessions (pm.bootstrap_spell_f1's formula)."""
    n_gt = flags.shape[1]
    f1s = []
    for row in flags:
        tp = int(row[picks].sum())
        fp, fn = max(0, n_pred - tp), n_gt - tp
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def paired_bootstrap(rows_a, rows_b, index=None, n=BOOT_N, seed=BOOT_SEED):
    """5-95 % interval of mean(spellF1_a - spellF1_b) over matches, resampling true possessions per match.

    The same resampled possessions are used for both systems (paired). As in the organisers'
    bootstrap_spell_f1, the number of predicted possessions stays fixed. It measures the sampling noise
    of the possessions inside these matches, not the variation between matches or between seeds.
    """
    index = range(len(rows_a)) if index is None else index
    rng = np.random.RandomState(seed)
    diffs = np.zeros(n)
    for i in index:
        fa, fb = rows_a[i]["flags"], rows_b[i]["flags"]
        n_gt = fa.shape[1]
        for k in range(n):
            picks = rng.randint(0, n_gt, n_gt)
            diffs[k] += _f1_from_flags(fa, picks, rows_a[i]["n_pred"]) - _f1_from_flags(fb, picks, rows_b[i]["n_pred"])
    diffs /= len(list(index))
    return float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))


def _row(label, rows, base, dfl):
    """One results-table line (REPORT_TEMPLATE columns) and its numbers."""
    s = np.array([r["spell"] for r in rows])
    sb = np.array([r["spell"] for r in base])
    d = s - sb
    lo, hi = paired_bootstrap(rows, base)
    return ("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {:.3f} | {:+.4f} to {:+.4f} | {}/{} | {} | "
            "{:.2f} / {:.2f} | {} |").format(
        label, s.mean(), d.mean(), s[dfl].mean(), d[dfl].mean(), int(np.sum(d > 1e-9)), sign_flip_p(d), lo, hi,
        sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows),
        " | ".join(str(sum(r["cats"][c] for r in rows)) for c in CATS),
        np.mean([r["start"] for r in rows]), np.mean([r["end"] for r in rows]),
        " ".join("{:.4f}".format(v) for v in s))


HEAD = ("| setting | spellF1 (8) | Δ 8 | spellF1 (DFL) | Δ DFL | better | sign-flip p | bootstrap 5–95 % of Δ 8 | "
        "pred / true | short | split_dead | split_team | merged | edge s/e (s) | per match |")


def _base(folds, cache):
    """p3-refined through the same decoder; stops unless it reproduces BASELINE."""
    names = [f["name"] for f in folds]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    base = decode_scores(folds, {f["name"]: f["probs_refined"] for f in folds}, 0.75, cache=cache)
    sb = np.array([r["spell"] for r in base])
    if (round(sb.mean(), 4), round(sb[dfl].mean(), 4)) != BASELINE:
        raise SystemExit("p3-refined base not reproduced: {:.4f} / {:.4f}".format(sb.mean(), sb[dfl].mean()))
    return names, dfl, base


def report_screen(configs, seeds, out_name, title):
    """E70: per configuration, standard decoder alpha 0.75: mean of the seeds and the seed ensemble, with
    and without averaging the prediction on the mirrored test match. Reference: the first configuration."""
    path = os.path.join(REPORT_DIR, out_name)
    if os.path.isfile(path):
        raise SystemExit("{} exists; reports are never overwritten".format(path))
    folds = min_duration.load_folds()
    cache = {}
    names, dfl, _ = _base(folds, cache)

    def load(config, seed, flip):
        stem = "probsflip_{}.npy" if flip else "probs_{}.npy"
        return {n: np.load(os.path.join(_seed_dir(config, seed), stem.format(n))).astype(np.float64) for n in names}

    def score(probs):
        sc = np.array([r["spell"] for r in decode_scores(folds, probs, 0.75, cache=cache)])
        return sc.mean(), sc[dfl].mean()

    lines = ["# " + title, "", "LOMO over the 8 calibration matches, trained on 4 realisations of profile `k`, 10 epochs, "
             "tested on `k`; standard decoder α 0.75; spellF1 over 8 matches / on DFL. Seeds {}. \"mirror\" averages "
             "the model's prediction on the test match and on its mirror image (y → W − y). Reference: `{}`. "
             "p3-refined 0.7077 / 0.6889.".format(list(seeds), configs[0]), "",
             "| configuration | change | per seed (8) | mean of seeds | mean, mirror | ensemble | ensemble, mirror | "
             "mean − reference (8 / DFL) | mirror mean − reference mirror mean |", "|---|---|---|---|---|---|---|---|---|"]
    ref = None
    for config in configs:
        plain = [load(config, s, False) for s in seeds]
        both = [{n: 0.5 * (a[n] + b[n]) for n in names} for a, b in zip(plain, [load(config, s, True) for s in seeds])]
        sp, sb = [score(p) for p in plain], [score(p) for p in both]
        mp, mb = np.mean(sp, axis=0), np.mean(sb, axis=0)
        ep = score({n: np.mean([p[n] for p in plain], axis=0) for n in names})
        eb = score({n: np.mean([p[n] for p in both], axis=0) for n in names})
        if ref is None:
            ref = (mp, mb)
        lines.append("| {} | {} | {} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:+.4f} / {:+.4f} | "
                     "{:+.4f} / {:+.4f} |".format(
                         config, ", ".join("{}={}".format(k, v) for k, v in SCREEN.get(config, {}).items()) or "–",
                         " ".join("{:.4f}".format(v[0]) for v in sp), mp[0], mp[1], mb[0], mb[1], ep[0], ep[1], eb[0], eb[1],
                         mp[0] - ref[0][0], mp[1] - ref[0][1], mb[0] - ref[1][0], mb[1] - ref[1][1]))
        print("  scored {}".format(config), flush=True)
    lines += ["", "Matches: " + ", ".join(names) + "."]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def report_ceiling():
    """E69: the recipe trained and tested on less degraded inputs; the P1 evidence is rebuilt per input."""
    import run
    path = os.path.join(REPORT_DIR, "tcn_gpu_ceiling.md")
    if os.path.isfile(path):
        raise SystemExit("{} exists; reports are never overwritten".format(path))
    rows = (("k (as BAR-ATH is simulated)", "k", "k4_e10"), ("nodrop (the ball never missing)", "nodrop", "ceil_nodrop"),
            ("clean (corrected tracking)", "clean", "ceil_clean"))
    names = [n for n, _ in lomo_mod.calibration_matches()]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    lines = ["# E69: how much the tracking quality costs (ceiling test)", "",
             "The `k4_e10` recipe (seed 0) trained and tested on the same kind of input, LOMO over the 8 calibration "
             "matches, standard decoder α 0.75, P1 evidence rebuilt from the same input. spellF1 over 8 matches / on DFL.", "",
             "| input | P1 rules alone | TCN (seed 0) | TCN, true dead frames given | TCN, true team given | ball visible | per match (TCN) |",
             "|---|---|---|---|---|---|---|"]
    for label, profile, config in rows:
        p1, tc, od, ot, vis = [], [], [], [], []
        for n in names:
            f, blocks, labels = load_realisation(n, profile, 0, False)
            ev = run.evidence(f, blocks)
            probs = np.load(os.path.join(_seed_dir(config, 0), "probs_{}.npy".format(n))).astype(np.float64)
            p1.append(pm.evaluate(labels, run.decode_evidence(ev, blocks))["spell_f1_mean"])
            b = diagnose_errors.breakdown(labels, run.decode_evidence(ev, blocks, probs, 0.75))
            tc.append(b["spell"]), od.append(b["oracle_dead"]), ot.append(b["oracle_team"])
            vis.append(float(np.mean(f["visible"][labels != X])))
            print("  scored {} {}".format(profile, n), flush=True)
        a = [np.array(v) for v in (p1, tc, od, ot)]
        lines.append("| {} | {} | {:.1%} | {} |".format(
            label, " | ".join("{:.4f} / {:.4f}".format(v.mean(), v[dfl].mean()) for v in a), np.mean(vis),
            " ".join("{:.4f}".format(v) for v in tc)))
    lines += ["", "Matches: " + ", ".join(names) + "."]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def report_e59(config="base", seeds=(0, 1, 2, 3, 4)):
    """E59: seed noise and the seed ensemble of the boundary TCN, settings A and B and the alpha grid."""
    started = time.time()
    folds = min_duration.load_folds()
    cache = {}
    names, dfl, base = _base(folds, cache)
    refined = {f["name"]: f["probs_refined"] for f in folds}
    settings = {   # label -> (blend weight or None, alpha, c_switch, l_min, restart)
        "A (TCN, α 0.75, R1)": (None, 0.75, R1[2], R1[1], R1[0]),
        "B (blend w 0.3, α 0.75, standard)": (0.3, 0.75, 8.0, None, None),
        "TCN, α 0.75, standard": (None, 0.75, 8.0, None, None),
    }

    def run_setting(probs, key):
        w, alpha, cs, l_min, restart = settings[key]
        p = probs if w is None else {n: blend(refined[n], probs[n], w) for n in names}
        return decode_scores(folds, p, alpha, cs, l_min, restart, cache)

    lines = ["# E59: seed noise and seed ensemble of the boundary TCN (GPU)", "",
             "LOMO, trained r1, tested kinematic. Baseline p3-refined {:.4f} / {:.4f} reproduced in this run. "
             "Bar {} / {}. Configuration `{}`, seeds {}.".format(*BASELINE, *BAR, config, list(seeds)),
             "Bootstrap: possessions resampled within matches, {} draws, paired with the baseline.".format(BOOT_N),
             "", "## Per seed and ensemble", "", HEAD, "|" + "---|" * (HEAD.count("|") - 1),
             _row("p3-refined (base)", base, base, dfl)]
    per_seed = {k: [] for k in settings}
    for seed in seeds:
        probs = seed_probs(config, seed, names)
        for key in settings:
            rows = run_setting(probs, key)
            per_seed[key].append(rows)
            lines.append(_row("{}, seed {}".format(key, seed), rows, base, dfl))
        print("  scored seed {} ({:.0f} s)".format(seed, time.time() - started), flush=True)
    ens = ensemble(config, seeds, names)
    ens_rows = {}
    for key in settings:
        ens_rows[key] = run_setting(ens, key)
        lines.append(_row("{}, ensemble of {}".format(key, len(seeds)), ens_rows[key], base, dfl))
    lines += ["", "## Seed noise", "",
              "| setting | mean of seeds (8 / DFL) | SD between seeds (8 / DFL) | min – max (8) | ensemble (8 / DFL) | "
              "ensemble − mean of seeds (8 / DFL) | ensemble vs single seeds, bootstrap-free: seeds below the ensemble |",
              "|---|---|---|---|---|---|---|"]
    for key in settings:
        m8 = np.array([np.mean([r["spell"] for r in rows]) for rows in per_seed[key]])
        md = np.array([np.mean([rows[i]["spell"] for i in dfl]) for rows in per_seed[key]])
        e = np.array([r["spell"] for r in ens_rows[key]])
        lines.append("| {} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:.4f} – {:.4f} | {:.4f} / {:.4f} | {:+.4f} / {:+.4f} | "
                     "{}/{} |".format(key, m8.mean(), md.mean(), m8.std(ddof=1), md.std(ddof=1), m8.min(), m8.max(),
                                      e.mean(), e[dfl].mean(), e.mean() - m8.mean(), e[dfl].mean() - md.mean(),
                                      int(np.sum(m8 < e.mean())), len(seeds)))
    # the alpha grid on the ensemble (information; nested choice of alpha on 7 matches)
    lines += ["", "## Decoder weight α on the ensemble (TCN alone)", "", HEAD, "|" + "---|" * (HEAD.count("|") - 1)]
    grid = {}
    for alpha, (dec, cs, l_min, restart) in itertools.product(
            (0.5, 0.75, 1.0, 1.5, 2.0), (("standard", 8.0, None, None), ("R1", R1[2], R1[1], R1[0]))):
        rows = decode_scores(folds, ens, alpha, cs, l_min, restart, cache)
        grid[(alpha, dec)] = np.array([r["spell"] for r in rows])
        lines.append(_row("ensemble, α {}, {}".format(alpha, dec), rows, base, dfl))
        print("  scored alpha {} {} ({:.0f} s)".format(alpha, dec, time.time() - started), flush=True)
    sb = np.array([r["spell"] for r in base])
    nested = []
    for i in range(len(names)):
        others = [j for j in range(len(names)) if j != i]
        pick = max(grid, key=lambda k: grid[k][others].mean())
        nested.append((names[i], pick, grid[pick][i] - sb[i]))
    g = np.array([x[2] for x in nested])
    lines += ["", "Nested choice (setting chosen on the other 7 matches, scored on the 8th): mean Δ {:+.4f} over 8, "
              "{:+.4f} on DFL; chosen: {}.".format(g.mean(), g[dfl].mean(),
                                                   "; ".join("{} α {} {}".format(n, *p) for n, p, _ in nested)),
              "", "Matches: " + ", ".join(names), "", "Scoring time {:.0f} s.".format(time.time() - started)]
    path = os.path.join(REPORT_DIR, "tcn_gpu_seeds.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


DECODES = {   # label -> (blend weight with p3-refined or None, alpha, c_switch, l_min, restart)
    "A": (None, 0.75, R1[2], R1[1], R1[0]),
    "B": (0.3, 0.75, 8.0, None, None),
    "std": (None, 0.75, 8.0, None, None),
}
DECODE_TEXT = {"A": "TCN alone, α 0.75, R1 (steady, L_min 50, C_SWITCH 10)",
               "B": "blend 0.3 · log p_refined + 0.7 · log p_tcn, α 0.75, standard decoder",
               "std": "TCN alone, α 0.75, standard decoder"}


def report_configs(title, configs, seeds, reference, out_name, decodes=("A", "B", "std"), header=()):
    """Compare configurations: per seed, 3-seed (or n-seed) mean and SD, and the seed ensemble.

    Args:
        title: first line of the report.
        configs: configuration names (each with LOMO probabilities for every seed).
        seeds: seeds to use.
        reference: {config: reference config} for the reading rule (difference of seed means against
            twice the larger SD between seeds).
        out_name: report file name in solution/reports/ (never an existing file).
        decodes: keys of DECODES.
        header: extra lines for the top of the report.
    """
    path = os.path.join(REPORT_DIR, out_name)
    if os.path.isfile(path):
        raise SystemExit("{} exists; reports are never overwritten".format(path))
    started = time.time()
    folds = min_duration.load_folds()
    cache = {}
    names, dfl, base = _base(folds, cache)
    refined = {f["name"]: f["probs_refined"] for f in folds}

    def score(probs, key):
        w, alpha, cs, l_min, restart = DECODES[key]
        p = probs if w is None else {n: blend(refined[n], probs[n], w) for n in names}
        return decode_scores(folds, p, alpha, cs, l_min, restart, cache)

    res = {}     # (config, decode) -> {"seeds": [rows], "ens": rows}
    for config in configs:
        ens = ensemble(config, seeds, names)
        for key in decodes:
            res[(config, key)] = {"seeds": [score(seed_probs(config, s, names), key) for s in seeds],
                                  "ens": score(ens, key)}
        print("  scored {} ({:.0f} s)".format(config, time.time() - started), flush=True)

    def means(rows_list):
        m8 = np.array([np.mean([r["spell"] for r in rows]) for rows in rows_list])
        md = np.array([np.mean([rows[i]["spell"] for i in dfl]) for rows in rows_list])
        return m8, md

    lines = ["# " + title, ""] + list(header) + [
        "LOMO over the 8 calibration matches, trained on the configuration's sources, tested on the kinematic "
        "profile `k`. Baseline p3-refined {:.4f} / {:.4f} (8 / DFL), reproduced in this run; bar {} / {}. "
        "Seeds {}. Decodes: {}.".format(*BASELINE, *BAR, list(seeds),
                                        "; ".join("**{}** = {}".format(k, DECODE_TEXT[k]) for k in decodes)), "",
        "## Summary: mean of the single seeds, SD between seeds, seed ensemble", "",
        "| configuration | decode | mean of seeds (8 / DFL) | SD between seeds (8 / DFL) | Δ vs p3-refined (8 / DFL) | "
        "ensemble (8 / DFL) | reference | mean − reference mean (8 / DFL) | 2 × larger SD (8) | reading |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for config in configs:
        for key in decodes:
            m8, md = means(res[(config, key)]["seeds"])
            e = np.array([r["spell"] for r in res[(config, key)]["ens"]])
            ref = reference.get(config)
            if ref is None:
                tail = "– | – | – | –"
            else:
                r8, rd = means(res[(ref, key)]["seeds"])
                d8, dd = m8.mean() - r8.mean(), md.mean() - rd.mean()
                thr = 2 * max(m8.std(ddof=1), r8.std(ddof=1))
                verdict = "better" if (d8 > thr and dd >= 0) else ("worse" if d8 < -thr else "no difference")
                tail = "{} | {:+.4f} / {:+.4f} | {:.4f} | {}".format(ref, d8, dd, thr, verdict)
            lines.append("| {} | {} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:+.4f} / {:+.4f} | {:.4f} / {:.4f} | {} |".format(
                config, key, m8.mean(), md.mean(), m8.std(ddof=1), md.std(ddof=1),
                m8.mean() - BASELINE[0], md.mean() - BASELINE[1], e.mean(), e[dfl].mean(), tail))
    lines += ["", "## Every run", "", HEAD, "|" + "---|" * (HEAD.count("|") - 1), _row("p3-refined (base)", base, base, dfl)]
    for config in configs:
        for key in decodes:
            for seed, rows in zip(seeds, res[(config, key)]["seeds"]):
                lines.append(_row("{}, {}, seed {}".format(config, key, seed), rows, base, dfl))
            lines.append(_row("{}, {}, ensemble of {}".format(config, key, len(seeds)), res[(config, key)]["ens"], base, dfl))
    fits = []
    for config in configs:
        secs = []
        for seed in seeds:
            for n in names:
                with open(os.path.join(_seed_dir(config, seed), "fit_{}.json".format(n))) as handle:
                    secs.append(json.load(handle)["seconds"])
        fits.append("{} {:.0f} s".format(config, np.mean(secs)))
    lines += ["", "Training time per fold model (CUDA): " + ", ".join(fits) + ".",
              "Matches: " + ", ".join(names) + ".", "Scoring time {:.0f} s.".format(time.time() - started)]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


RULES = (1.0, 1.5, 2.0, 3.0)          # E63: control rules of the truth, seconds
RULE_CONFIGS = {1.0: "ctrl1.0", 1.5: "ctrl1.5", 2.0: "aug4_e10", 3.0: "ctrl3.0"}
RULE_DECODES = {"std α 0.75": (0.75, 8.0, None, None), "std α 1.5": (1.5, 8.0, None, None),
                "R1 α 1.5": (1.5, R1[2], R1[1], R1[0])}


def report_rules(seeds=(0, 1, 2), configs=None, out_name="tcn_gpu_label_rule.md", recipe="aug4_e10", title="E63"):
    """E63 / E67: 3-seed ensembles trained on each control rule, scored against each rule's labels (8 matches).

    Args:
        seeds: seeds of the ensembles.
        configs: {T_train: configuration} (RULE_CONFIGS if None).
        out_name: report file in solution/reports/ (never an existing file).
        recipe, title: named in the report's header.
    """
    configs = RULE_CONFIGS if configs is None else configs
    path = os.path.join(REPORT_DIR, out_name)
    if os.path.isfile(path):
        raise SystemExit("{} exists; reports are never overwritten".format(path))
    folds = min_duration.load_folds()
    cache = {}
    names, dfl, _ = _base(folds, cache)
    truths = {t: {n: rule_labels(n, t) for n in names} for t in RULES}
    for n, f in zip(names, folds):
        if not np.array_equal(truths[2.0][n], f["labels"]):
            raise SystemExit("2 s labels of {} differ from the fold cache".format(n))
    lines = ["# " + title + ": control rule of the training labels against the control rule of the truth (TCN, GPU)", "",
             "3-seed ensembles (seeds {}) of the `" + recipe + "` recipe, trained on labels built with a T_train-second "
             "control rule, each scored against labels built with T_eval. LOMO over the 8 calibration matches, "
             "tested on profile `k`. spellF1 over 8 matches / on DFL. The same P1 evidence and decoder throughout; "
             "p3-refined (trained on 2 s labels) is the first row of each block.".format(list(seeds)), ""]
    refined = {f["name"]: f["probs_refined"] for f in folds}
    for dec, (alpha, cs, l_min, restart) in RULE_DECODES.items():
        lines += ["## Decode: {}".format(dec), "",
                  "| trained on | " + " | ".join("T_eval {} s".format(t) for t in RULES) + " | worst case (8) | mean (8) |",
                  "|---|" + "---|" * (len(RULES) + 2)]
        rows = [("p3-refined (2 s), its own decode", refined, (0.75, 8.0, None, None))]
        rows += [("T_train {} s".format(t), ensemble(configs[t], seeds, names), (alpha, cs, l_min, restart))
                 for t in sorted(configs)]
        for label, probs, (a, c, lm, rs) in rows:
            cells, vals = [], []
            for t in RULES:
                sc = np.array([r["spell"] for r in decode_scores(folds, probs, a, c, lm, rs, cache, truths[t])])
                cells.append("{:.4f} / {:.4f}".format(sc.mean(), sc[dfl].mean()))
                vals.append(sc.mean())
            lines.append("| {} | {} | {:.4f} | {:.4f} |".format(label, " | ".join(cells), min(vals), np.mean(vals)))
            print("  scored {} {}".format(dec, label), flush=True)
        lines.append("")
    stats = []
    for t in RULES:
        n_pos = sum(len(pm.possession_spells(truths[t][n], pm.valid_runs(truths[t][n]))) for n in names)
        stats.append("{} s: {}".format(t, n_pos))
    lines += ["True possessions over the 8 matches by rule: " + ", ".join(stats) + ".", "Matches: " + ", ".join(names) + "."]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


SKC_MODELS = {"base": (0, 1, 2, 3, 4), "aug4_e10": (0, 1, 2), "k4_e10": (0, 1, 2), "mix8": (0, 1, 2)}
SKC_DECODES = {"std α 0.75": (0.75, 8.0, None, None), "std α 1.5": (1.5, 8.0, None, None),
               "R1 α 1.5": (1.5, R1[2], R1[1], R1[0])}


def skc_predict():
    """Final models (trained on the 8 calibration matches) applied to the SkillCorner matches; the seed
    ensemble's probabilities go to cache/tcn_gpu/<config>/skc/<match>.npy."""
    torch = _torch()
    data = load_skc()
    for config, seeds in SKC_MODELS.items():
        out_dir = os.path.join(OUT_DIR, config, "skc")
        os.makedirs(out_dir, exist_ok=True)
        if all(os.path.isfile(os.path.join(out_dir, n + ".npy")) for n in data):
            continue
        models = [torch.load(os.path.join(_seed_dir(config, s), "model_final.pt"), weights_only=False) for s in seeds]
        for name, d in data.items():
            np.save(os.path.join(out_dir, name + ".npy"),
                    np.mean([predict(m, d) for m in models], axis=0).astype(np.float32))
        print("  predicted {} on {} SkillCorner matches".format(config, len(data)), flush=True)


def report_skc(out_name="tcn_gpu_skillcorner.md"):
    """E64a: models trained on the calibration matches, scored on the SkillCorner matches (2 s rule and
    touch-level truth). Report: reports/tcn_gpu_skillcorner.md."""
    path = os.path.join(REPORT_DIR, out_name)
    if os.path.isfile(path):
        raise SystemExit("{} exists; reports are never overwritten".format(path))
    data = load_skc()
    names = list(data)
    folds = [{"name": n, "labels": d["labels"], "blocks": [tuple(int(v) for v in b) for b in d["blocks"]],
              "f": dict({k[2:]: d[k] for k in d if k.startswith("f_")}, pitch_length=float(d["pitch_length"]))}
             for n, d in data.items()]
    two_s = {n: d["labels"] for n, d in data.items()}
    truths = {"2 s rule": (two_s, False), "touch level": ({n: d["touch"] for n, d in data.items()}, False),
              "2 s rule, the truth's dead-ball frames imposed on every prediction": (two_s, True)}
    cache = {}
    systems = [("p3-refined (submission), α 0.75 standard", {n: d["probs_refined"].astype(np.float64) for n, d in data.items()},
                (0.75, 8.0, None, None))]
    for config, seeds in SKC_MODELS.items():
        probs = {n: np.load(os.path.join(OUT_DIR, config, "skc", n + ".npy")).astype(np.float64) for n in names}
        for dec, setting in SKC_DECODES.items():
            systems.append(("{} ensemble of {}, {}".format(config, len(seeds), dec), probs, setting))
    lines = ["# E64a: models trained on the 8 calibration matches, tested on the 20 SkillCorner matches", "",
             "SkillCorner open data (A-League 2024/25, broadcast tracking at 10 fps interpolated to 25 fps, the ball only "
             "where it was detected, no simulated degradation). Truth: SkillCorner's possession, with the 2 s control "
             "rule and at touch level. No model saw these matches. Δ and the interval are against p3-refined, paired "
             "over the 20 matches (bootstrap over matches, 5–95 %, 10000 draws).", ""]
    rng = np.random.RandomState(0)
    for tname, (truth, true_dead) in truths.items():
        lines += ["## Truth: {}".format(tname), "",
                  "| system | spellF1 (20 matches) | Δ | matches better | 5–95 % of Δ | min – max | pred / true possessions |",
                  "|---|---|---|---|---|---|---|"]
        base = None
        for label, probs, (a, c, lm, rs) in systems:
            rows = decode_scores(folds, probs, a, c, lm, rs, cache, truth, true_dead)
            sc = np.array([r["spell"] for r in rows])
            if base is None:
                base = sc
            d = sc - base
            boot = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(10000)])
            lines.append("| {} | {:.4f} | {:+.4f} | {}/{} | {:+.4f} to {:+.4f} | {:.4f} – {:.4f} | {}/{} |".format(
                label, sc.mean(), d.mean(), int(np.sum(d > 1e-9)), len(d), np.percentile(boot, 5), np.percentile(boot, 95),
                sc.min(), sc.max(), sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows)))
            print("  scored {} / {}".format(tname, label), flush=True)
        lines.append("")
    lines.append("Matches: " + ", ".join(names) + ".")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


HOLDOUT_BASE = {"dfl": 0.6895, "metrica3": 0.7681}   # p3-refined, reports/holdout_log.md (as tcn.py)


def holdout(which, candidates):
    """Declared candidates on one held-out match, kinematic test inputs. ONLY WITH THE USER'S APPROVAL.

    Each candidate is the seed ensemble of a configuration's final models (trained on the 8 calibration
    matches), decoded with the standard decoder at alpha 0.75. p3-refined must first reproduce its logged
    held-out score. One line per candidate is appended to reports/holdout_log.md.

    Args:
        which: "dfl" (J03WR9) or "metrica3".
        candidates: [(config, seeds)].
    """
    import learned
    import run
    import stage2
    import team_shape
    from external import dropout_robustness, holdout as ho
    torch = _torch()
    dropout_robustness._use("k")
    base_groups = team_shape.LOCAL_GROUPS
    f, blocks, truth = ho.HOLDOUTS[which]()
    ev = run.evidence(f, blocks)
    bundle = learned.load()
    names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    if names != bundle["names"]:
        raise SystemExit("P3 feature set differs from the saved model")
    refiner = stage2.load()
    refined = stage2.refine(refiner, f, blocks, ev, learned.predict_proba(bundle["model"], x_home, x_swap))
    base = pm.evaluate(truth, run.decode_evidence(ev, blocks, refined, refiner["alpha"]))
    if round(base["spell_f1_mean"], 4) != HOLDOUT_BASE[which]:
        raise SystemExit("p3-refined held-out {} not reproduced: {:.4f}".format(which, base["spell_f1_mean"]))
    team_shape.LOCAL_GROUPS = GROUPS
    try:
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    finally:
        team_shape.LOCAL_GROUPS = base_groups
    keep = [i for i, c in enumerate(names) if not c.endswith(DROP_SUFFIXES)]
    d = {"x_home": x_home[:, keep], "x_swap": x_swap[:, keep], "names": np.array([names[i] for i in keep]),
         "blocks": np.array(blocks, dtype=np.int64)}
    fold = {"name": which, "f": f, "blocks": blocks}
    lines = []
    for config, seeds in candidates:
        models = [torch.load(os.path.join(_seed_dir(config, s), "model_final.pt"), weights_only=False) for s in seeds]
        probs = np.mean([predict(m, d) for m in models], axis=0)
        labels = review_ablation.decode_variant(dict(fold, probs=probs), {"alpha": 0.75, "c_switch": 8.0}, {})
        result = pm.evaluate(truth, labels)
        note = "TCN {} (GPU, final, 8 matches), ensemble of seeds {}, alpha 0.75, standard decoder, kinematic test".format(
            config, ",".join(str(s) for s in seeds))
        tagged = note if which == "metrica3" else "[{}] {}".format(which, note)
        lines.append("| {} | {} | {} | {:.4f} | {:.4f} | {}/{} |".format(
            time.strftime("%Y-%m-%d"), ho.commit_id(), tagged, result["spell_f1_mean"], result["macro_f1"],
            result["n_pred_spells"], result["n_gt_spells"]))
    with open(ho.LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("  p3-refined reproduced: {:.4f}".format(base["spell_f1_mean"]), flush=True)


AUG_SEED_STEP = 7919          # realisation r uses seed + r * AUG_SEED_STEP (r = 0 is the original)
GROUPS = ("ball", "near3", "near1")
DROP_SUFFIXES = ("_m25", "_m75")


def load_realisation(name, profile, r, flip):
    """One calibration match with degradation realisation r (external data only).

    The same steps as sportec_dfl.load_match / metrica_sample.load_game with their seed replaced by
    seed + r * AUG_SEED_STEP; with flip, players and ball are mirrored across the long axis (y -> width - y)
    before the features are built. Labels are unchanged. r = 0 without flip is the original.

    Args:
        name: "metrica1", "metrica2" or "dfl_<id>".
        profile: "r1" or "k" (dropout_robustness.PROFILES); for the ceiling test of E69 also "clean" (no
            degradation at all) and "nodrop" (noise and player dropout as usual, the ball never dropped).
        r: realisation index (>= 0).
        flip: mirror across the long axis.

    Returns:
        tuple: (features dict, blocks, labels) as the loaders return them.
    """
    import features
    from external import dropout_robustness, metrica_sample, sportec_dfl
    ceiling = profile if profile in ("clean", "nodrop") else None
    dropout_robustness._use("k" if ceiling else profile)
    if name.startswith("dfl_"):
        match = name[4:]
        with np.load(os.path.join(CACHE_DIR, "dfl_{}.npz".format(match))) as cached:
            raw = {k: cached[k] for k in cached.files}
        touch = np.where(raw["status"] == 0, D, np.where(raw["possession"] == 1, H,
                                                         np.where(raw["possession"] == 2, A, X))).astype(np.int8)
        labels = sportec_dfl.controlled(touch)
        seed = sum(ord(c) for c in match) + r * AUG_SEED_STEP
        players, ball = raw["players"].copy(), raw["ball"]
        players[:, :, 2] *= sportec_dfl.SPEED_TO_MS
        slot_team, periods = raw["slot_team"], raw["periods"]
        length, width = float(raw["pitch"][0]), float(raw["pitch"][1])
    else:
        game = int(name[-1])
        tracking = metrica_sample.load_tracking(game)
        periods = tracking["periods"]
        labels = metrica_sample.labels_from_events(metrica_sample.events_csv(game), len(periods), periods)
        if metrica_sample.CONTROLLED_LABELS:
            labels = sportec_dfl.controlled(labels)
        seed = game + r * AUG_SEED_STEP
        players, ball, slot_team = tracking["players"], tracking["ball"], tracking["slot_team"]
        length, width = metrica_sample.PITCH
    if ceiling != "clean":
        ball = metrica_sample.degrade_ball(ball, seed=seed)
        players = metrica_sample.degrade_players(players, seed=seed)
    if ceiling is None:
        ball = sportec_dfl.drop_ball(ball, labels, seed, features.team_speed(players))
    if flip:
        players, ball = players.copy(), ball.copy()
        players[:, :, 1] = width - players[:, :, 1]
        ball[:, 1] = width - ball[:, 1]
    ball = features.clean_ball(ball)
    near_home, near_away = features.nearest_by_team(players, ball, slot_team)
    f = {"ball": ball, "visible": ~np.isnan(ball[:, 0]), "ball_speed": features.ball_speed(ball),
         "near_home": near_home, "near_away": near_away, "team_speed": features.team_speed(players),
         "outside": features.outside_pitch(ball), "slot_team": slot_team, "players": players,
         "pitch_length": float(length)}
    blocks = []
    for period in (1, 2):
        index = np.flatnonzero(periods == period)
        if len(index):
            blocks.append((period, int(index[0]), int(index[-1])))
    return f, blocks, labels


def seq_aug(profile, r, flip, check=False):
    """Sequences (the 75 columns of tcn.seq) of realisation r for every calibration match.

    With check, realisation 0 without flip is compared with the cached original of tcn.py and nothing is
    written: it must be identical, which verifies load_realisation against the loaders.
    """
    import learned
    import run
    import team_shape
    team_shape.LOCAL_GROUPS = GROUPS
    out_dir = os.path.join(AUG_DIR, "{}_a{}".format(profile, r))
    os.makedirs(out_dir, exist_ok=True)
    for name, _ in lomo_mod.calibration_matches():
        target = os.path.join(out_dir, name + ".npz")
        if os.path.isfile(target) and not check:
            continue
        started = time.time()
        f, blocks, labels = load_realisation(name, profile, r, flip)
        ev = run.evidence(f, blocks)
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
        keep = [i for i, c in enumerate(names) if not c.endswith(DROP_SUFFIXES)]
        if check:
            with np.load(os.path.join(SEQ_DIR, profile, name + ".npz")) as z:
                same = (np.array_equal(z["x_home"], x_home[:, keep], equal_nan=True)
                        and np.array_equal(z["x_swap"], x_swap[:, keep], equal_nan=True)
                        and np.array_equal(z["labels"], labels))
            print("  check {} {}: {}".format(profile, name, "identical" if same else "DIFFERENT"), flush=True)
            if not same:
                raise SystemExit("load_realisation does not reproduce the loader")
            continue
        np.savez(target, x_home=x_home[:, keep], x_swap=x_swap[:, keep], labels=labels,
                 blocks=np.array(blocks, dtype=np.int64), names=np.array([names[i] for i in keep]))
        print("  seq {} a{} {} flip {} ({:.0f} s)".format(profile, r, name, flip, time.time() - started), flush=True)


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "lomo":
        lomo(sys.argv[2], [int(s) for s in sys.argv[3:]])
    elif phase == "final":
        final(sys.argv[2], [int(s) for s in sys.argv[3:]])
    elif phase == "e59":
        report_e59()
    elif phase == "e60":
        report_configs("E60: several degradation realisations per training match (boundary TCN, GPU)",
                       ("base", "aug4", "aug4_e10", "aug4k"), (0, 1, 2),
                       {"aug4": "base", "aug4_e10": "base", "aug4k": "base"}, "tcn_gpu_aug.md")
    elif phase == "e61":
        report_configs("E61: training profile and auxiliary touch-level target (boundary TCN, GPU)",
                       ("aug4", "aug4_e10", "k4", "k4_e10", "mix8", "aug4_touch0.3", "aug4_touch1"), (0, 1, 2),
                       {"k4": "aug4", "k4_e10": "aug4_e10", "mix8": "aug4_e10", "aug4_touch0.3": "aug4",
                        "aug4_touch1": "aug4"},
                       "tcn_gpu_profile_touch.md")
    elif phase == "e62":
        report_configs("E62: longer training with and without realisations; more realisations (boundary TCN, GPU)",
                       ("base", "base_e40", "aug4_e10", "aug4_e20", "aug8_e10"), (0, 1, 2),
                       {"base_e40": "base", "aug4_e10": "base_e40", "aug4_e20": "aug4_e10", "aug8_e10": "aug4_e20"},
                       "tcn_gpu_steps.md")
    elif phase == "e63":
        report_rules()
    elif phase == "seq_set":
        seq_set(sys.argv[2:])
    elif phase == "e65":
        report_configs("E65: a per-player encoder in front of the TCN (GPU)", ("k4_e10", "set_k4_e10"), (0, 1, 2),
                       {"set_k4_e10": "k4_e10"}, "tcn_gpu_set.md")
    elif phase == "e66":
        report_configs("E66: 8 realisations of profile k, and 8 + 8 of both profiles (boundary TCN, GPU)",
                       ("k4_e10", "aug8_e10", "k8_e10"), (0, 1, 2), {"k8_e10": "k4_e10"}, "tcn_gpu_k8.md")
    elif phase == "e67":
        report_rules(configs={1.5: "k4_e10_c1.5", 2.0: "k4_e10"}, out_name="tcn_gpu_label_rule_k.md",
                     recipe="k4_e10", title="E67")
    elif phase == "holdout":      # holdout <dfl|metrica3> <config>:<seeds,> ...   (declared candidates only)
        holdout(sys.argv[2], [(a.split(":")[0], tuple(int(v) for v in a.split(":")[1].split(","))) for a in sys.argv[3:]])
    elif phase == "e70":          # e70 <report name> <seeds, comma separated> <config> ...
        report_screen(sys.argv[4:], tuple(int(v) for v in sys.argv[3].split(",")), sys.argv[2],
                      "E70: architecture and training screen on the k4_e10 recipe (boundary TCN, GPU)")
    elif phase == "e69":
        report_ceiling()
    elif phase == "e71":
        report_configs("E71: AdamW with a cosine schedule and weight averaging on the k4_e10 recipe (boundary TCN, GPU)",
                       ("k4_e10", "x_ema", "x_cos", "y_cos_ema"), (0, 1, 2),
                       {"x_ema": "k4_e10", "x_cos": "k4_e10", "y_cos_ema": "k4_e10"}, "tcn_gpu_cos_ema.md")
    elif phase == "skc_predict":
        skc_predict()
    elif phase == "e64a":
        report_skc(*sys.argv[2:3])
    elif phase == "e64b":
        report_configs("E64b: the 20 SkillCorner matches as extra training data (boundary TCN, GPU)",
                       ("aug4_e10", "aug4_e10_skc"), (0, 1, 2), {"aug4_e10_skc": "aug4_e10"}, "tcn_gpu_skc_train.md")
    elif phase == "seq_aug":      # seq_aug <profile> <r> <flip 0|1> [check]
        seq_aug(sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", check=len(sys.argv) > 5)
