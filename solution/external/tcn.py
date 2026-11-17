"""R5 (+ R3) of the fourth review: a 25 fps two-stage dilated TCN (MS-TCN), alone and blended with p3-refined.

File structure
--------------
- seq(): per calibration match and profile (r1 = training inputs, k = kinematic test inputs), the P3
  feature matrices with groups ball, near3, near1 (team_shape.LOCAL_GROUPS), keeping every column
  that does not end in _m25 / _m75 (75 columns), both team views, with labels and blocks, saved to
  cache/tcn/<profile>/<match>.npz. One match in memory at a time.
- DilatedTCN: stage 1 = Conv1d(F->64, 1), 7 residual blocks (dilated Conv1d(64, 64, 3, d) -> ReLU ->
  Conv1d(64, 64, 1) -> Dropout 0.2 -> add) with d = 1, 2, 4, 8, 16, 32, 64 (receptive field 255
  frames), head Conv1d(64->3, 1); stage 2 = softmax(stage 1) + the 3 scaled P1 columns,
  Conv1d(6->32, 1), 6 residual blocks (32 channels, d = 1 .. 32, same block), head Conv1d(32->3, 1).
  Variant "boundary" adds Conv1d(64->1, 1) on the stage-1 trunk (ASRF boundary head).
- loss(): cross-entropy on both stages (X and padding ignored) + 0.15 * truncated MSE on both stages
  (mean of clamp((log_softmax[t] - log_softmax[t-1].detach())^2, max 16) over adjacent real frames of
  a window); "boundary" adds 0.1 * BCE(pos_weight) on frames within +/-2 of a true label change
  (H/A/D, X ignored).
- train(): one model per held-out match, trained on the other 7 (profile r1, both views, the swapped
  view with H <-> A labels swapped). z-score per feature from those frames, then NaN -> 0. Each
  graded block is tiled into 1500-frame windows (the last one padded and masked), shuffled each epoch
  (seed 0). Adam lr 5e-4, batch 8, 20 epochs, no early stopping, torch.manual_seed(0),
  use_deterministic_algorithms(True). Each fold's model is saved, so a stopped run resumes.
- predict(): each held-out match (profile k), whole blocks, both views;
  P = 0.5 * (P_home + swap(P_swap)), saved to cache/tcn/<variant>/probs_<match>.npy.
- score(): through run.decode_evidence on the p3-local fold cache (evidence) for each variant:
  the TCN alone at alpha 0.5 / 0.75, standard decoder and the best R1 setting
  (cache/min_duration/best.json: restart, L_min, C_SWITCH); the blend
  w * log p_refined + (1 - w) * log p_tcn (each clipped at 1e-3; passed as exp(blend), so the
  decoder's log returns it exactly), w 0.3 / 0.5 / 0.7 x alpha 0.5 / 0.75, standard and best R1.
  Report: reports/tcn.md.
- final(): the held-out model, the same training on all 8 calibration matches (r1), saved to
  cache/tcn/<variant>/model_final.pt.
- holdout(): the two declared candidates on one held-out match (dfl or
  metrica3, kinematic inputs): A = final boundary TCN, alpha 0.75, R1 steady L50 cs10; B = blend
  w 0.3 of the submission's p3-refined and the final boundary TCN, alpha 0.75, standard decoder.
  p3-refined must first reproduce its logged held-out score; one line per candidate goes to
  reports/holdout_log.md.
- preview(): label-free BAR-ATH preview of A and B against the submission (p3-refined must give its
  sha256 7daa4a47...); each computed twice; counts only, no labels. Report: reports/tcn_barath_preview.md.

Declared before running. Baseline p3-refined 0.7077 / 0.6889; held-out only at
>= 0.7177 over 8 matches and >= 0.6939 on DFL.

    .venv/Scripts/python solution/external/tcn.py seq
    .venv/Scripts/python solution/external/tcn.py train plain|boundary
    .venv/Scripts/python solution/external/tcn.py predict plain|boundary
    .venv/Scripts/python solution/external/tcn.py score
    .venv/Scripts/python solution/external/tcn.py final boundary
    .venv/Scripts/python solution/external/tcn.py holdout dfl|metrica3
    .venv/Scripts/python solution/external/tcn.py preview
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common  # noqa: E402

common.FLAGS_ENABLED = False

import learned  # noqa: E402
import run  # noqa: E402
import team_shape  # noqa: E402
from common import A, CACHE_DIR, D, H, REPORT_DIR, X  # noqa: E402
from external import dropout_robustness, lomo, min_duration  # noqa: E402

# Params
TCN_DIR = os.path.join(CACHE_DIR, "tcn")
GROUPS = ("ball", "near3", "near1")
DROP_SUFFIXES = ("_m25", "_m75")
P1_NAMES = ("p1_h", "p1_a", "p1_d")
HIDDEN, DILATIONS, DROPOUT = 64, (1, 2, 4, 8, 16, 32, 64), 0.2
HIDDEN2, DILATIONS2 = 32, (1, 2, 4, 8, 16, 32)
LAMBDA_SMOOTH, SMOOTH_CLAMP = 0.15, 16.0
LAMBDA_BOUNDARY, BOUNDARY_RADIUS = 0.1, 2
LR, EPOCHS, WINDOW, BATCH, SEED = 5e-4, 20, 1500, 8, 0
ALPHAS = (0.5, 0.75)
BLEND_W = (0.3, 0.5, 0.7)
FLOOR = 1e-3
VARIANTS = ("plain", "boundary")
REPORT_PATH = os.path.join(REPORT_DIR, "tcn.md")
HOLDOUT_BASE = {"dfl": 0.6895, "metrica3": 0.7681}   # p3-refined, reports/holdout_log.md (E1a lines)
HOLDOUT_W = 0.3                                       # candidate B's blend weight
SWAP_LABEL = np.array([A, H, D, X], dtype=np.int64)


def _seq_path(profile, name):
    return os.path.join(TCN_DIR, profile, name + ".npz")


def seq():
    """Per-frame inputs of every calibration match, profiles r1 and k."""
    team_shape.LOCAL_GROUPS = GROUPS
    for profile in ("r1", "k"):
        dropout_robustness._use(profile)
        os.makedirs(os.path.join(TCN_DIR, profile), exist_ok=True)
        for name, loader in lomo.calibration_matches():
            target = _seq_path(profile, name)
            if os.path.isfile(target):
                continue
            started = time.time()
            f, blocks, labels = loader()
            ev = run.evidence(f, blocks)
            names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
            keep = [i for i, c in enumerate(names) if not c.endswith(DROP_SUFFIXES)]
            np.savez(target, x_home=x_home[:, keep], x_swap=x_swap[:, keep], labels=labels,
                     blocks=np.array(blocks, dtype=np.int64), names=np.array([names[i] for i in keep]))
            print("  seq {} {} ({} columns, {:.0f} s)".format(profile, name, len(keep), time.time() - started),
                  flush=True)
            del f, ev, x_home, x_swap


def _load_seq(profile, name):
    with np.load(_seq_path(profile, name)) as z:
        return {k: z[k] for k in z.files}


def _torch():
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    return torch


def build_model(n_in, p1_index, boundary):
    """The two-stage dilated TCN (see the module docstring)."""
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class Residual(nn.Module):
        def __init__(self, channels, dilation):
            super().__init__()
            self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.point = nn.Conv1d(channels, channels, 1)
            self.drop = nn.Dropout(DROPOUT)

        def forward(self, x):
            return x + self.drop(self.point(F.relu(self.dilated(x))))

    class Stage(nn.Module):
        def __init__(self, n_in, channels, dilations):
            super().__init__()
            self.inp = nn.Conv1d(n_in, channels, 1)
            self.blocks = nn.ModuleList([Residual(channels, d) for d in dilations])
            self.out = nn.Conv1d(channels, 3, 1)

        def forward(self, x):
            h = self.inp(x)
            for block in self.blocks:
                h = block(h)
            return h, self.out(h)

    class DilatedTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage1 = Stage(n_in, HIDDEN, DILATIONS)
            self.stage2 = Stage(3 + len(p1_index), HIDDEN2, DILATIONS2)
            self.boundary = nn.Conv1d(HIDDEN, 1, 1) if boundary else None
            self.register_buffer("p1_index", torch.tensor(p1_index, dtype=torch.long))

        def forward(self, x):
            trunk, logits1 = self.stage1(x)
            z = torch.cat((F.softmax(logits1, dim=1), x.index_select(1, self.p1_index)), dim=1)
            _, logits2 = self.stage2(z)
            bnd = self.boundary(trunk).squeeze(1) if self.boundary is not None else None
            return logits1, logits2, bnd

    return DilatedTCN()


def boundary_target(labels):
    """1 within +/-BOUNDARY_RADIUS frames of a change between two non-X labels, else 0."""
    change = np.zeros(len(labels), bool)
    change[1:] = (labels[1:] != labels[:-1]) & (labels[1:] != X) & (labels[:-1] != X)
    out = np.zeros(len(labels), np.float32)
    for t in np.flatnonzero(change):
        out[max(t - BOUNDARY_RADIUS, 0):t + BOUNDARY_RADIUS + 1] = 1.0
    return out


def _windows(x, labels, bnd, blocks):
    """Tile each block into WINDOW-frame windows; padded frames have label -100 and mask 0."""
    out = []
    for _, first, last in blocks:
        for start in range(first, last + 1, WINDOW):
            stop = min(start + WINDOW, last + 1)
            m = stop - start
            xw = np.zeros((x.shape[1], WINDOW), np.float32)
            xw[:, :m] = x[start:stop].T
            yw = np.full(WINDOW, -100, np.int64)
            yw[:m] = np.where(labels[start:stop] == X, -100, labels[start:stop])
            bw = np.zeros(WINDOW, np.float32)
            bw[:m] = bnd[start:stop]
            mw = np.zeros(WINDOW, np.float32)
            mw[:m] = 1.0
            out.append((xw, yw, bw, mw))
    return out


def _scale(x, mean, std):
    return np.nan_to_num((x - mean) / std, nan=0.0).astype(np.float32)


def loss_fn(torch, logits1, logits2, bnd, y, b, mask, pos_weight):
    """CE (both stages) + LAMBDA_SMOOTH * T-MSE (both stages) [+ LAMBDA_BOUNDARY * BCE]."""
    F = torch.nn.functional
    total = 0.0
    pair = mask[:, 1:] * mask[:, :-1]
    for logits in (logits1, logits2):
        total = total + F.cross_entropy(logits, y, ignore_index=-100)
        logp = F.log_softmax(logits, dim=1)
        diff = (logp[:, :, 1:] - logp[:, :, :-1].detach()) ** 2
        diff = torch.clamp(diff, max=SMOOTH_CLAMP).mean(1)
        total = total + LAMBDA_SMOOTH * (diff * pair).sum() / pair.sum().clamp(min=1.0)
    if bnd is not None:
        valid = (y != -100).float()
        bce = F.binary_cross_entropy_with_logits(bnd, b, pos_weight=pos_weight, reduction="none")
        total = total + LAMBDA_BOUNDARY * (bce * valid).sum() / valid.sum().clamp(min=1.0)
    return total


def train(variant):
    """One model per held-out calibration match, trained on the other 7."""
    names = [n for n, _ in lomo.calibration_matches()]
    out_dir = os.path.join(TCN_DIR, variant)
    os.makedirs(out_dir, exist_ok=True)
    for held in names:
        target = os.path.join(out_dir, "model_{}.pt".format(held))
        if os.path.isfile(target):
            continue
        _fit([n for n in names if n != held], variant, "without " + held, target)


def final(variant):
    """The held-out model: the same training on all 8 calibration matches."""
    names = [n for n, _ in lomo.calibration_matches()]
    os.makedirs(os.path.join(TCN_DIR, variant), exist_ok=True)
    target = os.path.join(TCN_DIR, variant, "model_final.pt")
    if not os.path.isfile(target):
        _fit(names, variant, "final", target)


def _fit(train_names, variant, tag, target):
    """Train one model on the r1 sequences of train_names and save it to target."""
    torch = _torch()
    started = time.time()
    data = [_load_seq("r1", n) for n in train_names]
    cols = list(data[0]["names"])
    p1_index = [cols.index(c) for c in P1_NAMES]
    stack = np.vstack([np.vstack((d["x_home"][d["labels"] != X], d["x_swap"][d["labels"] != X]))
                       for d in data])
    mean, std = np.nanmean(stack, 0, dtype=np.float64), np.nanstd(stack, 0, dtype=np.float64)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    mean = np.nan_to_num(mean, nan=0.0)
    del stack
    windows, pos, neg = [], 0.0, 0.0
    for d in data:
        labels = d["labels"].astype(np.int64)
        blocks = [tuple(int(v) for v in b) for b in d["blocks"]]
        bnd = boundary_target(labels)
        valid = labels != X
        pos += float(bnd[valid].sum()) * 2
        neg += float((1 - bnd[valid]).sum()) * 2
        windows += _windows(_scale(d["x_home"], mean, std), labels, bnd, blocks)
        windows += _windows(_scale(d["x_swap"], mean, std), SWAP_LABEL[labels], bnd, blocks)
    del data
    torch.manual_seed(SEED)
    model = build_model(len(cols), p1_index, variant == "boundary")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    log = []
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(windows))
        total, steps = 0.0, 0
        for i in range(0, len(order), BATCH):
            batch = [windows[j] for j in order[i:i + BATCH]]
            x = torch.from_numpy(np.stack([w[0] for w in batch]))
            y = torch.from_numpy(np.stack([w[1] for w in batch]))
            b = torch.from_numpy(np.stack([w[2] for w in batch]))
            mask = torch.from_numpy(np.stack([w[3] for w in batch]))
            if not bool((y != -100).any()):
                continue
            logits1, logits2, bnd = model(x)
            loss = loss_fn(torch, logits1, logits2, bnd, y, b, mask, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        log.append(total / steps)
        print("  {} {}: epoch {} loss {:.4f} ({:.0f} s)".format(
            variant, tag, epoch + 1, log[-1], time.time() - started), flush=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "cols": cols,
                "p1_index": p1_index, "variant": variant, "loss": log,
                "pos_weight": float(pos_weight), "seconds": time.time() - started}, target)


def _probs(torch, model, x, blocks, n):
    """(n, 3) softmax of stage 2 inside the blocks; 1/3 elsewhere (never decoded)."""
    out = np.full((n, 3), 1.0 / 3.0)
    model.eval()
    with torch.no_grad():
        for _, first, last in blocks:
            xb = torch.from_numpy(np.ascontiguousarray(x[first:last + 1].T)[None])
            _, logits2, _ = model(xb)
            out[first:last + 1] = torch.softmax(logits2, dim=1)[0].T.numpy()
    return out


def predict(variant):
    torch = _torch()
    out_dir = os.path.join(TCN_DIR, variant)
    for name, _ in lomo.calibration_matches():
        target = os.path.join(out_dir, "probs_{}.npy".format(name))
        if os.path.isfile(target):
            continue
        saved = torch.load(os.path.join(out_dir, "model_{}.pt".format(name)), weights_only=False)
        model = build_model(len(saved["cols"]), saved["p1_index"], saved["variant"] == "boundary")
        model.load_state_dict(saved["state"])
        d = _load_seq("k", name)
        if list(d["names"]) != saved["cols"]:
            raise SystemExit("columns differ between training and test sequences")
        blocks = [tuple(int(v) for v in b) for b in d["blocks"]]
        n = len(d["labels"])
        p_home = _probs(torch, model, _scale(d["x_home"], saved["mean"], saved["std"]), blocks, n)
        p_swap = _probs(torch, model, _scale(d["x_swap"], saved["mean"], saved["std"]), blocks, n)
        p = np.empty_like(p_home)
        p[:, 0] = 0.5 * (p_home[:, 0] + p_swap[:, 1])
        p[:, 1] = 0.5 * (p_home[:, 1] + p_swap[:, 0])
        p[:, 2] = 0.5 * (p_home[:, 2] + p_swap[:, 2])
        # symmetry check: the views exchanged give the team-swapped probabilities
        q = np.empty_like(p)
        q[:, 0] = 0.5 * (p_swap[:, 0] + p_home[:, 1])
        q[:, 1] = 0.5 * (p_swap[:, 1] + p_home[:, 0])
        q[:, 2] = 0.5 * (p_swap[:, 2] + p_home[:, 2])
        assert np.allclose(q[:, [1, 0, 2]], p)
        np.save(target, p.astype(np.float32))
        print("  probs {} {}".format(variant, name), flush=True)


def holdout(which):
    """The two declared candidates on one held-out match, kinematic test inputs.

    Candidate A: the final boundary TCN alone, alpha 0.75, R1 (steady, L_min 50, C_SWITCH 10).
    Candidate B: blend 0.3 * log p_refined + 0.7 * log p_tcn (final boundary TCN), alpha 0.75,
    standard decoder (C_SWITCH 8). p_refined is the submission's P3 + Stage 2 (models/), which must
    first reproduce its logged held-out score. One line per candidate goes to reports/holdout_log.md.
    """
    import possession
    import possession_metrics as pm
    import stage2
    from external import holdout as ho
    from external import review_ablation
    torch = _torch()
    dropout_robustness._use("k")
    base_groups = team_shape.LOCAL_GROUPS
    f, blocks, truth = ho.HOLDOUTS[which]()
    ev = run.evidence(f, blocks)
    # p3-refined, exactly as the submission computes it
    bundle = learned.load()
    names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    if names != bundle["names"]:
        raise SystemExit("P3 feature set differs from the saved model")
    refiner = stage2.load()
    if refiner["stage1_names"] != bundle["names"]:
        raise SystemExit("Stage-2 model trained on another P3 feature set")
    refined = stage2.refine(refiner, f, blocks, ev, learned.predict_proba(bundle["model"], x_home, x_swap))
    base = pm.evaluate(truth, run.decode_evidence(ev, blocks, refined, refiner["alpha"]))
    if round(base["spell_f1_mean"], 4) != HOLDOUT_BASE[which]:
        raise SystemExit("p3-refined held-out {} not reproduced: {:.4f}".format(which, base["spell_f1_mean"]))
    del x_home, x_swap
    # the final boundary TCN
    team_shape.LOCAL_GROUPS = GROUPS
    try:
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    finally:
        team_shape.LOCAL_GROUPS = base_groups
    keep = [i for i, c in enumerate(names) if not c.endswith(DROP_SUFFIXES)]
    saved = torch.load(os.path.join(TCN_DIR, "boundary", "model_final.pt"), weights_only=False)
    if [names[i] for i in keep] != saved["cols"]:
        raise SystemExit("held-out columns differ from the final model's")
    model = build_model(len(saved["cols"]), saved["p1_index"], True)
    model.load_state_dict(saved["state"])
    n = len(truth)
    p_home = _probs(torch, model, _scale(x_home[:, keep], saved["mean"], saved["std"]), blocks, n)
    p_swap = _probs(torch, model, _scale(x_swap[:, keep], saved["mean"], saved["std"]), blocks, n)
    p_tcn = np.empty_like(p_home)
    p_tcn[:, 0] = 0.5 * (p_home[:, 0] + p_swap[:, 1])
    p_tcn[:, 1] = 0.5 * (p_home[:, 1] + p_swap[:, 0])
    p_tcn[:, 2] = 0.5 * (p_home[:, 2] + p_swap[:, 2])
    blend = np.exp(HOLDOUT_W * np.log(np.clip(refined, FLOOR, 1.0))
                   + (1 - HOLDOUT_W) * np.log(np.clip(p_tcn, FLOOR, 1.0)))
    candidates = [
        ("R5 TCN boundary (final, 8 matches), alpha 0.75, R1 steady L50 cs10, kinematic test",
         p_tcn, 0.75, 10.0, 50, "steady"),
        ("R5 blend w 0.3 p3-refined + TCN boundary (final, 8 matches), alpha 0.75, standard decoder, kinematic test",
         blend, 0.75, 8.0, None, None),
    ]
    out = {"which": which, "p3_refined": {k: base[k] for k in ("spell_f1_mean", "macro_f1",
                                                                "n_pred_spells", "n_gt_spells")}}
    fold = {"name": which, "f": f, "blocks": blocks}
    lines = []
    for note, probs, alpha, c_switch, l_min, restart in candidates:
        saved_md = possession.MIN_DURATION, possession.MIN_RESTART
        possession.MIN_DURATION, possession.MIN_RESTART = l_min, restart
        try:
            labels = review_ablation.decode_variant(dict(fold, probs=probs), {"alpha": alpha, "c_switch": c_switch}, {})
        finally:
            possession.MIN_DURATION, possession.MIN_RESTART = saved_md
        result = pm.evaluate(truth, labels)
        tagged = note if which == "metrica3" else "[{}] {}".format(which, note)
        lines.append("| {} | {} | {} | {:.4f} | {:.4f} | {}/{} |".format(
            time.strftime("%Y-%m-%d"), ho.commit_id(), tagged, result["spell_f1_mean"], result["macro_f1"],
            result["n_pred_spells"], result["n_gt_spells"]))
        out[note] = {k: result[k] for k in ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells")}
    with open(ho.LOG_PATH, "a", encoding="utf-8") as handle:   # both lines, after both are scored
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("  p3-refined reproduced: {:.4f}".format(base["spell_f1_mean"]), flush=True)
    with open(os.path.join(TCN_DIR, "holdout_{}.json".format(which)), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)


SUBMISSION_SHA = "7daa4a47bc5883b6"   # p3-refined BAR-ATH CSV (README, tag p3-refined)
PREVIEW_PATH = os.path.join(REPORT_DIR, "tcn_barath_preview.md")


def _barath_labels(torch):
    """BAR-ATH labels of p3-refined, A and B, computed as run.predict computes the submission. No labels used."""
    import atd_cache
    import dead_ball
    import features
    import possession
    import stage2
    from common import PITCH_LENGTH, half_masks
    f = features.build()
    blocks, graded = half_masks()
    ev = run.evidence(f, blocks, dead_ball.tagged_kickoffs(f), dead_ball.set_piece_windows(len(f["visible"])))
    f = dict(f, players=atd_cache.load()["players"], pitch_length=PITCH_LENGTH)
    bundle = learned.load()
    names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    if names != bundle["names"]:
        raise SystemExit("P3 feature set differs from the saved model")
    refiner = stage2.load()
    refined = stage2.refine(refiner, f, blocks, ev, learned.predict_proba(bundle["model"], x_home, x_swap))
    del x_home, x_swap
    base_groups = team_shape.LOCAL_GROUPS
    team_shape.LOCAL_GROUPS = GROUPS
    try:
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    finally:
        team_shape.LOCAL_GROUPS = base_groups
    keep = [i for i, c in enumerate(names) if not c.endswith(DROP_SUFFIXES)]
    saved = torch.load(os.path.join(TCN_DIR, "boundary", "model_final.pt"), weights_only=False)
    if [names[i] for i in keep] != saved["cols"]:
        raise SystemExit("BAR-ATH columns differ from the final model's")
    model = build_model(len(saved["cols"]), saved["p1_index"], True)
    model.load_state_dict(saved["state"])
    n = len(f["visible"])
    p_home = _probs(torch, model, _scale(x_home[:, keep], saved["mean"], saved["std"]), blocks, n)
    p_swap = _probs(torch, model, _scale(x_swap[:, keep], saved["mean"], saved["std"]), blocks, n)
    del x_home, x_swap
    p_tcn = np.empty_like(p_home)
    p_tcn[:, 0] = 0.5 * (p_home[:, 0] + p_swap[:, 1])
    p_tcn[:, 1] = 0.5 * (p_home[:, 1] + p_swap[:, 0])
    p_tcn[:, 2] = 0.5 * (p_home[:, 2] + p_swap[:, 2])
    blend = np.exp(HOLDOUT_W * np.log(np.clip(refined, FLOOR, 1.0))
                   + (1 - HOLDOUT_W) * np.log(np.clip(p_tcn, FLOOR, 1.0)))
    out = {}
    for key, probs, alpha, c_switch, l_min, restart in (
            ("p3-refined", refined, refiner["alpha"], possession.C_SWITCH, None, "steady"),
            ("A", p_tcn, 0.75, 10.0, 50, "steady"),
            ("B", blend, 0.75, 8.0, None, "steady")):
        saved_md = possession.C_SWITCH, possession.MIN_DURATION, possession.MIN_RESTART
        possession.C_SWITCH, possession.MIN_DURATION, possession.MIN_RESTART = c_switch, l_min, restart
        try:
            out[key] = run.decode_evidence(ev, blocks, probs, alpha)
        finally:
            possession.C_SWITCH, possession.MIN_DURATION, possession.MIN_RESTART = saved_md
    return out, graded


def _csv_sha(labels, path):
    import hashlib
    run.write_csv(path, labels)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def _rates(labels, blocks):
    """Per graded hour: possessions, stoppages; % of possessions under 2 s; % dead frames; graded minutes."""
    from common import FPS, runs_of
    pos = stop = short = frames = dead = 0
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        starts, stops, values = runs_of(seg)
        alive = (values == H) | (values == A)
        pos += int(alive.sum())
        stop += int((values == D).sum())
        short += int((alive & ((stops - starts) / FPS < 2.0)).sum())
        graded = seg != X
        frames += int(graded.sum())
        dead += int((seg[graded] == D).sum())
    hours = frames / FPS / 3600
    return pos / hours, stop / hours, 100.0 * short / pos, 100.0 * dead / frames, frames / FPS / 60


def preview():
    """Label-free BAR-ATH preview of candidates A and B against the submission (p3-refined).

    Each candidate is computed twice (determinism). p3-refined must first give the submitted file's
    sha256. CSVs go to cache/tcn/barath_<candidate>.csv (gitignored), never to evaluation/results.
    Counts: possessions, durations, stoppages, shares, possessions shorter than 2 s, frames changed
    against the submission and between A and B, on the graded frames. Report: reports/tcn_barath_preview.md.
    """
    from common import FPS, runs_of
    from external import holdout as ho
    torch = _torch()
    started = time.time()
    first, graded = _barath_labels(torch)
    second, _ = _barath_labels(torch)
    sha = {k: _csv_sha(v, os.path.join(TCN_DIR, "barath_{}.csv".format(k))) for k, v in first.items()}
    if sha["p3-refined"] != SUBMISSION_SHA:
        raise SystemExit("p3-refined BAR-ATH sha {} is not the submission's {}".format(sha["p3-refined"], SUBMISSION_SHA))
    sub = first["p3-refined"]
    lines = ["# BAR-ATH preview of TCN candidates A and B", "",
             "No labels. Frames changed are on the graded frames, against the submission (p3-refined).",
             "Candidates as declared before the run. Commit {}.".format(ho.commit_id()), "",
             "| candidate | sha256 | deterministic | frames changed | possessions < 2 s | summary |",
             "|---|---|---|---|---|---|"]
    for key, labels in first.items():
        starts, stops, values = runs_of(np.where(graded, labels, X))
        alive = (values == H) | (values == A)
        short = int((alive & ((stops - starts) / FPS < 2.0)).sum())
        lines.append("| {} | `{}` | {} | {:.2%} | {} | {} |".format(
            key, sha[key], "yes" if np.array_equal(labels, second[key]) else "NO",
            np.mean(labels[graded] != sub[graded]), short, run.summarise(labels, graded)))
    lines += ["", "A vs B: {:.2%} of graded frames differ.".format(np.mean(first["A"][graded] != first["B"][graded])),
              "", "## Rates against the true labels of the 8 calibration matches (kinematic profile)", "",
              "Calibration rows are TRUE labels (external data); BAR-ATH rows are predictions.", "",
              "| match | possessions / h | stoppages / h | % possessions < 2 s | % dead | graded min |",
              "|---|---|---|---|---|---|"]
    from common import half_masks
    blocks, _ = half_masks()
    for name, _ in lomo.calibration_matches():
        z = np.load(_seq_path("k", name), allow_pickle=True)
        lines.append("| {} | {:.0f} | {:.0f} | {:.1f} | {:.1f} | {:.0f} |".format(
            name, *_rates(z["labels"], [tuple(b) for b in z["blocks"]])))
        del z
    for key, labels in first.items():
        lines.append("| BAR-ATH {} | {:.0f} | {:.0f} | {:.1f} | {:.1f} | {:.0f} |".format(
            key, *_rates(np.where(graded, labels, X), blocks)))
    lines += ["", "Run time {:.0f} s.".format(time.time() - started)]
    with open(PREVIEW_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def score():
    with open(os.path.join(min_duration.OUT_DIR, "best.json"), encoding="utf-8") as handle:
        best = json.load(handle)
    r1 = (best["restart"], int(best["l_min"]), float(best["c_switch"]))
    folds = min_duration.load_folds()
    names = [f["name"] for f in folds]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    rows = {}
    cache = {}
    base = min_duration.run_setting(folds, "refined", 0.75, 8.0, None, None, cache)
    sb = np.array([r["spell"] for r in base])
    if (round(sb.mean(), 4), round(sb[dfl].mean(), 4)) != min_duration.BASELINE:
        raise SystemExit("p3-refined base not reproduced")
    rows["p3-refined (standard, alpha 0.75)"] = base
    for variant in VARIANTS:
        for f in folds:
            p = np.load(os.path.join(TCN_DIR, variant, "probs_{}.npy".format(f["name"]))).astype(np.float64)
            f["probs_tcn"] = p
            for w in BLEND_W:
                blend = w * np.log(np.clip(f["probs_refined"], FLOOR, 1.0)) + (1 - w) * np.log(np.clip(p, FLOOR, 1.0))
                f["probs_blend{}".format(w)] = np.exp(blend)
        sets = [("tcn", "TCN {}".format(variant))] + [
            ("blend{}".format(w), "blend w {} ({})".format(w, variant)) for w in BLEND_W]
        for probs, label in sets:
            for alpha in ALPHAS:
                rows["{}, alpha {}, standard".format(label, alpha)] = min_duration.run_setting(
                    folds, probs, alpha, 8.0, None, None, cache)
                rows["{}, alpha {}, R1 {} L{} cs{}".format(label, alpha, *r1)] = min_duration.run_setting(
                    folds, probs, alpha, r1[2], r1[1], r1[0], cache)
                print("  scored {} alpha {}".format(label, alpha), flush=True)
    head = ("| setting | spellF1 (8) | vs p3-refined | spellF1 (DFL) | vs p3-refined | better | predicted | "
            "short | split_dead | split_team | merged | edge s/e (s) | per match |")
    lines = ["# R5: dilated TCN, alone and blended with p3-refined", "",
             "LOMO, trained r1, tested kinematic. Best R1 setting from reports/min_duration.md: restart {}, "
             "L_min {}, C_SWITCH {}. Bar {} / {}.".format(*r1, *min_duration.BAR), "",
             head, "|" + "---|" * (head.count("|") - 1)]
    for k, rs in rows.items():
        s = np.array([r["spell"] for r in rs])
        lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {}/{} | {} | {:.2f} / {:.2f} | {} |".format(
            k, s.mean(), s.mean() - sb.mean(), s[dfl].mean(), s[dfl].mean() - sb[dfl].mean(),
            int(np.sum(s > sb + 1e-9)), sum(r["n_pred"] for r in rs), sum(r["n_gt"] for r in rs),
            " | ".join(str(sum(r["cats"][c] for r in rs)) for c in min_duration.CATS),
            np.mean([r["start"] for r in rs]), np.mean([r["end"] for r in rs]),
            " ".join("{:.4f}".format(v) for v in s)))
    for variant in VARIANTS:
        secs = [torch_meta(variant, n) for n in names]
        lines.append("\n{}: training {:.0f} s per fold on average; final-epoch loss {}.".format(
            variant, np.mean([s[0] for s in secs]), " ".join("{:.4f}".format(s[1]) for s in secs)))
    lines.append("\nMatches: " + ", ".join(names))
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(TCN_DIR, "score.json"), "w", encoding="utf-8") as handle:
        json.dump({"names": names, "rows": rows}, handle)
    print("\n".join(lines))


def torch_meta(variant, name):
    torch = _torch()
    saved = torch.load(os.path.join(TCN_DIR, variant, "model_{}.pt".format(name)), weights_only=False)
    return saved["seconds"], saved["loss"][-1]


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase in ("train", "predict", "final", "holdout"):
        {"train": train, "predict": predict, "final": final, "holdout": holdout}[phase](sys.argv[2])
    else:
        {"seq": seq, "score": score, "preview": preview}[phase]()
