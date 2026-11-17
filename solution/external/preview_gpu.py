"""Label-free BAR-ATH preview of GPU TCN candidates against the submission (p3-refined).

File structure
--------------
- Params: the systems previewed.
- _inputs(): BAR-ATH evidence, blocks, the submission's refined probabilities and the TCN input views,
  built as run.py builds them. No labels are read.
- _labels(): decoded labels of p3-refined and of each candidate (torch ensemble, standard decoder),
  plus the NumPy float64 path (tcn_infer) for the candidates that have exported models.
- main(): two passes (determinism), rates, frames that differ from the submission split into
  alive/dead and team, differences between candidates. Report: reports/tcn_gpu_barath_preview.md.

Label-free: nothing is fitted or tuned; the public window is not
scored; CSVs go to cache/tcn_gpu/preview/ (gitignored).

Usage: .venv/Scripts/python solution/external/preview_gpu.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from external import tcn_gpu as tg  # noqa: E402
from external import tcn as tcn_cpu  # noqa: E402
import learned  # noqa: E402
import possession  # noqa: E402
import run  # noqa: E402
import tcn_infer  # noqa: E402
import team_shape  # noqa: E402
from common import A, D, H, REPORT_DIR, X  # noqa: E402

# Params
SYSTEMS = (("K2", "k4_e10", (0, 1, 2)), ("K1.5", "k4_e10_c1.5", (0, 1, 2)), ("aug4_e10 (r1)", "aug4_e10", (0, 1, 2)))
NUMPY = ("K2", "K1.5")            # candidates with exported models in cache/tcn_gpu/<config>/export
OUT = os.path.join(tg.OUT_DIR, "preview")
REPORT = os.path.join(REPORT_DIR, "tcn_gpu_barath_preview.md")


def _inputs():
    """BAR-ATH inputs. Returns (f, blocks, graded, ev, refined, alpha, d) with d the TCN sequence dict."""
    import atd_cache
    import dead_ball
    import features
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
    base_groups = team_shape.LOCAL_GROUPS
    team_shape.LOCAL_GROUPS = tg.GROUPS
    try:
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    finally:
        team_shape.LOCAL_GROUPS = base_groups
    keep = [i for i, c in enumerate(names) if not c.endswith(tg.DROP_SUFFIXES)]
    d = {"x_home": x_home[:, keep], "x_swap": x_swap[:, keep], "names": np.array([names[i] for i in keep]),
         "blocks": np.array(blocks, dtype=np.int64)}
    return f, blocks, graded, ev, refined, refiner["alpha"], d


def _decode(ev, blocks, probs, alpha, c_switch):
    saved = possession.C_SWITCH, possession.MIN_DURATION, possession.MIN_RESTART
    possession.C_SWITCH, possession.MIN_DURATION, possession.MIN_RESTART = c_switch, None, "steady"
    try:
        return run.decode_evidence(ev, blocks, probs, alpha)
    finally:
        possession.C_SWITCH, possession.MIN_DURATION, possession.MIN_RESTART = saved


def _labels():
    """{system: labels} for one full pass, and the graded mask and blocks."""
    torch = tg._torch()
    f, blocks, graded, ev, refined, alpha, d = _inputs()
    out = {"p3-refined": _decode(ev, blocks, refined, alpha, possession.C_SWITCH)}
    for label, config, seeds in SYSTEMS:
        models = [torch.load(os.path.join(tg._seed_dir(config, s), "model_final.pt"), weights_only=False) for s in seeds]
        out[label] = _decode(ev, blocks, np.mean([tg.predict(m, d) for m in models], axis=0), 0.75, 8.0)
        if label in NUMPY:
            exported = [tcn_infer.load(os.path.join(tg.OUT_DIR, config, "export", "tcn_s{}.npz".format(s))) for s in seeds]
            if exported[0]["cols"] != list(d["names"]):
                raise SystemExit("BAR-ATH columns differ from the exported model's")
            probs = tcn_infer.predict_ensemble(exported, d["x_home"], d["x_swap"], blocks)
            out[label + " (NumPy)"] = _decode(ev, blocks, probs, 0.75, 8.0)
    return out, graded, blocks


def main():
    if os.path.isfile(REPORT):
        raise SystemExit("{} exists; reports are never overwritten".format(REPORT))
    from external import holdout as ho
    os.makedirs(OUT, exist_ok=True)
    started = time.time()
    first, graded, blocks = _labels()
    second, _, _ = _labels()
    sha = {k: tcn_cpu._csv_sha(v, os.path.join(OUT, "barath_{}.csv".format(k.replace(" ", "_").replace("(", "").replace(")", ""))))
           for k, v in first.items()}
    if sha["p3-refined"] != tcn_cpu.SUBMISSION_SHA:
        raise SystemExit("p3-refined BAR-ATH sha {} is not the submission's".format(sha["p3-refined"]))
    sub = first["p3-refined"][graded]
    live_s = sub != D
    lines = ["# BAR-ATH preview of GPU TCN candidates (label-free)", "",
             "No labels are read and the public window is not scored. K2 = `k4_e10`, K1.5 = `k4_e10_c1.5`, 3-seed "
             "ensembles of final models, standard decoder α 0.75; `aug4_e10 (r1)` is the "
             "`r1`-trained ensemble, for comparison only. \"(NumPy)\" is the float64 inference of `solution/tcn_infer.py` "
             "on the same models. Shares are of the graded frames, against the submission (p3-refined). "
             "Commit {}.".format(ho.commit_id()), "",
             "| system | sha256 | deterministic | possessions / h | stoppages / h | % possessions < 2 s | % dead | "
             "frames changed | submission live → dead | submission dead → live | both live, other team |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, labels in first.items():
        lab = labels[graded]
        live = lab != D
        rates = tcn_cpu._rates(np.where(graded, labels, X), blocks)
        lines.append("| {} | `{}` | {} | {:.0f} | {:.0f} | {:.1f} | {:.1f} | {:.2%} | {:.2%} | {:.2%} | {:.2%} |".format(
            key, sha[key], "yes" if np.array_equal(labels, second[key]) else "NO", rates[0], rates[1], rates[2], rates[3],
            np.mean(lab != sub), np.mean(live_s & ~live), np.mean(~live_s & live), np.mean(live_s & live & (lab != sub))))
    lines += ["", "## Between systems (share of graded frames that differ)", ""]
    keys = list(first)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a == "p3-refined":
                continue
            la, lb = first[a][graded], first[b][graded]
            lines.append("- {} vs {}: {:.2%} (alive/dead {:.2%}, team {:.2%})".format(
                a, b, np.mean(la != lb), np.mean((la == D) != (lb == D)), np.mean((la != D) & (lb != D) & (la != lb))))
    lines += ["", "Run time {:.0f} s.".format(time.time() - started)]
    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
