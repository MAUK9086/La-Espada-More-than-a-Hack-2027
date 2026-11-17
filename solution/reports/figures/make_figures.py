"""Figures of the technical report, built from committed reports and external matches only.

File structure
--------------
- STYLE, COLOURS: one print style matching the IEEEtran report (serif, 7-8 pt, column / text width).
- table_rows(): parse the rows of a Markdown table in a committed report.
- fig_tracking(): fig_missing_ball (report Fig. 1), BAR-ATH's missing ball: share of missing frames by
  gap length (reports/diagnostics.md) and the missing share per player-speed band that the training
  degradation reproduces (external/sportec_dfl.BAND_EDGES / BAND_TARGETS). Label-free aggregates only.
- fig_per_match(): fig_per_match (report Fig. 5), leave-one-match-out spellF1 per external match,
  previous version against the submission (reports/tcn_gpu_cos_ema.md, reports/tcn_gpu_seeds.md).
- fig_timeline(): fig_timeline (report Fig. 4), the first three minutes of the second half of the
  held-out DFL match J03WR9: true and predicted labels and the frames without a ball. Uses the committed
  models, the kinematic test degradation and the submission's decoder; checks the match score against
  the held-out log (0.7885) before drawing. No frame of the provided match is plotted.
- main(): writes vector PDF (for the LaTeX report) and PNG files to docs/report/figures/.
  The pipeline (report Fig. 2) and the TCN architecture (Fig. 3) are drawn in TikZ in docs/report/La_Espada.tex.

Needs matplotlib (not needed for the submission) and, for fig_timeline, the DFL data and lxml.

    python solution/reports/figures/make_figures.py
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(SOLUTION)
sys.path[:0] = [SOLUTION, os.path.join(SOLUTION, "external"), os.path.join(ROOT, "starter")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Params
OUT_DIR = os.path.join(ROOT, "docs", "report", "figures")
REPORTS = os.path.join(SOLUTION, "reports")
COLOURS = {"H": "#2C6E9E", "A": "#E0A030", "D": "#C8C8C8", "ink": "#000000", "muted": "#555555",
           "prev": "#9A9A9A", "new": "#2C6E9E", "accent": "#2C6E9E"}
COLUMN_IN = 3.45            # IEEEtran column width (inches)
PAGE_IN = 7.0               # IEEEtran text width (inches)
STYLE = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42}
MATCHES = ("metrica1", "metrica2", "J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY")
HELDOUT_SCORE = 0.7885      # reports/holdout_log.md, TCN k4_e10 ensemble on DFL J03WR9
TIMELINE_S = 180            # Figure 4: seconds shown


def table_rows(report, first_cell):
    """Cells of every Markdown table row in `report` whose first cell starts with `first_cell`.

    Args:
        report: file name in solution/reports/.
        first_cell: prefix of the first cell.

    Returns:
        list: one list of stripped cell strings per matching row.
    """
    rows = []
    with open(os.path.join(REPORTS, report), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if cells[0].startswith(first_cell):
                    rows.append(cells)
    if not rows:
        raise ValueError("no row '{}' in {}".format(first_cell, report))
    return rows


def _save(fig, name):
    """Write one figure as PNG and SVG."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, "{}.{}".format(name, ext)), dpi=300)
    plt.close(fig)
    print("wrote", name, flush=True)


def fig_tracking():
    """Figure 2: where the ball is missing in the provided tracking (label-free aggregates)."""
    from external import sportec_dfl
    rows = [r for r in table_rows("diagnostics.md", "") if re.match(r"^(\d+-\d+|>= \d+)$", r[0])]
    shares = [float(r[2].rstrip("%")) for r in rows]
    starts = [int(r[0].split("-")[0].replace(">= ", "")) for r in rows]
    keep = [i for i, lo in enumerate(starts) if lo >= 50]          # gaps of 2 s and longer
    labels = ["<2"] + ["2–5", "5–10", "10–30", "≥30"]
    values = [sum(v for i, v in enumerate(shares) if i not in keep)] + [shares[i] for i in keep]
    assert len(values) == len(labels)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.38), gridspec_kw={"width_ratios": [1.35, 1]})
    a.bar(range(len(values)), values, color=COLOURS["new"], width=0.62)
    a.set_xticks(range(len(values)), labels)
    a.set_xlabel("gap length (s)")
    a.set_ylabel("missing frames (%)")
    a.set_ylim(0, 52)
    for i, v in enumerate(values):
        a.text(i, v + 1.2, "{:.1f}".format(v), ha="center", fontsize=6)
    a.set_title("(a)", loc="left", fontsize=8)
    edges = sportec_dfl.BAND_EDGES
    targets = [100 * t for t in sportec_dfl.BAND_TARGETS]
    b.bar(range(3), targets, color=COLOURS["new"], width=0.55)
    b.set_xticks(range(3), ["<{:g}".format(edges[0]), "{:g}–{:g}".format(*edges), ">{:g}".format(edges[1])])
    b.set_xlabel("mean player speed (m/s)")
    b.set_ylabel("ball missing (%)")
    b.set_ylim(0, 85)
    for i, t in enumerate(targets):
        b.text(i, t + 2, "{:.1f}".format(t), ha="center", fontsize=6)
    b.set_title("(b)", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig_missing_ball")


def fig_per_match():
    """Figure 3: per-match LOMO spellF1, previous version against the submission."""
    prev = [float(v) for v in table_rows("tcn_gpu_seeds.md", "p3-refined (base)")[0][-1].split()]
    new = [float(v) for v in table_rows("tcn_gpu_cos_ema.md", "k4_e10, std, ensemble of 3")[0][-1].split()]
    assert round(float(np.mean(new)), 4) == 0.7590 and round(float(np.mean(prev)), 4) == 0.7077
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.72))
    x = np.arange(len(MATCHES))
    for i in x:
        ax.plot([i, i], [prev[i], new[i]], color=COLOURS["muted"], lw=0.6, zorder=1)
    ax.scatter(x, prev, color=COLOURS["prev"], s=10, zorder=2, label="previous version (gradient boosting)")
    ax.scatter(x, new, color=COLOURS["new"], s=10, zorder=3, label="submission (TCN ensemble)")
    for i in x:
        ax.text(i, new[i] + 0.011, "+{:.3f}".format(new[i] - prev[i]), fontsize=5.5, ha="center")
    ax.set_xticks(x, ["Met. 1", "Met. 2"] + list(MATCHES[2:]), rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("spellF1 (LOMO)")
    ax.set_ylim(0.60, 0.83)
    ax.set_xlim(-0.5, len(MATCHES) - 0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, handletextpad=0.2, columnspacing=1.0)
    ax.grid(axis="y", color="#E3E3E3", lw=0.4)
    ax.set_axisbelow(True)
    _save(fig, "fig_per_match")


def fig_timeline():
    """Figure 4: true and predicted labels, held-out DFL J03WR9, first 3 minutes of the second half."""
    import possession_metrics as pm
    import run
    import tcn_infer
    from common import A, D, FPS, H, X
    from external import dropout_robustness, holdout as ho
    dropout_robustness._use("k")
    f, blocks, truth = ho.HOLDOUTS["dfl"]()
    ev = run.evidence(f, blocks)
    probs, alpha = tcn_infer.match_probs(f, blocks, ev)
    pred = run.decode_evidence(ev, blocks, probs, alpha)
    score = pm.evaluate(truth, pred)["spell_f1_mean"]
    print("J03WR9 spellF1 {:.4f} (held-out log {:.4f})".format(score, HELDOUT_SCORE), flush=True)
    assert abs(score - HELDOUT_SCORE) < 0.0005, "the held-out score does not reproduce"
    half, first, last = blocks[1]
    start = first + int(np.argmax(truth[first:last + 1] != D))       # second half's first live frame
    span = slice(start, start + TIMELINE_S * FPS)
    seen = f["visible"][span]
    fig, ax = plt.subplots(figsize=(PAGE_IN, 0.9))
    t = np.arange(span.stop - span.start) / FPS
    for row, labels in ((1.0, truth[span]), (0.0, pred[span])):
        for code, key in ((H, "H"), (A, "A"), (D, "D")):
            on = labels == code
            ax.fill_between(t, row - 0.32, row + 0.32, where=on, step="post", color=COLOURS[key], lw=0)
    ax.fill_between(t, -0.62, -0.5, where=~seen, step="post", color=COLOURS["ink"], lw=0)
    ax.set_yticks([1.0, 0.0, -0.56], ["truth", "prediction", "no ball"])
    ax.set_ylim(-0.75, 1.4)
    ax.set_xlim(0, TIMELINE_S)
    ax.set_xlabel("seconds from the second half's kick-off")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[k]) for k in ("H", "A", "D")]
    ax.legend(handles, ["home possession", "away possession", "dead ball"], loc="upper center",
              bbox_to_anchor=(0.5, 1.42), ncol=3, handlelength=1.2)
    _save(fig, "fig_timeline")
    return score


def main():
    """CLI entry point."""
    plt.rcParams.update(STYLE)
    fig_tracking()
    fig_per_match()
    fig_timeline()


if __name__ == "__main__":
    main()
