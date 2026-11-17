# Team La Espada — frame-level possession, FC Barcelona v Athletic Club (More than a Hack 2027)

This is team La Espada's submission to [More than a Hack 2027](CHALLENGE_README.md), the Metrica Sports × FC
Barcelona challenge. It labels every one of the 199,319 frames of the provided match `H`, `A`, `D` or `X`.

**Method in one line:** rule-based evidence from the tracking, plus a 3-model ensemble of dilated temporal
convolutional networks (TCN) trained only on external matches, decoded by an event-gated Viterbi.

**Technical document:** [docs/report/La_Espada.pdf](docs/report/La_Espada.pdf), typeset from
[docs/report/La_Espada.tex](docs/report/La_Espada.tex) with the IEEE conference template (`latexmk -pdf La_Espada.tex`
in `docs/report/`). Its plots are drawn by `solution/reports/figures/make_figures.py` (needs matplotlib). The
organisers' documentation is unchanged:
[CHALLENGE_README.md](CHALLENGE_README.md) (their README), [CHALLENGE.md](CHALLENGE.md), [RULES.md](RULES.md),
[SUBMISSION.md](SUBMISSION.md), [EVALUATION.md](EVALUATION.md), [DATA.md](DATA.md), [FAQ.md](FAQ.md).

| | spellF1 |
|---|---|
| leave-one-match-out over 8 external matches (mean / mean of the 6 DFL matches) | 0.7590 / 0.7460 |
| held-out Sportec DFL J03WR9 / Metrica sample game 3 | 0.7885 / 0.8143 |

## Regenerate the submission

1. **Python 3.13** (tested with 3.13.11) and the pinned packages:

   ```
   pip install -r solution/requirements.txt
   ```

   This installs numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 and joblib 1.6.0. No GPU or deep-learning library is
   needed.

2. **Data.** Unpack the organisers' gated bundle so that `data/tactical/` exists (see [DATA.md](DATA.md)). The code
   also finds it in `../data` next to the repository, or at the path in `MTAH_DATA_DIR`. The data is not in this
   repository.

3. **Run**, from the repository root:

   ```
   python solution/run.py
   ```

   This writes `evaluation/results/La_Espada.csv` and validates it with `evaluation/validate_submission.py`. The
   file's SHA-256 is

   ```
   8286097e28c9e9ab6cb843413daf669d8b6e919d22d67a86c828e5e856b4888c
   ```

   The script prints its first 16 characters. The first run parses the tracking into `solution/cache/`
   (gitignored). After that a run takes about 30 s on the workstation we trained on, and about 100 s on a laptop
   with an Intel Core i5-1235U. Inference is NumPy float64 on the committed weights
   (`solution/models/tcn_k2_s{0,1,2}.npz`), and the ensemble's probabilities are rounded to 6 decimals before
   decoding, so the same data always gives the same file.

That one command is all the submission needs.

## Earlier versions (optional)

These reproduce the comparisons in the technical document. They are not needed for the submission.

- `python solution/run.py --model p3`: the previous version (gradient-boosted team-shape model and refiner);
- `python solution/run.py --model p1`: rules only, no trained model.

Further `run.py` flags: `--name <stem>` writes another output file name; `--rebuild` recomputes the cached features.

## Retrain the models (optional)

This needs a CUDA GPU, `torch==2.12.0` (we used the cu130 build on an NVIDIA RTX PRO 4000) and `lxml==6.1.3`. It
also needs the external data:
- **Sportec Open DFL** (CC-BY 4.0), matches J03WMX, J03WN1, J03WPY, J03WOH, J03WQQ and J03WOY, plus J03WR9 for the
  held-out test. Put them in `../external/sportec-dfl/`, or set `DFL_DIR`.
- **Metrica sample data**, games 1–3. Put them in `../external/metrica-sample-data/`, or set `METRICA_SAMPLE_DIR`.

```
python solution/external/tcn.py seq                      # degraded input sequences of the 8 calibration matches
python solution/external/tcn_gpu.py seq_aug k 4 0        # four further realisations of the kinematic profile k
python solution/external/tcn_gpu.py seq_aug k 5 1        #   (the last argument mirrors the pitch)
python solution/external/tcn_gpu.py seq_aug k 6 0
python solution/external/tcn_gpu.py seq_aug k 7 1
python solution/external/tcn_gpu.py final k4_e10 0 1 2   # one model per seed, trained on all 8 matches
python solution/external/tcn_gpu.py lomo k4_e10 0 1 2    # leave-one-match-out evaluation of the same recipe
```

Then export the trained models to the format that `run.py` reads:

```
python -c "import sys, torch; sys.path[:0] = ['solution', 'solution/external']; import tcn_infer; [tcn_infer.export(torch.load('solution/cache/tcn_gpu/k4_e10/s%d/model_final.pt' % s, weights_only=False), 'solution/models/tcn_k2_s%d.npz' % s) for s in (0, 1, 2)]"
```

Training is deterministic on one GPU but is not expected to give bit-identical weights on other hardware. The
submission is defined by the committed model files.

## Layout

| path | what |
|---|---|
| `solution/run.py` | entry point: features, rule evidence, TCN ensemble, Viterbi decoder, CSV, validation |
| `solution/atd_cache.py`, `features.py` | parse the tactical tracking once; clean ball, speeds, nearest players |
| `solution/dead_ball.py`, `possession.py` | rule evidence for dead ball and team; stop and kick events; the decoder |
| `solution/team_shape.py`, `learned.py` | the 75 team-level input columns, in two team views |
| `solution/tcn_infer.py`, `solution/models/` | NumPy inference and the three submitted models (`p3_*`: the previous version) |
| `solution/stage2.py`, `turnover.py`, `snap.py`, `player_tokens.py` | the previous version's refiner and options tested during development |
| `solution/external/` | external-data loaders (`sportec_dfl.py`, `metrica_sample.py`, `skillcorner.py`) and the training and evaluation scripts (`tcn_gpu.py`, `tcn.py`, `lomo.py`, …) |
| `solution/reports/` | the aggregate result files behind the numbers in the technical report (no frame-level data); index in [solution/reports/README.md](solution/reports/README.md) |
| `docs/report/` | the technical document: LaTeX source, PDF and figures (external matches and label-free aggregates only) |
| `evaluation/`, `starter/`, `games.py`, `data/README.md` | the organisers' tooling, unchanged |

## Licences and data

- Our work (`solution/`, this README, the technical report) is under the MIT licence in
  [solution/LICENSE](solution/LICENSE). The organisers' files are under their [LICENSE](LICENSE).
- The provided match data is **not** in this repository: `data/`, `solution/cache/` and `evaluation/results/*.csv`
  are gitignored.
- Training data:
  - Sportec Open DFL (Bassek et al., *Scientific Data* 2025), CC-BY 4.0;
  - Metrica Sports sample data, github.com/metrica-sports/sample-data.
- SkillCorner open data (MIT) was used as a test set only.
