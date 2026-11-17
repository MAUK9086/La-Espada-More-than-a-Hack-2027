# Result files

These are the aggregate results behind the numbers in the technical document,
[docs/report/La_Espada.pdf](../../docs/report/La_Espada.pdf). No file holds frame-level data of the provided match.
Experiment labels (E55–E71) are the numbers the code uses (`solution/external/tcn_gpu.py`).

Unless a file says otherwise, the setup is:
- leave-one-match-out over the 8 external training matches;
- trained on degraded copies and tested on the kinematic dropout profile `k`;
- spellF1 given as "over 8 matches / over the 6 DFL matches".

| file | experiment | content | used in the report |
|---|---|---|---|
| `diagnostics.md` | – | label-free statistics of the provided tracking: ball visibility, gaps, jumps, track slots | §II, Fig. 1 |
| `tcn_gpu_seeds.md` | E59 | seed noise and the 5-seed ensemble of the first GPU TCN | §VI-C (Table V), Fig. 5 |
| `tcn_gpu_aug.md` | E60 | several degradation realisations per training match | §VI-C |
| `tcn_gpu_profile_touch.md` | E61 | training dropout profile (`k` against the earlier profile); auxiliary touch-level target | §VI-C, §VI-D |
| `tcn_gpu_steps.md` | E62 | longer training without realisations (equal-steps control); more realisations | §VI-C |
| `tcn_gpu_label_rule.md` | E63 | control rule of the training labels against the rule of the truth (earlier profile) | – |
| `tcn_gpu_skillcorner_v2.md` | E64a | models trained on the 8 matches, tested on 20 SkillCorner matches of real automatic tracking | §VI-D |
| `tcn_gpu_skc_train.md` | E64b | the SkillCorner matches as extra training data | §VI-D |
| `tcn_gpu_set.md` | E65 | a per-player set encoder in front of the TCN | §VI-C |
| `tcn_gpu_k8.md` | E66 | 8 realisations of profile `k`; both profiles mixed | §VI-C |
| `tcn_gpu_label_rule_k.md` | E67 | control-rule robustness of the submitted recipe | §IV-B, §VI-E (Table VI) |
| `tcn_gpu_ceiling.md` | E69 | the cost of the tracking: the recipe on corrected and on ball-complete tracking | §VI-F, Table VII |
| `tcn_gpu_screen.md` | E70 | architecture and training screen (width, depth, stages, GRU head, regularisation, test-time mirroring) | §VI-C |
| `tcn_gpu_cos_ema.md` | E71 | AdamW with a cosine schedule and weight averaging; also holds the submitted recipe's ensemble row (`k4_e10, std, ensemble of 3`) | §VI-B, §VI-C, Fig. 5 |
| `error_anatomy_calib.md` | – | where the errors are: oracle scores, frame errors, recall by possession length (8 matches) | §VI-B, §VI-F |
| `error_anatomy_skc.md` | – | the same analysis on the 20 SkillCorner matches | §VI-D |
| `tcn_gpu_barath_preview.md` | – | label-free counts on the provided match for the candidate models; determinism of the NumPy inference | §VI-G |
| `holdout_log.md` | – | every run on the two held-out matches, in order | §VI-B (Table IV), §VI-E |

**Names used in these files**
- `p3-refined`: the previous version (gradient boosting on frame windows plus a refining stage).
- `base`: the first GPU TCN. It used one realisation of the earlier dropout profile `r1` and 20 epochs, and is the
  network of E57, the earlier CPU experiment of `solution/external/tcn.py`.
- `aug4_e10`: 4 realisations of `r1`, 10 epochs.
- `k4_e10`: 4 realisations of `k`, 10 epochs. This is the submitted recipe; its seeds 0, 1 and 2 are in
  `solution/models/`.
- `std`, `A`, `B`: decoders.
  - `std` is the standard decoder (submitted).
  - `A` enforces a 2 s minimum possession (the R1 decoder of E55, `solution/external/min_duration.py`).
  - `B` blends with the previous version.
- `K2` / `K1.5`: the `k4_e10` recipe trained on 2 s / 1.5 s control-rule labels.
