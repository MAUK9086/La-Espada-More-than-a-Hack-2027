# BAR-ATH preview of GPU TCN candidates (label-free)

No labels are read and the public window is not scored. K2 = `k4_e10`, K1.5 = `k4_e10_c1.5`, 3-seed ensembles of final models, standard decoder α 0.75; `aug4_e10 (r1)` is the `r1`-trained ensemble, for comparison only. "(NumPy)" is the float64 inference of `solution/tcn_infer.py` on the same models. Shares are of the graded frames, against the previous version (p3-refined).

| system | sha256 | deterministic | possessions / h | stoppages / h | % possessions < 2 s | % dead | frames changed | submission live → dead | submission dead → live | both live, other team |
|---|---|---|---|---|---|---|---|---|---|---|
| p3-refined | `7daa4a47bc5883b6` | yes | 156 | 65 | 8.7 | 44.3 | 0.00% | 0.00% | 0.00% | 0.00% |
| K2 | `8286097e28c9e9ab` | yes | 148 | 68 | 7.7 | 45.4 | 5.22% | 2.09% | 0.98% | 2.15% |
| K2 (NumPy) | `8286097e28c9e9ab` | yes | 148 | 68 | 7.7 | 45.4 | 5.22% | 2.09% | 0.98% | 2.15% |
| K1.5 | `53d3e1ceeaa6c59b` | yes | 164 | 69 | 11.1 | 45.2 | 5.38% | 2.03% | 1.11% | 2.24% |
| K1.5 (NumPy) | `285c631e4d5a2275` | yes | 164 | 69 | 11.1 | 45.2 | 5.38% | 2.03% | 1.11% | 2.24% |
| aug4_e10 (r1) | `1dd300d859391e47` | yes | 147 | 65 | 7.7 | 44.3 | 4.72% | 1.35% | 1.33% | 2.04% |

## Between systems (share of graded frames that differ)

- K2 vs K2 (NumPy): 0.00% (alive/dead 0.00%, team 0.00%)
- K2 vs K1.5: 1.40% (alive/dead 0.45%, team 0.95%)
- K2 vs K1.5 (NumPy): 1.39% (alive/dead 0.45%, team 0.95%)
- K2 vs aug4_e10 (r1): 3.31% (alive/dead 2.20%, team 1.11%)
- K2 (NumPy) vs K1.5: 1.40% (alive/dead 0.45%, team 0.95%)
- K2 (NumPy) vs K1.5 (NumPy): 1.39% (alive/dead 0.45%, team 0.95%)
- K2 (NumPy) vs aug4_e10 (r1): 3.31% (alive/dead 2.20%, team 1.11%)
- K1.5 vs K1.5 (NumPy): 0.00% (alive/dead 0.00%, team 0.00%)
- K1.5 vs aug4_e10 (r1): 3.77% (alive/dead 2.30%, team 1.46%)
- K1.5 (NumPy) vs aug4_e10 (r1): 3.77% (alive/dead 2.30%, team 1.46%)

Run time 139 s.
