# E67: control rule of the training labels against the control rule of the truth (TCN, GPU)

3-seed ensembles (seeds {}) of the `k4_e10` recipe, trained on labels built with a T_train-second control rule, each scored against labels built with T_eval. LOMO over the 8 calibration matches, tested on profile `k`. spellF1 over 8 matches / on DFL. The same P1 evidence and decoder throughout; p3-refined (trained on 2 s labels) is the first row of each block.

## Decode: std α 0.75

| trained on | T_eval 1.0 s | T_eval 1.5 s | T_eval 2.0 s | T_eval 3.0 s | worst case (8) | mean (8) |
|---|---|---|---|---|---|---|
| p3-refined (2 s), its own decode | 0.6385 / 0.6171 | 0.6837 / 0.6625 | 0.7077 / 0.6889 | 0.7067 / 0.6897 | 0.6385 | 0.6842 |
| T_train 1.5 s | 0.7160 / 0.7042 | 0.7511 / 0.7364 | 0.7517 / 0.7379 | 0.7135 / 0.6973 | 0.7135 | 0.7331 |
| T_train 2.0 s | 0.6905 / 0.6753 | 0.7357 / 0.7204 | 0.7590 / 0.7460 | 0.7469 / 0.7305 | 0.6905 | 0.7330 |

## Decode: std α 1.5

| trained on | T_eval 1.0 s | T_eval 1.5 s | T_eval 2.0 s | T_eval 3.0 s | worst case (8) | mean (8) |
|---|---|---|---|---|---|---|
| p3-refined (2 s), its own decode | 0.6385 / 0.6171 | 0.6837 / 0.6625 | 0.7077 / 0.6889 | 0.7067 / 0.6897 | 0.6385 | 0.6842 |
| T_train 1.5 s | 0.7239 / 0.7128 | 0.7575 / 0.7431 | 0.7541 / 0.7393 | 0.7134 / 0.6961 | 0.7134 | 0.7372 |
| T_train 2.0 s | 0.6966 / 0.6839 | 0.7402 / 0.7265 | 0.7602 / 0.7477 | 0.7435 / 0.7278 | 0.6966 | 0.7351 |

## Decode: R1 α 1.5

| trained on | T_eval 1.0 s | T_eval 1.5 s | T_eval 2.0 s | T_eval 3.0 s | worst case (8) | mean (8) |
|---|---|---|---|---|---|---|
| p3-refined (2 s), its own decode | 0.6385 / 0.6171 | 0.6837 / 0.6625 | 0.7077 / 0.6889 | 0.7067 / 0.6897 | 0.6385 | 0.6842 |
| T_train 1.5 s | 0.7200 / 0.7056 | 0.7591 / 0.7432 | 0.7611 / 0.7464 | 0.7224 / 0.7057 | 0.7200 | 0.7406 |
| T_train 2.0 s | 0.6933 / 0.6802 | 0.7390 / 0.7256 | 0.7619 / 0.7493 | 0.7490 / 0.7337 | 0.6933 | 0.7358 |

True possessions over the 8 matches by rule: 1.0 s: 2845, 1.5 s: 2515, 2.0 s: 2257, 3.0 s: 1870.
Matches: metrica1, metrica2, dfl_J03WMX, dfl_J03WN1, dfl_J03WPY, dfl_J03WOH, dfl_J03WQQ, dfl_J03WOY.
