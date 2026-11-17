# E63: control rule of the training labels against the control rule of the truth (TCN, GPU)

3-seed ensembles (seeds [0, 1, 2]) of the `aug4_e10` recipe, trained on labels built with a T_train-second control rule, each scored against labels built with T_eval. LOMO over the 8 calibration matches, tested on profile `k`. spellF1 over 8 matches / on DFL. The same P1 evidence and decoder throughout; p3-refined (trained on 2 s labels) is the first row of each block.

## Decode: std α 0.75

| trained on | T_eval 1.0 s | T_eval 1.5 s | T_eval 2.0 s | T_eval 3.0 s | worst case (8) | mean (8) |
|---|---|---|---|---|---|---|
| p3-refined (2 s), its own decode | 0.6385 / 0.6171 | 0.6837 / 0.6625 | 0.7077 / 0.6889 | 0.7067 / 0.6897 | 0.6385 | 0.6842 |
| T_train 1.0 s | 0.7198 / 0.7077 | 0.7418 / 0.7265 | 0.7320 / 0.7170 | 0.6861 / 0.6687 | 0.6861 | 0.7199 |
| T_train 1.5 s | 0.7122 / 0.7000 | 0.7476 / 0.7328 | 0.7483 / 0.7338 | 0.7120 / 0.6950 | 0.7120 | 0.7300 |
| T_train 2.0 s | 0.6849 / 0.6712 | 0.7296 / 0.7155 | 0.7524 / 0.7416 | 0.7360 / 0.7224 | 0.6849 | 0.7257 |
| T_train 3.0 s | 0.6252 / 0.6067 | 0.6775 / 0.6607 | 0.7158 / 0.7016 | 0.7546 / 0.7421 | 0.6252 | 0.6933 |

## Decode: std α 1.5

| trained on | T_eval 1.0 s | T_eval 1.5 s | T_eval 2.0 s | T_eval 3.0 s | worst case (8) | mean (8) |
|---|---|---|---|---|---|---|
| p3-refined (2 s), its own decode | 0.6385 / 0.6171 | 0.6837 / 0.6625 | 0.7077 / 0.6889 | 0.7067 / 0.6897 | 0.6385 | 0.6842 |
| T_train 1.0 s | 0.7220 / 0.7091 | 0.7401 / 0.7259 | 0.7286 / 0.7143 | 0.6805 / 0.6630 | 0.6805 | 0.7178 |
| T_train 1.5 s | 0.7181 / 0.7084 | 0.7505 / 0.7379 | 0.7461 / 0.7332 | 0.7069 / 0.6909 | 0.7069 | 0.7304 |
| T_train 2.0 s | 0.6876 / 0.6763 | 0.7327 / 0.7212 | 0.7509 / 0.7414 | 0.7324 / 0.7192 | 0.6876 | 0.7259 |
| T_train 3.0 s | 0.6268 / 0.6114 | 0.6788 / 0.6653 | 0.7198 / 0.7084 | 0.7575 / 0.7467 | 0.6268 | 0.6957 |

## Decode: R1 α 1.5

| trained on | T_eval 1.0 s | T_eval 1.5 s | T_eval 2.0 s | T_eval 3.0 s | worst case (8) | mean (8) |
|---|---|---|---|---|---|---|
| p3-refined (2 s), its own decode | 0.6385 / 0.6171 | 0.6837 / 0.6625 | 0.7077 / 0.6889 | 0.7067 / 0.6897 | 0.6385 | 0.6842 |
| T_train 1.0 s | 0.7255 / 0.7128 | 0.7543 / 0.7396 | 0.7436 / 0.7293 | 0.6966 / 0.6783 | 0.6966 | 0.7300 |
| T_train 1.5 s | 0.7137 / 0.7012 | 0.7523 / 0.7373 | 0.7570 / 0.7439 | 0.7200 / 0.7042 | 0.7137 | 0.7357 |
| T_train 2.0 s | 0.6871 / 0.6753 | 0.7335 / 0.7220 | 0.7548 / 0.7464 | 0.7375 / 0.7252 | 0.6871 | 0.7282 |
| T_train 3.0 s | 0.6259 / 0.6105 | 0.6781 / 0.6646 | 0.7205 / 0.7082 | 0.7600 / 0.7479 | 0.6259 | 0.6961 |

True possessions over the 8 matches by rule: 1.0 s: 2845, 1.5 s: 2515, 2.0 s: 2257, 3.0 s: 1870.
Matches: metrica1, metrica2, dfl_J03WMX, dfl_J03WN1, dfl_J03WPY, dfl_J03WOH, dfl_J03WQQ, dfl_J03WOY.
