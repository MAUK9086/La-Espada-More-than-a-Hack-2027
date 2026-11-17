# Error anatomy, models trained on the calibration matches, on the 20 SkillCorner matches

20 matches; standard decoder, α 0.75; matching at IoU 0.5 for the recall and precision tables.

## Scores and oracles (mean over matches)

| | p3-refined | aug4_e10 ×3 (r1) | k4_e10 ×3 (k) |
|---|---|---|---|
| spellF1 | 0.4483 | 0.4316 | 0.4828 |
| spellF1, true dead frames given (team as predicted) | 0.5631 | 0.5635 | 0.5570 |
| spellF1, true team given (alive/dead as predicted) | 0.7790 | 0.7643 | 0.8316 |
| predicted / true possessions | 1.230 | 1.246 | 1.143 |
| recall at IoU 0.5 | 0.505 | 0.491 | 0.528 |
| precision at IoU 0.5 | 0.411 | 0.394 | 0.463 |

## Frame-level errors (share of graded frames)

| | p3-refined | aug4_e10 ×3 (r1) | k4_e10 ×3 (k) |
|---|---|---|---|
| true dead called live | 17.91 % | 26.24 % | 4.77 % |
| true live called dead | 1.49 % | 1.12 % | 1.86 % |
| wrong team | 8.21 % | 8.43 % | 8.21 % |
| of the true dead frames, called live | 38.5 % | 56.5 % | 10.3 % |
| of the true live frames, called dead | 2.8 % | 2.1 % | 3.5 % |

## True possessions by duration: share of all, recall, and share of all misses

| duration | share of true possessions | recall, p3-refined | share of misses | recall, aug4_e10 ×3 (r1) | share of misses | recall, k4_e10 ×3 (k) | share of misses |
|---|---|---|---|---|---|---|---|
| < 1 s | 2.7 % | 0.007 | 5.4 % | 0.015 | 5.3 % | 0.015 | 5.7 % |
| 1–2 s | 6.3 % | 0.067 | 11.8 % | 0.064 | 11.5 % | 0.067 | 12.4 % |
| 2–4 s | 16.4 % | 0.220 | 25.9 % | 0.220 | 25.1 % | 0.244 | 26.3 % |
| 4–8 s | 23.6 % | 0.438 | 26.8 % | 0.436 | 26.1 % | 0.483 | 25.9 % |
| 8–16 s | 24.3 % | 0.616 | 18.9 % | 0.567 | 20.6 % | 0.647 | 18.2 % |
| ≥ 16 s | 26.7 % | 0.792 | 11.2 % | 0.784 | 11.3 % | 0.796 | 11.6 % |

## Predicted possessions by duration: share of the system's predictions, precision

| duration | share, p3-refined | precision | share, aug4_e10 ×3 (r1) | precision | share, k4_e10 ×3 (k) | precision |
|---|---|---|---|---|---|---|
| < 1 s | 0.9 % | 0.019 | 1.0 % | 0.050 | 1.2 % | 0.014 |
| 1–2 s | 6.6 % | 0.071 | 5.1 % | 0.111 | 6.4 % | 0.088 |
| 2–4 s | 18.3 % | 0.153 | 17.2 % | 0.164 | 20.9 % | 0.167 |
| 4–8 s | 23.4 % | 0.327 | 23.0 % | 0.322 | 26.1 % | 0.353 |
| 8–16 s | 22.8 % | 0.525 | 21.7 % | 0.506 | 21.8 % | 0.625 |
| ≥ 16 s | 28.0 % | 0.650 | 32.1 % | 0.549 | 23.6 % | 0.821 |

## Recall by how the true possession starts

| | share of true possessions | recall, p3-refined | recall, aug4_e10 ×3 (r1) | recall, k4_e10 ×3 (k) |
|---|---|---|---|---|
| after stoppage | 41.4 % | 0.474 | 0.446 | 0.505 |
| after the other team | 58.6 % | 0.527 | 0.522 | 0.544 |
| start of a half | 0.1 % | 1.000 | 1.000 | 1.000 |

## Recall by how the true possession ends

| | share of true possessions | recall, p3-refined | recall, aug4_e10 ×3 (r1) | recall, k4_e10 ×3 (k) |
|---|---|---|---|---|
| into stoppage | 41.4 % | 0.549 | 0.529 | 0.583 |
| to the other team | 58.6 % | 0.474 | 0.464 | 0.490 |
| end of a half | 0.0 % | 0.000 | 0.000 | 0.000 |

## Why a true possession has no prediction with IoU ≥ 0.5 of its team (share of true possessions)

| | p3-refined | aug4_e10 ×3 (r1) | k4_e10 ×3 (k) |
|---|---|---|---|
| short | 21.3 % | 21.3 % | 20.9 % |
| split_dead | 4.5 % | 3.4 % | 4.7 % |
| split_team | 16.5 % | 18.2 % | 15.8 % |
| merged | 7.2 % | 8.0 % | 5.6 % |
| boundary | 0.0 % | 0.0 % | 0.0 % |
| other | 0.0 % | 0.0 % | 0.0 % |

short: the true possession is under 4.0 s; split_dead / split_team: a predicted stoppage / other-team run of at least 0.5 s lies inside it; merged: a prediction of its team covers it but is much longer; boundary: best IoU 0.3–0.5; other: the rest.

Matches: skc_1874553, skc_1886347, skc_1899585, skc_1925299, skc_1927964, skc_1953632, skc_1959846, skc_1986691, skc_1996435, skc_1996436, skc_2006229, skc_2006363, skc_2007448, skc_2007721, skc_2010085, skc_2011166, skc_2013725, skc_2015213, skc_2016236, skc_2017461.

