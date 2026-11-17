# Error anatomy, LOMO on the 8 calibration matches (tested on profile k)

8 matches; standard decoder, α 0.75; matching at IoU 0.5 for the recall and precision tables.

## Scores and oracles (mean over matches)

| | p3-refined | aug4_e10 ×3 (r1) | k4_e10 ×3 (k) |
|---|---|---|---|
| spellF1 | 0.7077 | 0.7524 | 0.7590 |
| spellF1, true dead frames given (team as predicted) | 0.7374 | 0.7792 | 0.7872 |
| spellF1, true team given (alive/dead as predicted) | 0.9155 | 0.9368 | 0.9367 |
| predicted / true possessions | 0.907 | 0.940 | 0.933 |
| recall at IoU 0.5 | 0.687 | 0.744 | 0.747 |
| precision at IoU 0.5 | 0.758 | 0.791 | 0.800 |

## Frame-level errors (share of graded frames)

| | p3-refined | aug4_e10 ×3 (r1) | k4_e10 ×3 (k) |
|---|---|---|---|
| true dead called live | 1.13 % | 1.12 % | 1.11 % |
| true live called dead | 2.90 % | 2.30 % | 2.14 % |
| wrong team | 4.71 % | 4.25 % | 4.16 % |
| of the true dead frames, called live | 2.6 % | 2.6 % | 2.6 % |
| of the true live frames, called dead | 5.1 % | 4.1 % | 3.8 % |

## True possessions by duration: share of all, recall, and share of all misses

| duration | share of true possessions | recall, p3-refined | share of misses | recall, aug4_e10 ×3 (r1) | share of misses | recall, k4_e10 ×3 (k) | share of misses |
|---|---|---|---|---|---|---|---|
| < 1 s | 0.6 % | 0.077 | 1.7 % | 0.231 | 1.7 % | 0.231 | 1.7 % |
| 1–2 s | 3.4 % | 0.468 | 5.8 % | 0.636 | 4.8 % | 0.610 | 5.2 % |
| 2–4 s | 23.7 % | 0.440 | 42.4 % | 0.524 | 43.9 % | 0.524 | 44.4 % |
| 4–8 s | 26.8 % | 0.662 | 28.9 % | 0.724 | 28.9 % | 0.732 | 28.3 % |
| 8–16 s | 24.8 % | 0.830 | 13.5 % | 0.863 | 13.3 % | 0.861 | 13.6 % |
| ≥ 16 s | 20.8 % | 0.883 | 7.8 % | 0.910 | 7.3 % | 0.919 | 6.6 % |

## Predicted possessions by duration: share of the system's predictions, precision

| duration | share, p3-refined | precision | share, aug4_e10 ×3 (r1) | precision | share, k4_e10 ×3 (k) | precision |
|---|---|---|---|---|---|---|
| < 1 s | 0.5 % | 0.000 | 0.6 % | 0.083 | 0.6 % | 0.250 |
| 1–2 s | 5.8 % | 0.373 | 5.7 % | 0.483 | 5.3 % | 0.505 |
| 2–4 s | 19.1 % | 0.589 | 20.3 % | 0.688 | 18.6 % | 0.702 |
| 4–8 s | 26.7 % | 0.740 | 26.2 % | 0.764 | 27.7 % | 0.758 |
| 8–16 s | 24.3 % | 0.869 | 24.7 % | 0.874 | 24.8 % | 0.883 |
| ≥ 16 s | 23.5 % | 0.911 | 22.7 % | 0.919 | 23.0 % | 0.924 |

## Recall by how the true possession starts

| | share of true possessions | recall, p3-refined | recall, aug4_e10 ×3 (r1) | recall, k4_e10 ×3 (k) |
|---|---|---|---|---|
| after stoppage | 34.6 % | 0.730 | 0.796 | 0.796 |
| after the other team | 64.2 % | 0.663 | 0.714 | 0.719 |
| start of a half | 1.2 % | 0.741 | 0.815 | 0.778 |

## Recall by how the true possession ends

| | share of true possessions | recall, p3-refined | recall, aug4_e10 ×3 (r1) | recall, k4_e10 ×3 (k) |
|---|---|---|---|---|
| into stoppage | 34.8 % | 0.706 | 0.757 | 0.766 |
| to the other team | 64.2 % | 0.677 | 0.738 | 0.737 |
| end of a half | 1.0 % | 0.696 | 0.696 | 0.696 |

## Why a true possession has no prediction with IoU ≥ 0.5 of its team (share of true possessions)

| | p3-refined | aug4_e10 ×3 (r1) | k4_e10 ×3 (k) |
|---|---|---|---|
| short | 15.6 % | 13.0 % | 13.1 % |
| split_dead | 5.8 % | 3.9 % | 3.7 % |
| split_team | 5.8 % | 5.1 % | 4.9 % |
| merged | 4.1 % | 3.7 % | 3.8 % |
| boundary | 0.0 % | 0.0 % | 0.0 % |
| other | 0.0 % | 0.0 % | 0.0 % |

short: the true possession is under 4.0 s; split_dead / split_team: a predicted stoppage / other-team run of at least 0.5 s lies inside it; merged: a prediction of its team covers it but is much longer; boundary: best IoU 0.3–0.5; other: the rest.

Matches: metrica1, metrica2, dfl_J03WMX, dfl_J03WN1, dfl_J03WPY, dfl_J03WOH, dfl_J03WQQ, dfl_J03WOY.

