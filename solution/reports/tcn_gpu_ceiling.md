# E69: how much the tracking quality costs (ceiling test)

The `k4_e10` recipe (seed 0) trained and tested on the same kind of input, LOMO over the 8 calibration matches, standard decoder α 0.75, P1 evidence rebuilt from the same input. spellF1 over 8 matches / on DFL.

| input | P1 rules alone | TCN (seed 0) | TCN, true dead frames given | TCN, true team given | ball visible | per match (TCN) |
|---|---|---|---|---|---|---|
| k (as BAR-ATH is simulated) | 0.4625 / 0.4278 | 0.7523 / 0.7383 | 0.7864 / 0.7658 | 0.9354 / 0.9391 | 60.6% | 0.8154 0.7733 0.7094 0.7122 0.7693 0.7566 0.7471 0.7355 |
| nodrop (the ball never missing) | 0.5088 / 0.4849 | 0.7903 / 0.7880 | 0.8065 / 0.7985 | 0.9619 / 0.9725 | 90.9% | 0.8118 0.7824 0.7745 0.7831 0.8052 0.8000 0.7737 0.7912 |
| clean (corrected tracking) | 0.6110 / 0.5951 | 0.8217 / 0.8113 | 0.8338 / 0.8189 | 0.9730 / 0.9786 | 91.0% | 0.8517 0.8538 0.8093 0.8120 0.8281 0.8135 0.7917 0.8130 |

Matches: metrica1, metrica2, dfl_J03WMX, dfl_J03WN1, dfl_J03WPY, dfl_J03WOH, dfl_J03WQQ, dfl_J03WOY.
