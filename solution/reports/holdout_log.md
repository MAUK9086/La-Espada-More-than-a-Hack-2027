# Held-out log: Metrica sample game 3 and Sportec DFL J03WR9 (rows marked [dfl])

One line per held-out run, in order. The commit column identifies the code version in the team's development
history, which is not part of this repository. `+dirty`: uncommitted changes were present.

| date | commit | change checked | spellF1 | macroF1 | poss pred/true |
|---|---|---|---|---|---|
| 2026-10-28 | e51f916+dirty | P1 as committed (p1-viterbi + throw-in fix + game-3 loader) | 0.5691 | 0.8818 | 292/401 |
| 2026-10-28 | e51f916+dirty | P1, ATD-like degraded game 3 (old kick windows) | 0.4938 | 0.8614 | 281/401 |
| 2026-10-28 | e51f916+dirty | kick windows (-20,-5)/(5,15), decided on degraded games 1-2 | 0.5333 | 0.8720 | 289/401 |
| 2026-10-28 | a2c56f0+dirty | throw-in allowance limited to touchlines (Law 15) | 0.5333 | 0.8720 | 289/401 |
| 2026-10-28 | f1eb7e6+dirty | P1 reference, degraded players + ball (game 3) | 0.4909 | 0.8527 | 303/401 |
| 2026-10-28 | f1eb7e6+dirty | P3 team-shape HGB alpha=0.5, trained on games 1-2, decided by LOMO | 0.6365 | 0.9261 | 316/401 |
| 2026-10-28 | 73ba986+dirty | [dfl] P1 baseline (first DFL check) | 0.4537 | 0.7172 | 195/222 |
| 2026-10-28 | 73ba986+dirty | [dfl] P3 trained on Metrica 1+2 only, a=0.5 (before DFL) | 0.5267 | 0.7641 | 258/222 |
| 2026-10-28 | 73ba986+dirty | P3 trained on 8 matches (Metrica 1-2 + 6 DFL, controlled 3 s labels), a=0.5 | 0.6107 | 0.9250 | 288/401 |
| 2026-10-28 | 73ba986+dirty | [dfl] P3 trained on 8 matches (Metrica 1-2 + 6 DFL, controlled 3 s labels), a=0.5 | 0.7103 | 0.9181 | 224/222 |
| 2026-10-30 | c264615+dirty | [dfl] reference: p3-dfl8 model rescored against 2 s labels | 0.6542 | 0.9122 | 224/267 |
| 2026-10-30 | c264615+dirty | P3 retrained on 2 s control-rule labels (robustness choice), a=0.5 | 0.6189 | 0.9241 | 297/401 |
| 2026-10-30 | c264615+dirty | [dfl] P3 retrained on 2 s control-rule labels (robustness choice), a=0.5 | 0.6669 | 0.9181 | 232/267 |
| 2026-10-31 | b92263f | [dfl] reference: p3-ctrl2s model on label-free dropout (B0) test inputs and controlled labels | 0.6600 | 0.8999 | 236/267 |
| 2026-10-31 | b92263f+dirty | reference: p3-ctrl2s model on label-free dropout (B0) test inputs and controlled labels | 0.4657 | 0.6717 | 244/286 |
| 2026-10-31 | b92263f+dirty | [dfl] P3 retrained on label-free dropout r1 (B0), controlled Metrica labels, a=0.5 | 0.6461 | 0.9010 | 232/267 |
| 2026-10-31 | b92263f+dirty | [dfl] same model on the old test profile | 0.6539 | 0.9148 | 234/267 |
| 2026-10-31 | b92263f+dirty | P3 retrained on label-free dropout r1 (B0), controlled Metrica labels, a=0.5 | 0.6610 | 0.8697 | 232/286 |
| 2026-10-31 | b92263f+dirty | same model on the old test profile | 0.6100 | 0.9136 | 277/401 |
| 2026-10-31 | 3084922 | [dfl] reference: p3-labelfree model, speed-fix inputs, kinematic test | 0.5858 | 0.8793 | 213/267 |
| 2026-10-31 | 3084922+dirty | [dfl] reference: p3-labelfree model, speed-fix inputs, old test | 0.6103 | 0.8933 | 218/267 |
| 2026-10-31 | 3084922+dirty | reference: p3-labelfree model, speed-fix inputs, kinematic test | 0.7286 | 0.9037 | 274/286 |
| 2026-10-31 | 3084922+dirty | reference: p3-labelfree model, speed-fix inputs, old test | 0.6100 | 0.9136 | 277/401 |
| 2026-10-31 | 3084922+dirty | [dfl] P3 retrained with DFL speeds in m/s, r1 training, kinematic test | 0.6443 | 0.9044 | 234/267 |
| 2026-10-31 | 3084922+dirty | [dfl] same model, old test | 0.6610 | 0.9185 | 231/267 |
| 2026-10-31 | 3084922+dirty | P3 retrained with DFL speeds in m/s, r1 training, kinematic test | 0.7300 | 0.9066 | 274/286 |
| 2026-10-31 | 3084922+dirty | same model, old test | 0.6201 | 0.9126 | 277/401 |
| 2026-10-31 | 9c3732d | [dfl] decoder candidate (W_NEAR 0 decoder only, a=0.65, C_SWITCH 12, islands 35 keep), kinematic test | 0.6457 | 0.9053 | 223/267 |
| 2026-10-31 | 9c3732d+dirty | [dfl] same, old test | 0.6571 | 0.9186 | 223/267 |
| 2026-10-31 | 9c3732d+dirty | decoder candidate (W_NEAR 0 decoder only, a=0.65, C_SWITCH 12, islands 35 keep), kinematic test | 0.7428 | 0.9071 | 259/286 |
| 2026-10-31 | 9c3732d+dirty | same, old test | 0.6166 | 0.9131 | 260/401 |
| 2026-11-01 | f36a018 | [dfl] Stage-2 refiner, hgb, a=0.75, kinematic test | 0.6594 | 0.9092 | 231/267 |
| 2026-11-01 | f36a018+dirty | [dfl] same, old test | 0.6850 | 0.9227 | 227/267 |
| 2026-11-01 | f36a018+dirty | Stage-2 refiner, hgb, a=0.75, kinematic test | 0.7462 | 0.9066 | 279/286 |
| 2026-11-01 | f36a018+dirty | same, old test | 0.6362 | 0.9144 | 285/401 |
| 2026-11-01 | f36a018+dirty | [dfl] L_local, signed ball velocity and nearest three players, a=0.5, kinematic test | 0.6851 | 0.9114 | 231/267 |
| 2026-11-01 | f36a018+dirty | [dfl] same, old test | 0.7008 | 0.9241 | 237/267 |
| 2026-11-01 | f36a018+dirty | L_local, signed ball velocity and nearest three players, a=0.5, kinematic test | 0.7566 | 0.9107 | 286/286 |
| 2026-11-01 | f36a018+dirty | same, old test | 0.6345 | 0.9159 | 288/401 |
| 2026-11-05 | e63a258+dirty | [dfl] E1a refiner on p3-local, hgb a=0.75, kinematic test | 0.6895 | 0.9114 | 229/267 |
| 2026-11-05 | e63a258+dirty | [dfl] E1b D1 decoder (a=0.65, C_SWITCH 12, W_NEAR 0, islands 35 keep) on p3-local, kinematic test | 0.6879 | 0.9125 | 229/267 |
| 2026-11-05 | e63a258+dirty | E1a refiner on p3-local, hgb a=0.75, kinematic test | 0.7681 | 0.9125 | 279/286 |
| 2026-11-05 | e63a258+dirty | E1b D1 decoder (a=0.65, C_SWITCH 12, W_NEAR 0, islands 35 keep) on p3-local, kinematic test | 0.7719 | 0.9122 | 270/286 |
| 2026-11-05 | 1e5f50f | [dfl] E3_both1 sample weights short 1 edge 1 on p3-local, kinematic test | 0.6844 | 0.9104 | 238/267 |
| 2026-11-05 | 1e5f50f+dirty | E3_both1 sample weights short 1 edge 1 on p3-local, kinematic test | 0.7604 | 0.9114 | 300/286 |
| 2026-11-06 | 577b273+dirty | [dfl] R5 TCN boundary (final, 8 matches), alpha 0.75, R1 steady L50 cs10, kinematic test | 0.7728 | 0.9312 | 248/267 |
| 2026-11-06 | 577b273+dirty | [dfl] R5 blend w 0.3 p3-refined + TCN boundary (final, 8 matches), alpha 0.75, standard decoder, kinematic test | 0.7761 | 0.9302 | 252/267 |
| 2026-11-06 | 577b273+dirty | R5 TCN boundary (final, 8 matches), alpha 0.75, R1 steady L50 cs10, kinematic test | 0.7752 | 0.9058 | 279/286 |
| 2026-11-06 | 577b273+dirty | R5 blend w 0.3 p3-refined + TCN boundary (final, 8 matches), alpha 0.75, standard decoder, kinematic test | 0.7790 | 0.9115 | 285/286 |
| 2026-11-12 | 5ae09d7 | [dfl] TCN k4_e10 (GPU, final, 8 matches), ensemble of seeds 0,1,2, alpha 0.75, standard decoder, kinematic test | 0.7885 | 0.9355 | 254/267 |
| 2026-11-12 | 5ae09d7 | [dfl] TCN k4_e10_c1.5 (GPU, final, 8 matches), ensemble of seeds 0,1,2, alpha 0.75, standard decoder, kinematic test | 0.7671 | 0.9333 | 268/267 |
| 2026-11-12 | 5ae09d7+dirty | TCN k4_e10 (GPU, final, 8 matches), ensemble of seeds 0,1,2, alpha 0.75, standard decoder, kinematic test | 0.8143 | 0.9211 | 273/286 |
| 2026-11-12 | 5ae09d7+dirty | TCN k4_e10_c1.5 (GPU, final, 8 matches), ensemble of seeds 0,1,2, alpha 0.75, standard decoder, kinematic test | 0.8176 | 0.9262 | 304/286 |
