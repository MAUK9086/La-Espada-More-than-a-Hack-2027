# Data diagnostics — BAR-ATH (tactical)

Teams: H = Barcelona, A = Atletic Club. Graded halves (video frames): [(1, 11185, 90651), (2, 110808, 190197)].

## Ball
| scope | ball visible |
|---|---|
| all frames | 56.9% |
| graded frames | 59.3% |
| half 1 | 59.3% |
| half 2 | 59.3% |

Ball gaps inside graded play (150 gaps, 64643 missing frames). Length in frames (25 = 1 s):

| gap length | n gaps | share of missing frames |
|---|---|---|
| 1-1 | 2 | 0.0% |
| 2-4 | 11 | 0.0% |
| 5-12 | 3 | 0.0% |
| 13-24 | 1 | 0.0% |
| 25-49 | 0 | 0.0% |
| 50-124 | 1 | 0.2% |
| 125-249 | 53 | 14.5% |
| 250-749 | 60 | 41.4% |
| >= 750 | 19 | 43.8% |

Frame-to-frame ball speed where consecutive frames both see it (m/s, p5/p25/p50/p75/p95): 0.52 / 2.51 / 5.54 / 10.81 / 23.80
Implied speed > 40 m/s (likely false detections / jumps): 2043 frames (1.8% of consecutive pairs)
Tracker ball speed channel (m/s): 0.16 / 0.70 / 1.21 / 1.77 / 4.07
Ball outside the pitch lines on graded frames: 3949 frames in 81 runs
Ball position range on visible frames: x -6.9..117.3 m, y -7.3..101.2 m

## Players
| per graded frame | p5/p25/p50/p75/p95 | mean |
|---|---|---|
| tracked objects | 14.00 / 21.00 / 22.00 / 22.00 / 22.00 | 20.5 |
| home objects | 6.00 / 10.00 / 11.00 / 11.00 / 12.00 | 10.2 |
| away objects | 7.00 / 10.00 / 11.00 / 11.00 / 11.00 | 10.3 |

Graded frames with > 22 objects: 2.4%; with > 11 for one team: 12.9%; with < 9 for one team: 12.7%
Tracklets (contiguous visible runs of one slot): 2218; length in seconds p5/p25/p50/p75/p95: 1.52 / 4.97 / 13.58 / 63.15 / 320.75
Slots ever used: 45 of 46

## Ball-player proximity (visible ball, graded frames)
Nearest player distance (m) p5/p25/p50/p75/p95: 0.27 / 0.89 / 1.91 / 4.06 / 9.56
| radius | frames with a player within | of those, opponent within r+1 m |
|---|---|---|
| 0.5 m | 11.9% | 8.3% |
| 1.0 m | 28.5% | 13.6% |
| 1.5 m | 41.8% | 18.5% |
| 2.0 m | 51.5% | 23.7% |
| 3.0 m | 65.3% | 34.2% |
| 5.0 m | 81.4% | 52.1% |

## Collective motion
Mean player speed over graded frames (m/s) p5/p25/p50/p75/p95: 0.55 / 1.02 / 1.52 / 2.32 / 3.68

## Smart Tagging
| code | n | median s | team tags |
|---|---|---|---|
| ATTACKING TRANSITION | 24 | 15.4 | Barcelona=13, Atletic Club=11 |
| BALL IN FINAL THIRD | 93 | 27.0 | Barcelona=54, Atletic Club=39 |
| BALL IN THE BOX | 72 | 17.4 | Barcelona=42, Atletic Club=30 |
| BUILD UP | 22 | 20.5 | Barcelona=15, Atletic Club=7 |
| CREATING CHANCES | 30 | 30.0 | Barcelona=19, Atletic Club=9, N/A=2 |
| DEFENDING IN ATTACKING THIRD | 22 | 20.5 | Atletic Club=15, Barcelona=7 |
| DEFENDING IN DEFENSIVE THIRD | 30 | 30.0 | Atletic Club=19, Barcelona=9, N/A=2 |
| DEFENDING IN MIDDLE THIRD | 66 | 17.6 | Atletic Club=47, Barcelona=19 |
| DEFENSIVE TRANSITION | 24 | 15.4 | Atletic Club=13, Barcelona=11 |
| GOALS | 4 | 169.3 | Barcelona=2, Atletic Club=2 |
| LONG BALL | 29 | 12.4 | Atletic Club=20, Barcelona=9 |
| PLAYERS IN THE BOX | 97 | 27.7 | Barcelona=51, Atletic Club=46 |
| PROGRESSION | 66 | 17.6 | Barcelona=47, Atletic Club=19 |
| SET PIECES | 59 | 35.0 | N/A=32, Barcelona=16, Atletic Club=11 |

Set pieces by type:

| type | n | team tags | median s |
|---|---|---|---|
| Corner Kick | 12 | Barcelona=7, Atletic Club=5 | 35.0 |
| Free Kick | 5 | Atletic Club=3, Barcelona=2 | 35.0 |
| Goal Kick | 10 | Barcelona=7, Atletic Club=3 | 35.0 |
| Kick Off (After Goal) | 4 | N/A=4 | 5.0 |
| Kick Off (Start) | 2 | N/A=2 | 5.0 |
| Throw-in | 26 | N/A=26 | 35.0 |

## Homography (tactical)
valid on 100.0%; reprojection error graded p5/p25/p50/p75/p95: 0.16 / 0.20 / 0.23 / 0.26 / 0.36; ungraded: 0.16 / 0.25 / 0.35 / 0.48 / 4.60

## Public window: what the labels look like in the tracking
_Semantics check only. These numbers are not used to set any parameter._

| GT | frames | ball visible | visible ball inside pitch | median team speed m/s | median nearest m | nearest player's team agrees |
|---|---|---|---|---|---|---|
| H | 3454 | 99.9% | 100.0% | 2.05 | 1.6 | 79.7% |
| A | 2577 | 85.0% | 99.9% | 2.47 | 1.7 | 74.9% |
| D | 1469 | 66.4% | 89.3% | 1.19 | 2.9 | - |

GT segments: 21 ; lengths (s) by label:
- H: n=8 lengths 8.1, 41.6, 28.6, 42.4, 0.8, 1.3, 3.5, 11.8
- A: n=9 lengths 6.6, 2.8, 7.4, 3.2, 20.0, 16.0, 3.6, 35.7, 7.6
- D: n=4 lengths 5.5, 20.4, 18.3, 14.5

## Baseline smooth50 over the full match
594 possessions, mean 7.2 s, median 4.6 s; 133 D segments
label shares on graded frames: {'H': '38.6%', 'A': '29.1%', 'D': '32.3%'}
