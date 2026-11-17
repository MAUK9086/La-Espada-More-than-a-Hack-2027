# E64a: models trained on the 8 calibration matches, tested on the 20 SkillCorner matches

SkillCorner open data (A-League 2024/25, broadcast tracking at 10 fps interpolated to 25 fps, the ball only where it was detected, no simulated degradation). Truth: SkillCorner's possession, with the 2 s control rule and at touch level. No model saw these matches. Δ and the interval are against p3-refined, paired over the 20 matches (bootstrap over matches, 5–95 %, 10000 draws).

## Truth: 2 s rule

| system | spellF1 (20 matches) | Δ | matches better | 5–95 % of Δ | min – max | pred / true possessions |
|---|---|---|---|---|---|---|
| p3-refined (submission), α 0.75 standard | 0.4483 | +0.0000 | 0/20 | +0.0000 to +0.0000 | 0.3723 – 0.5266 | 6121/4975 |
| base ensemble of 5, std α 0.75 | 0.4363 | -0.0120 | 7/20 | -0.0178 to -0.0063 | 0.3502 – 0.5249 | 6272/4975 |
| base ensemble of 5, std α 1.5 | 0.4235 | -0.0247 | 1/20 | -0.0306 to -0.0187 | 0.3460 – 0.5054 | 6574/4975 |
| base ensemble of 5, R1 α 1.5 | 0.4279 | -0.0204 | 2/20 | -0.0265 to -0.0144 | 0.3511 – 0.5150 | 6415/4975 |
| aug4_e10 ensemble of 3, std α 0.75 | 0.4316 | -0.0167 | 3/20 | -0.0225 to -0.0110 | 0.3556 – 0.4976 | 6197/4975 |
| aug4_e10 ensemble of 3, std α 1.5 | 0.4270 | -0.0213 | 1/20 | -0.0267 to -0.0157 | 0.3502 – 0.4834 | 6343/4975 |
| aug4_e10 ensemble of 3, R1 α 1.5 | 0.4306 | -0.0177 | 1/20 | -0.0229 to -0.0125 | 0.3579 – 0.4906 | 6219/4975 |
| k4_e10 ensemble of 3, std α 0.75 | 0.4828 | +0.0345 | 19/20 | +0.0293 to +0.0393 | 0.3952 – 0.5522 | 5684/4975 |
| k4_e10 ensemble of 3, std α 1.5 | 0.4784 | +0.0301 | 19/20 | +0.0248 to +0.0351 | 0.3938 – 0.5431 | 5770/4975 |
| k4_e10 ensemble of 3, R1 α 1.5 | 0.4836 | +0.0353 | 19/20 | +0.0296 to +0.0405 | 0.3931 – 0.5495 | 5652/4975 |
| mix8 ensemble of 3, std α 0.75 | 0.4403 | -0.0080 | 7/20 | -0.0131 to -0.0026 | 0.3489 – 0.5077 | 6096/4975 |
| mix8 ensemble of 3, std α 1.5 | 0.4275 | -0.0208 | 1/20 | -0.0270 to -0.0147 | 0.3490 – 0.4926 | 6392/4975 |
| mix8 ensemble of 3, R1 α 1.5 | 0.4327 | -0.0156 | 4/20 | -0.0218 to -0.0096 | 0.3539 – 0.4995 | 6264/4975 |

## Truth: touch level

| system | spellF1 (20 matches) | Δ | matches better | 5–95 % of Δ | min – max | pred / true possessions |
|---|---|---|---|---|---|---|
| p3-refined (submission), α 0.75 standard | 0.3854 | +0.0000 | 0/20 | +0.0000 to +0.0000 | 0.3398 – 0.4409 | 6121/6499 |
| base ensemble of 5, std α 0.75 | 0.3757 | -0.0097 | 6/20 | -0.0146 to -0.0049 | 0.3173 – 0.4439 | 6272/6499 |
| base ensemble of 5, std α 1.5 | 0.3663 | -0.0192 | 1/20 | -0.0242 to -0.0143 | 0.3098 – 0.4388 | 6574/6499 |
| base ensemble of 5, R1 α 1.5 | 0.3688 | -0.0167 | 2/20 | -0.0214 to -0.0118 | 0.3088 – 0.4394 | 6415/6499 |
| aug4_e10 ensemble of 3, std α 0.75 | 0.3706 | -0.0148 | 5/20 | -0.0195 to -0.0101 | 0.3194 – 0.4426 | 6197/6499 |
| aug4_e10 ensemble of 3, std α 1.5 | 0.3675 | -0.0179 | 3/20 | -0.0231 to -0.0126 | 0.3055 – 0.4356 | 6343/6499 |
| aug4_e10 ensemble of 3, R1 α 1.5 | 0.3692 | -0.0163 | 2/20 | -0.0209 to -0.0116 | 0.3096 – 0.4324 | 6219/6499 |
| k4_e10 ensemble of 3, std α 0.75 | 0.4144 | +0.0289 | 19/20 | +0.0242 to +0.0332 | 0.3451 – 0.4771 | 5684/6499 |
| k4_e10 ensemble of 3, std α 1.5 | 0.4111 | +0.0257 | 19/20 | +0.0209 to +0.0303 | 0.3437 – 0.4796 | 5770/6499 |
| k4_e10 ensemble of 3, R1 α 1.5 | 0.4137 | +0.0283 | 18/20 | +0.0231 to +0.0329 | 0.3434 – 0.4839 | 5652/6499 |
| mix8 ensemble of 3, std α 0.75 | 0.3779 | -0.0076 | 7/20 | -0.0127 to -0.0024 | 0.3043 – 0.4500 | 6096/6499 |
| mix8 ensemble of 3, std α 1.5 | 0.3680 | -0.0174 | 4/20 | -0.0228 to -0.0119 | 0.3032 – 0.4374 | 6392/6499 |
| mix8 ensemble of 3, R1 α 1.5 | 0.3710 | -0.0144 | 5/20 | -0.0199 to -0.0090 | 0.3055 – 0.4431 | 6264/6499 |

## Truth: 2 s rule, the truth's dead-ball frames imposed on every prediction

| system | spellF1 (20 matches) | Δ | matches better | 5–95 % of Δ | min – max | pred / true possessions |
|---|---|---|---|---|---|---|
| p3-refined (submission), α 0.75 standard | 0.5517 | +0.0000 | 0/20 | +0.0000 to +0.0000 | 0.4519 – 0.6343 | 5387/4975 |
| base ensemble of 5, std α 0.75 | 0.5524 | +0.0006 | 7/20 | -0.0050 to +0.0066 | 0.4308 – 0.6418 | 5511/4975 |
| base ensemble of 5, std α 1.5 | 0.5481 | -0.0036 | 6/20 | -0.0097 to +0.0028 | 0.4289 – 0.6352 | 5587/4975 |
| base ensemble of 5, R1 α 1.5 | 0.5528 | +0.0010 | 10/20 | -0.0047 to +0.0073 | 0.4351 – 0.6400 | 5476/4975 |
| aug4_e10 ensemble of 3, std α 0.75 | 0.5564 | +0.0047 | 11/20 | -0.0005 to +0.0101 | 0.4385 – 0.6434 | 5452/4975 |
| aug4_e10 ensemble of 3, std α 1.5 | 0.5547 | +0.0029 | 11/20 | -0.0034 to +0.0094 | 0.4349 – 0.6456 | 5515/4975 |
| aug4_e10 ensemble of 3, R1 α 1.5 | 0.5596 | +0.0078 | 12/20 | +0.0024 to +0.0135 | 0.4434 – 0.6456 | 5404/4975 |
| k4_e10 ensemble of 3, std α 0.75 | 0.5457 | -0.0060 | 5/20 | -0.0105 to -0.0012 | 0.4413 – 0.6241 | 5432/4975 |
| k4_e10 ensemble of 3, std α 1.5 | 0.5432 | -0.0086 | 6/20 | -0.0140 to -0.0030 | 0.4393 – 0.6194 | 5497/4975 |
| k4_e10 ensemble of 3, R1 α 1.5 | 0.5482 | -0.0035 | 7/20 | -0.0089 to +0.0021 | 0.4381 – 0.6234 | 5400/4975 |
| mix8 ensemble of 3, std α 0.75 | 0.5505 | -0.0012 | 9/20 | -0.0076 to +0.0057 | 0.4273 – 0.6488 | 5510/4975 |
| mix8 ensemble of 3, std α 1.5 | 0.5474 | -0.0043 | 8/20 | -0.0112 to +0.0025 | 0.4302 – 0.6392 | 5568/4975 |
| mix8 ensemble of 3, R1 α 1.5 | 0.5535 | +0.0017 | 11/20 | -0.0045 to +0.0081 | 0.4343 – 0.6444 | 5460/4975 |

Matches: skc_1874553, skc_1886347, skc_1899585, skc_1925299, skc_1927964, skc_1953632, skc_1959846, skc_1986691, skc_1996435, skc_1996436, skc_2006229, skc_2006363, skc_2007448, skc_2007721, skc_2010085, skc_2011166, skc_2013725, skc_2015213, skc_2016236, skc_2017461.
