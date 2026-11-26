# Head-floor rrs check

Mean over the 10 LOSO test subjects. rrs = residual_r - residual_r_other_trial; window_effect = residual_r (whole scored window) - same-trial correlation on the shared bins; matched = same-trial - other-trial, both on the shared bins. A rigid predictor is one function of time since LED-on.

## A. Rigid predictors

| group              | window      |   residual_r |   residual_r_other_trial |     rrs |   own_shared |   window_effect |   matched |   shared_fraction |   rrs_sd |   matched_abs_max |   n_lifts_excluded |
|:-------------------|:------------|-------------:|-------------------------:|--------:|-------------:|----------------:|----------:|------------------:|---------:|------------------:|-------------------:|
| constant_0         | fixed       |      -0.0446 |                  -0.0446 | -0      |      -0.0446 |         -0      |         0 |            1      |   0      |                 0 |                 45 |
| constant_0         | fixed_end   |      -0.0603 |                  -0.0616 |  0.0013 |      -0.0619 |          0.0016 |         0 |            0.9865 |   0.0086 |                 0 |                 45 |
| constant_0         | fixed_start |       0.0135 |                   0.0039 |  0.0096 |       0.0041 |          0.0094 |         0 |            0.9518 |   0.0238 |                 0 |                  0 |
| constant_0         | scored      |       0.0023 |                  -0.011  |  0.0133 |      -0.0108 |          0.013  |         0 |            0.9372 |   0.0282 |                 0 |                  0 |
| cue_shift_+0.2s    | fixed       |       0.007  |                   0.007  | -0      |       0.007  |          0      |         0 |            1      |   0      |                 0 |                 45 |
| cue_shift_+0.2s    | fixed_end   |       0.0129 |                   0.0126 |  0.0003 |       0.0124 |          0.0004 |         0 |            0.9865 |   0.0026 |                 0 |                 45 |
| cue_shift_+0.2s    | fixed_start |      -0.0475 |                  -0.02   | -0.0275 |      -0.0205 |         -0.027  |         0 |            0.9518 |   0.0496 |                 0 |                  0 |
| cue_shift_+0.2s    | scored      |      -0.0469 |                  -0.0181 | -0.0288 |      -0.0187 |         -0.0281 |         0 |            0.9372 |   0.0503 |                 0 |                  0 |
| cue_shift_-0.2s    | fixed       |      -0.0067 |                  -0.0067 |  0      |      -0.0067 |          0      |         0 |            1      |   0      |                 0 |                 45 |
| cue_shift_-0.2s    | fixed_end   |      -0.0166 |                  -0.0204 |  0.0039 |      -0.0204 |          0.0038 |         0 |            0.9865 |   0.0096 |                 0 |                 45 |
| cue_shift_-0.2s    | fixed_start |       0.0521 |                   0.0173 |  0.0348 |       0.0178 |          0.0343 |         0 |            0.9518 |   0.0456 |                 0 |                  0 |
| cue_shift_-0.2s    | scored      |       0.0506 |                   0.0093 |  0.0413 |       0.01   |          0.0406 |         0 |            0.9372 |   0.0497 |                 0 |                  0 |
| head_floor         | fixed       |      -0.0367 |                  -0.0367 |  0      |      -0.0367 |          0      |         0 |            1      |   0      |                 0 |                 45 |
| head_floor         | fixed_end   |      -0.037  |                  -0.0349 | -0.0022 |      -0.0349 |         -0.0021 |         0 |            0.9865 |   0.0057 |                 0 |                 45 |
| head_floor         | fixed_start |       0.0638 |                   0.03   |  0.0338 |       0.0306 |          0.0332 |         0 |            0.9518 |   0.0405 |                 0 |                  0 |
| head_floor         | scored      |       0.0627 |                   0.0295 |  0.0333 |       0.0301 |          0.0326 |         0 |            0.9372 |   0.0398 |                 0 |                  0 |
| random_smooth (20) | fixed       |      -0.0047 |                  -0.0047 | -0      |      -0.0047 |          0      |         0 |            1      |   0      |                 0 |                 45 |
| random_smooth (20) | fixed_end   |      -0.0038 |                  -0.004  |  0.0002 |      -0.004  |          0.0001 |         0 |            0.9865 |   0.0005 |                 0 |                 45 |
| random_smooth (20) | fixed_start |      -0.0007 |                   0.0037 | -0.0044 |       0.0037 |         -0.0044 |         0 |            0.9518 |   0.004  |                 0 |                  0 |
| random_smooth (20) | scored      |      -0.0004 |                   0.0039 | -0.0043 |       0.0039 |         -0.0043 |         0 |            0.9372 |   0.004  |                 0 |                  0 |
| unconditional_mean | fixed       |       0.2274 |                   0.2274 | -0      |       0.2274 |          0      |         0 |            1      |   0      |                 0 |                 45 |
| unconditional_mean | fixed_end   |       0.2238 |                   0.2231 |  0.0006 |       0.2233 |          0.0005 |         0 |            0.9865 |   0.0026 |                 0 |                 45 |
| unconditional_mean | fixed_start |       0.2738 |                   0.1017 |  0.1721 |       0.1008 |          0.1729 |         0 |            0.9518 |   0.0581 |                 0 |                  0 |
| unconditional_mean | scored      |       0.2724 |                   0.1006 |  0.1718 |       0.0998 |          0.1726 |         0 |            0.9372 |   0.0566 |                 0 |                  0 |

Random smooth functions, scored window, mean rrs per function over subjects: min -0.0413, median -0.0097, max 0.0390; 6 / 20 positive.

## B. Experiment 1 configurations (scored window, seed-averaged)

| branches   | setting   | encoder         |    rrs |   matched |   window_effect |   residual_r |   own_shared |   shared_fraction |   rrs_sd |   matched_sd |
|:-----------|:----------|:----------------|-------:|----------:|----------------:|-------------:|-------------:|------------------:|---------:|-------------:|
| neural     | finetune  | jepa_unguided   | 0.1893 |    0.1623 |          0.0277 |       0.3019 |       0.2742 |            0.9372 |   0.0769 |       0.0759 |
| hybrid     | finetune  | jepa_unguided   | 0.1826 |    0.16   |          0.0237 |       0.293  |       0.2692 |            0.9372 |   0.0729 |       0.0761 |
| hybrid     | finetune  | mae             | 0.1754 |    0.1472 |          0.0285 |       0.2776 |       0.2491 |            0.9372 |   0.0643 |       0.0657 |
| neural     | finetune  | random0         | 0.1644 |    0.1377 |          0.0278 |       0.2765 |       0.2487 |            0.9372 |   0.052  |       0.0581 |
| neural     | finetune  | mae             | 0.1642 |    0.1439 |          0.0209 |       0.2533 |       0.2324 |            0.9372 |   0.0676 |       0.0741 |
| hybrid     | frozen    | jepa_unguided   | 0.1586 |    0.1342 |          0.0248 |       0.239  |       0.2142 |            0.9372 |   0.0589 |       0.0599 |
| hybrid     | finetune  | random0         | 0.1494 |    0.1308 |          0.0187 |       0.2519 |       0.2332 |            0.9372 |   0.047  |       0.0547 |
| neural     | finetune  | lejepa_guided   | 0.1459 |    0.1292 |          0.0166 |       0.274  |       0.2573 |            0.9372 |   0.061  |       0.0595 |
| neural     | finetune  | lejepa_unguided | 0.1452 |    0.1275 |          0.0176 |       0.243  |       0.2254 |            0.9372 |   0.0576 |       0.063  |
| hybrid     | finetune  | lejepa_unguided | 0.1427 |    0.1243 |          0.0173 |       0.2474 |       0.2301 |            0.9372 |   0.056  |       0.0578 |
| hybrid     | finetune  | lejepa_guided   | 0.1401 |    0.1214 |          0.0185 |       0.2401 |       0.2215 |            0.9372 |   0.0561 |       0.0624 |
| neural     | frozen    | jepa_unguided   | 0.1396 |    0.1195 |          0.0208 |       0.2154 |       0.1946 |            0.9372 |   0.0528 |       0.0527 |
| hybrid     | frozen    | mae             | 0.1301 |    0.11   |          0.0207 |       0.1801 |       0.1594 |            0.9372 |   0.05   |       0.0557 |
| neural     | frozen    | mae             | 0.13   |    0.1029 |          0.0271 |       0.1875 |       0.1604 |            0.9372 |   0.0482 |       0.0502 |
| hybrid     | frozen    | random0         | 0.1294 |    0.107  |          0.0225 |       0.1892 |       0.1667 |            0.9372 |   0.044  |       0.0502 |
| hybrid     | frozen    | lejepa_unguided | 0.1153 |    0.1005 |          0.0138 |       0.1914 |       0.1776 |            0.9372 |   0.0389 |       0.0458 |
| hybrid     | frozen    | jepa_guided     | 0.1099 |    0.0996 |          0.0105 |       0.1658 |       0.1553 |            0.9372 |   0.051  |       0.0545 |
| hybrid     | frozen    | lejepa_guided   | 0.1087 |    0.0942 |          0.0134 |       0.1873 |       0.1739 |            0.9372 |   0.0427 |       0.0489 |
| ocular     | frozen    | noenc           | 0.1047 |    0.096  |          0.0097 |       0.1301 |       0.1205 |            0.9372 |   0.0619 |       0.0612 |
| neural     | frozen    | random0         | 0.0898 |    0.0752 |          0.0143 |       0.1412 |       0.1269 |            0.9372 |   0.0504 |       0.0404 |
| neural     | frozen    | lejepa_guided   | 0.0843 |    0.078  |          0.0058 |       0.1549 |       0.1491 |            0.9372 |   0.043  |       0.039  |
| neural     | frozen    | lejepa_unguided | 0.0827 |    0.0788 |          0.0035 |       0.1629 |       0.1594 |            0.9372 |   0.0402 |       0.0384 |
| neural     | frozen    | jepa_guided     | 0.0508 |    0.0514 |         -0.0019 |       0.0924 |       0.0944 |            0.9372 |   0.0564 |       0.0427 |

## C. Paired comparisons, rrs vs matched

| comparison                         | a                               |   mean_diff_matched |   mean_diff_rrs |   n_pos_matched |   n_pos_rrs |   p_wilcoxon_matched |   p_wilcoxon_rrs |
|:-----------------------------------|:--------------------------------|--------------------:|----------------:|----------------:|------------:|---------------------:|-----------------:|
| config - head floor                | hybrid/finetune/jepa_unguided   |              0.16   |          0.1493 |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | hybrid/finetune/lejepa_guided   |              0.1214 |          0.1068 |              10 |           9 |               0.002  |           0.0039 |
| config - head floor                | hybrid/finetune/lejepa_unguided |              0.1243 |          0.1094 |              10 |           9 |               0.002  |           0.0039 |
| config - head floor                | hybrid/finetune/mae             |              0.1472 |          0.1421 |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | hybrid/finetune/random0         |              0.1308 |          0.1161 |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | hybrid/frozen/jepa_guided       |              0.0996 |          0.0766 |              10 |           9 |               0.002  |           0.0098 |
| config - head floor                | hybrid/frozen/jepa_unguided     |              0.1342 |          0.1253 |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | hybrid/frozen/lejepa_guided     |              0.0942 |          0.0754 |              10 |           9 |               0.002  |           0.0098 |
| config - head floor                | hybrid/frozen/lejepa_unguided   |              0.1005 |          0.082  |              10 |           9 |               0.002  |           0.0098 |
| config - head floor                | hybrid/frozen/mae               |              0.11   |          0.0969 |              10 |           9 |               0.002  |           0.0059 |
| config - head floor                | hybrid/frozen/random0           |              0.107  |          0.0961 |              10 |           9 |               0.002  |           0.0039 |
| config - head floor                | neural/finetune/jepa_unguided   |              0.1623 |          0.156  |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | neural/finetune/lejepa_guided   |              0.1292 |          0.1126 |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | neural/finetune/lejepa_unguided |              0.1275 |          0.112  |              10 |           9 |               0.002  |           0.0039 |
| config - head floor                | neural/finetune/mae             |              0.1439 |          0.1309 |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | neural/finetune/random0         |              0.1377 |          0.1312 |              10 |          10 |               0.002  |           0.002  |
| config - head floor                | neural/frozen/jepa_guided       |              0.0514 |          0.0175 |              10 |           7 |               0.002  |           0.2754 |
| config - head floor                | neural/frozen/jepa_unguided     |              0.1195 |          0.1063 |              10 |           9 |               0.002  |           0.0039 |
| config - head floor                | neural/frozen/lejepa_guided     |              0.078  |          0.051  |              10 |           8 |               0.002  |           0.0273 |
| config - head floor                | neural/frozen/lejepa_unguided   |              0.0788 |          0.0494 |              10 |           8 |               0.002  |           0.0273 |
| config - head floor                | neural/frozen/mae               |              0.1029 |          0.0967 |              10 |           9 |               0.002  |           0.0039 |
| config - head floor                | neural/frozen/random0           |              0.0752 |          0.0565 |              10 |           8 |               0.002  |           0.0273 |
| config - head floor                | ocular/frozen/noenc             |              0.096  |          0.0714 |              10 |           9 |               0.002  |           0.0273 |
| hybrid - ocular-only               | hybrid/finetune/jepa_unguided   |              0.064  |          0.0779 |               9 |           9 |               0.0039 |           0.0059 |
| hybrid - ocular-only               | hybrid/finetune/lejepa_guided   |              0.0253 |          0.0354 |               6 |           9 |               0.1934 |           0.0371 |
| hybrid - ocular-only               | hybrid/finetune/lejepa_unguided |              0.0283 |          0.0379 |               7 |           8 |               0.0488 |           0.0645 |
| hybrid - ocular-only               | hybrid/finetune/mae             |              0.0512 |          0.0707 |               8 |           9 |               0.0137 |           0.0059 |
| hybrid - ocular-only               | hybrid/finetune/random0         |              0.0347 |          0.0447 |               8 |           8 |               0.0098 |           0.0371 |
| hybrid - ocular-only               | hybrid/frozen/jepa_guided       |              0.0035 |          0.0051 |               5 |           5 |               0.7695 |           0.625  |
| hybrid - ocular-only               | hybrid/frozen/jepa_unguided     |              0.0382 |          0.0538 |               8 |           9 |               0.0098 |           0.0137 |
| hybrid - ocular-only               | hybrid/frozen/lejepa_guided     |             -0.0018 |          0.0039 |               4 |           4 |               0.8457 |           1      |
| hybrid - ocular-only               | hybrid/frozen/lejepa_unguided   |              0.0045 |          0.0106 |               5 |           5 |               0.8457 |           0.6953 |
| hybrid - ocular-only               | hybrid/frozen/mae               |              0.0139 |          0.0254 |               5 |           6 |               0.375  |           0.1934 |
| hybrid - ocular-only               | hybrid/frozen/random0           |              0.011  |          0.0247 |               7 |           7 |               0.375  |           0.1934 |
| neural - ocular-only               | neural/finetune/jepa_unguided   |              0.0662 |          0.0846 |              10 |           9 |               0.002  |           0.0039 |
| neural - ocular-only               | neural/finetune/lejepa_guided   |              0.0331 |          0.0412 |               7 |           8 |               0.0488 |           0.0371 |
| neural - ocular-only               | neural/finetune/lejepa_unguided |              0.0315 |          0.0405 |               7 |           8 |               0.084  |           0.0371 |
| neural - ocular-only               | neural/finetune/mae             |              0.0478 |          0.0595 |               8 |           8 |               0.0273 |           0.0371 |
| neural - ocular-only               | neural/finetune/random0         |              0.0417 |          0.0597 |               9 |           9 |               0.0098 |           0.0098 |
| neural - ocular-only               | neural/frozen/jepa_guided       |             -0.0447 |         -0.0539 |               3 |           2 |               0.0371 |           0.0488 |
| neural - ocular-only               | neural/frozen/jepa_unguided     |              0.0234 |          0.0349 |               6 |           6 |               0.2754 |           0.1309 |
| neural - ocular-only               | neural/frozen/lejepa_guided     |             -0.018  |         -0.0204 |               3 |           4 |               0.4316 |           0.4316 |
| neural - ocular-only               | neural/frozen/lejepa_unguided   |             -0.0172 |         -0.022  |               3 |           4 |               0.375  |           0.3223 |
| neural - ocular-only               | neural/frozen/mae               |              0.0069 |          0.0252 |               6 |           5 |               0.4922 |           0.2754 |
| neural - ocular-only               | neural/frozen/random0           |             -0.0208 |         -0.0149 |               3 |           4 |               0.3223 |           0.6953 |
| pretrained - random-init / scratch | hybrid/finetune/jepa_unguided   |              0.0292 |          0.0332 |               9 |           8 |               0.0039 |           0.0098 |
| pretrained - random-init / scratch | hybrid/finetune/lejepa_guided   |             -0.0094 |         -0.0093 |               3 |           4 |               0.1934 |           0.3223 |
| pretrained - random-init / scratch | hybrid/finetune/lejepa_unguided |             -0.0065 |         -0.0067 |               4 |           4 |               0.2324 |           0.4922 |
| pretrained - random-init / scratch | hybrid/finetune/mae             |              0.0165 |          0.026  |               8 |           9 |               0.0645 |           0.0273 |
| pretrained - random-init / scratch | hybrid/frozen/jepa_guided       |             -0.0075 |         -0.0195 |               3 |           3 |               0.1602 |           0.084  |
| pretrained - random-init / scratch | hybrid/frozen/jepa_unguided     |              0.0272 |          0.0292 |               9 |           9 |               0.0195 |           0.0137 |
| pretrained - random-init / scratch | hybrid/frozen/lejepa_guided     |             -0.0128 |         -0.0207 |               2 |           3 |               0.0488 |           0.0273 |
| pretrained - random-init / scratch | hybrid/frozen/lejepa_unguided   |             -0.0065 |         -0.0141 |               3 |           2 |               0.4316 |           0.1309 |
| pretrained - random-init / scratch | hybrid/frozen/mae               |              0.0029 |          0.0007 |               4 |           3 |               0.7695 |           0.625  |
| pretrained - random-init / scratch | neural/finetune/jepa_unguided   |              0.0246 |          0.0248 |               7 |           7 |               0.0273 |           0.1055 |
| pretrained - random-init / scratch | neural/finetune/lejepa_guided   |             -0.0085 |         -0.0185 |               2 |           4 |               0.1934 |           0.1055 |
| pretrained - random-init / scratch | neural/finetune/lejepa_unguided |             -0.0102 |         -0.0192 |               3 |           2 |               0.1602 |           0.0645 |
| pretrained - random-init / scratch | neural/finetune/mae             |              0.0062 |         -0.0002 |               6 |           4 |               0.6953 |           1      |
| pretrained - random-init / scratch | neural/frozen/jepa_guided       |             -0.0239 |         -0.039  |               1 |           2 |               0.0645 |           0.0273 |
| pretrained - random-init / scratch | neural/frozen/jepa_unguided     |              0.0442 |          0.0498 |               9 |          10 |               0.0039 |           0.002  |
| pretrained - random-init / scratch | neural/frozen/lejepa_guided     |              0.0028 |         -0.0055 |               6 |           5 |               0.5566 |           0.6953 |
| pretrained - random-init / scratch | neural/frozen/lejepa_unguided   |              0.0036 |         -0.0071 |               5 |           4 |               0.9219 |           0.4922 |
| pretrained - random-init / scratch | neural/frozen/mae               |              0.0277 |          0.0402 |               8 |          10 |               0.0098 |           0.002  |

Reproduction of saved rrs: max |difference| models 0.00e+00, head floor 9.86e-09.
