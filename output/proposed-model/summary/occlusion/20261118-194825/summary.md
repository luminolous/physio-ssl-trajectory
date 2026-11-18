# Neural-branch occlusion (Experiment 1)

Drop = full - occluded score per test subject (seed mean), mean over subjects; channel group zeroed in the neural input only, ocular proxies intact. `p_*`: Wilcoxon of the drop against 0; `rel_drop_*`: mean drop / mean full score.

| group         | config                          |   n_subjects |   drop_r |   n_pos_r |    p_r |   drop_residual_r_specific |   drop_residual_r_specific_sd |   n_pos_residual_r_specific |   p_residual_r_specific |   rel_drop_residual_r_specific |
|:--------------|:--------------------------------|-------------:|---------:|----------:|-------:|---------------------------:|------------------------------:|----------------------------:|------------------------:|-------------------------------:|
| contralateral | hybrid_finetune_jepa_unguided   |           10 |   0.0096 |         7 | 0.0371 |                     0.0388 |                        0.017  |                          10 |                  0.002  |                         0.2122 |
| contralateral | hybrid_finetune_lejepa_guided   |           10 |   0.0184 |        10 | 0.002  |                     0.0523 |                        0.0295 |                          10 |                  0.002  |                         0.3452 |
| contralateral | hybrid_finetune_lejepa_unguided |           10 |   0.017  |        10 | 0.002  |                     0.0565 |                        0.0158 |                          10 |                  0.002  |                         0.3957 |
| contralateral | hybrid_finetune_mae             |           10 |   0.015  |         9 | 0.0039 |                     0.0497 |                        0.025  |                          10 |                  0.002  |                         0.2832 |
| contralateral | hybrid_finetune_random0         |           10 |   0.0228 |        10 | 0.002  |                     0.0572 |                        0.0273 |                          10 |                  0.002  |                         0.3699 |
| contralateral | hybrid_frozen_jepa_guided       |           10 |   0.0064 |         5 | 0.3223 |                     0.0284 |                        0.0243 |                           9 |                  0.0098 |                         0.2543 |
| contralateral | hybrid_frozen_jepa_unguided     |           10 |   0.0047 |         7 | 0.1934 |                     0.0264 |                        0.0189 |                           9 |                  0.0039 |                         0.1852 |
| contralateral | hybrid_frozen_lejepa_guided     |           10 |   0.0078 |         7 | 0.084  |                     0.0429 |                        0.0358 |                           8 |                  0.0137 |                         0.3832 |
| contralateral | hybrid_frozen_lejepa_unguided   |           10 |   0.0142 |        10 | 0.002  |                     0.0524 |                        0.0278 |                          10 |                  0.002  |                         0.4301 |
| contralateral | hybrid_frozen_mae               |           10 |   0.0042 |         6 | 0.375  |                     0.0417 |                        0.0265 |                          10 |                  0.002  |                         0.3011 |
| contralateral | hybrid_frozen_random0           |           10 |   0.0047 |         6 | 0.1309 |                     0.034  |                        0.0368 |                           9 |                  0.0137 |                         0.2591 |
| contralateral | neural_finetune_jepa_unguided   |           10 |   0.0121 |         9 | 0.0039 |                     0.0389 |                        0.0177 |                          10 |                  0.002  |                         0.2055 |
| contralateral | neural_finetune_lejepa_guided   |           10 |   0.0161 |         9 | 0.0195 |                     0.0555 |                        0.0316 |                          10 |                  0.002  |                         0.3794 |
| contralateral | neural_finetune_lejepa_unguided |           10 |   0.015  |         9 | 0.0098 |                     0.0589 |                        0.0183 |                          10 |                  0.002  |                         0.4056 |
| contralateral | neural_finetune_mae             |           10 |   0.0184 |         9 | 0.0059 |                     0.0527 |                        0.0269 |                          10 |                  0.002  |                         0.3208 |
| frontal       | hybrid_finetune_jepa_unguided   |           10 |   0.0103 |         8 | 0.1055 |                     0.0464 |                        0.0336 |                          10 |                  0.002  |                         0.2543 |
| frontal       | hybrid_finetune_lejepa_guided   |           10 |   0.0072 |         4 | 1      |                     0.0094 |                        0.0161 |                           7 |                  0.1602 |                         0.0619 |
| frontal       | hybrid_finetune_lejepa_unguided |           10 |   0.0056 |         6 | 0.9219 |                     0.0088 |                        0.0129 |                           6 |                  0.084  |                         0.0619 |
| frontal       | hybrid_finetune_mae             |           10 |   0.0066 |         7 | 0.2324 |                     0.0152 |                        0.0126 |                          10 |                  0.002  |                         0.0864 |
| frontal       | hybrid_finetune_random0         |           10 |   0.007  |         7 | 0.1934 |                     0.0197 |                        0.0075 |                          10 |                  0.002  |                         0.1271 |
| frontal       | hybrid_frozen_jepa_guided       |           10 |   0.0025 |         6 | 0.375  |                     0.0132 |                        0.0173 |                           8 |                  0.084  |                         0.1185 |
| frontal       | hybrid_frozen_jepa_unguided     |           10 |   0.011  |         8 | 0.1055 |                     0.0239 |                        0.0381 |                           7 |                  0.1602 |                         0.1679 |
| frontal       | hybrid_frozen_lejepa_guided     |           10 |   0.005  |         6 | 0.3223 |                     0.0173 |                        0.0156 |                           9 |                  0.0098 |                         0.1542 |
| frontal       | hybrid_frozen_lejepa_unguided   |           10 |   0.0054 |         7 | 0.1055 |                     0.0093 |                        0.0137 |                           6 |                  0.1055 |                         0.0762 |
| frontal       | hybrid_frozen_mae               |           10 |  -0.001  |         4 | 0.6953 |                     0.0119 |                        0.0141 |                           9 |                  0.0488 |                         0.0855 |
| frontal       | hybrid_frozen_random0           |           10 |   0.0027 |         7 | 0.3223 |                     0.0091 |                        0.0156 |                           7 |                  0.1309 |                         0.0698 |
| frontal       | neural_finetune_jepa_unguided   |           10 |   0.0111 |         8 | 0.2754 |                     0.0464 |                        0.0325 |                          10 |                  0.002  |                         0.2452 |
| frontal       | neural_finetune_lejepa_guided   |           10 |   0.0087 |         5 | 0.9219 |                     0.0034 |                        0.0176 |                           4 |                  0.8457 |                         0.023  |
| frontal       | neural_finetune_lejepa_unguided |           10 |   0.0053 |         5 | 1      |                     0.0123 |                        0.0158 |                           8 |                  0.0371 |                         0.0849 |
| frontal       | neural_finetune_mae             |           10 |   0.0042 |         6 | 0.5566 |                     0.0152 |                        0.016  |                           9 |                  0.0137 |                         0.0925 |
| ipsilateral   | hybrid_finetune_jepa_unguided   |           10 |   0.0025 |         6 | 0.375  |                     0.0133 |                        0.0175 |                           8 |                  0.0273 |                         0.0726 |
| ipsilateral   | hybrid_finetune_lejepa_guided   |           10 |   0.0034 |         6 | 0.2324 |                     0.0146 |                        0.0162 |                           8 |                  0.0195 |                         0.0963 |
| ipsilateral   | hybrid_finetune_lejepa_unguided |           10 |   0.0035 |         7 | 0.2754 |                     0.0166 |                        0.0143 |                           9 |                  0.0098 |                         0.1162 |
| ipsilateral   | hybrid_finetune_mae             |           10 |   0.0036 |         7 | 0.2754 |                     0.0094 |                        0.0171 |                           6 |                  0.1934 |                         0.0534 |
| ipsilateral   | hybrid_finetune_random0         |           10 |   0.0023 |         6 | 0.5566 |                     0.0096 |                        0.0155 |                           7 |                  0.0645 |                         0.0624 |
| ipsilateral   | hybrid_frozen_jepa_guided       |           10 |  -0.0008 |         4 | 0.7695 |                     0.01   |                        0.0164 |                           7 |                  0.1055 |                         0.0893 |
| ipsilateral   | hybrid_frozen_jepa_unguided     |           10 |   0.0021 |         8 | 0.1602 |                     0.0132 |                        0.0174 |                           9 |                  0.0371 |                         0.0925 |
| ipsilateral   | hybrid_frozen_lejepa_guided     |           10 |   0.0007 |         6 | 0.6953 |                     0.0125 |                        0.0154 |                           7 |                  0.0488 |                         0.1117 |
| ipsilateral   | hybrid_frozen_lejepa_unguided   |           10 |   0.0014 |         6 | 0.375  |                     0.0255 |                        0.027  |                           8 |                  0.0098 |                         0.2093 |
| ipsilateral   | hybrid_frozen_mae               |           10 |   0.0009 |         7 | 0.2754 |                     0.0122 |                        0.0223 |                           7 |                  0.1309 |                         0.0878 |
| ipsilateral   | hybrid_frozen_random0           |           10 |   0.0008 |         5 | 0.6953 |                     0.0079 |                        0.0097 |                           8 |                  0.0273 |                         0.0605 |
| ipsilateral   | neural_finetune_jepa_unguided   |           10 |   0.004  |         6 | 0.1055 |                     0.017  |                        0.0198 |                           8 |                  0.0137 |                         0.09   |
| ipsilateral   | neural_finetune_lejepa_guided   |           10 |   0.0042 |         6 | 0.3223 |                     0.0171 |                        0.0261 |                           8 |                  0.1055 |                         0.1169 |
| ipsilateral   | neural_finetune_lejepa_unguided |           10 |   0.006  |         7 | 0.1055 |                     0.0185 |                        0.0196 |                           8 |                  0.0195 |                         0.1272 |
| ipsilateral   | neural_finetune_mae             |           10 |   0.0038 |         6 | 0.4922 |                     0.0071 |                        0.0199 |                           5 |                  0.375  |                         0.0435 |
| occipital     | hybrid_finetune_jepa_unguided   |           10 |   0.0094 |         9 | 0.0137 |                     0.0383 |                        0.0184 |                           9 |                  0.0039 |                         0.2098 |
| occipital     | hybrid_finetune_lejepa_guided   |           10 |   0.0115 |         9 | 0.0137 |                     0.0514 |                        0.0303 |                          10 |                  0.002  |                         0.3397 |
| occipital     | hybrid_finetune_lejepa_unguided |           10 |   0.0132 |        10 | 0.002  |                     0.0579 |                        0.0319 |                          10 |                  0.002  |                         0.4055 |
| occipital     | hybrid_finetune_mae             |           10 |   0.0146 |         9 | 0.0098 |                     0.0581 |                        0.0334 |                          10 |                  0.002  |                         0.3311 |
| occipital     | hybrid_finetune_random0         |           10 |   0.0142 |        10 | 0.002  |                     0.0475 |                        0.0291 |                          10 |                  0.002  |                         0.3068 |
| occipital     | hybrid_frozen_jepa_guided       |           10 |   0.0048 |         8 | 0.0371 |                     0.0083 |                        0.0186 |                           7 |                  0.375  |                         0.0739 |
| occipital     | hybrid_frozen_jepa_unguided     |           10 |   0.0049 |         7 | 0.084  |                     0.0123 |                        0.0176 |                           8 |                  0.0488 |                         0.0866 |
| occipital     | hybrid_frozen_lejepa_guided     |           10 |   0.0091 |         7 | 0.0273 |                     0.0117 |                        0.0194 |                           7 |                  0.1055 |                         0.1043 |
| occipital     | hybrid_frozen_lejepa_unguided   |           10 |   0.007  |         8 | 0.0098 |                     0.019  |                        0.0287 |                           8 |                  0.084  |                         0.1564 |
| occipital     | hybrid_frozen_mae               |           10 |   0.0036 |         9 | 0.0195 |                     0.0148 |                        0.019  |                           8 |                  0.0273 |                         0.1069 |
| occipital     | hybrid_frozen_random0           |           10 |   0.002  |         7 | 0.2754 |                     0.0083 |                        0.0088 |                           9 |                  0.0137 |                         0.0632 |
| occipital     | neural_finetune_jepa_unguided   |           10 |   0.0125 |        10 | 0.002  |                     0.0412 |                        0.0267 |                          10 |                  0.002  |                         0.2179 |
| occipital     | neural_finetune_lejepa_guided   |           10 |   0.0161 |        10 | 0.002  |                     0.0633 |                        0.0386 |                          10 |                  0.002  |                         0.4331 |
| occipital     | neural_finetune_lejepa_unguided |           10 |   0.0139 |        10 | 0.002  |                     0.0649 |                        0.0344 |                          10 |                  0.002  |                         0.447  |
| occipital     | neural_finetune_mae             |           10 |   0.0128 |         9 | 0.0059 |                     0.0581 |                        0.0367 |                          10 |                  0.002  |                         0.3539 |
| prefrontal    | hybrid_finetune_jepa_unguided   |           10 |   0.0088 |         7 | 0.0645 |                     0.0649 |                        0.0386 |                          10 |                  0.002  |                         0.3555 |
| prefrontal    | hybrid_finetune_lejepa_guided   |           10 |   0.0013 |         6 | 0.4922 |                     0.0077 |                        0.0078 |                           9 |                  0.0059 |                         0.051  |
| prefrontal    | hybrid_finetune_lejepa_unguided |           10 |   0.0016 |         5 | 0.4922 |                     0.0036 |                        0.0035 |                           9 |                  0.0195 |                         0.0251 |
| prefrontal    | hybrid_finetune_mae             |           10 |   0.0023 |         5 | 0.4316 |                     0.0181 |                        0.0139 |                           9 |                  0.0039 |                         0.1033 |
| prefrontal    | hybrid_finetune_random0         |           10 |  -0.0005 |         5 | 1      |                     0.011  |                        0.0061 |                          10 |                  0.002  |                         0.0708 |
| prefrontal    | hybrid_frozen_jepa_guided       |           10 |  -0.0005 |         3 | 0.4316 |                    -0.0037 |                        0.0054 |                           3 |                  0.0645 |                        -0.0335 |
| prefrontal    | hybrid_frozen_jepa_unguided     |           10 |   0.0092 |         8 | 0.1602 |                     0.0328 |                        0.0373 |                           8 |                  0.0137 |                         0.2302 |
| prefrontal    | hybrid_frozen_lejepa_guided     |           10 |   0.0009 |         6 | 0.8457 |                     0.0007 |                        0.003  |                           6 |                  0.4316 |                         0.0064 |
| prefrontal    | hybrid_frozen_lejepa_unguided   |           10 |  -0.0001 |         4 | 0.9219 |                    -0.0004 |                        0.0016 |                           4 |                  0.625  |                        -0.0029 |
| prefrontal    | hybrid_frozen_mae               |           10 |  -0.0006 |         5 | 0.8457 |                     0.0103 |                        0.0135 |                           8 |                  0.0488 |                         0.074  |
| prefrontal    | hybrid_frozen_random0           |           10 |   0.0004 |         5 | 0.9219 |                     0.0015 |                        0.0039 |                           7 |                  0.2754 |                         0.0113 |
| prefrontal    | neural_finetune_jepa_unguided   |           10 |   0.0114 |         9 | 0.0137 |                     0.0659 |                        0.0352 |                          10 |                  0.002  |                         0.3483 |
| prefrontal    | neural_finetune_lejepa_guided   |           10 |   0.0023 |         8 | 0.1309 |                     0.011  |                        0.0084 |                           9 |                  0.0059 |                         0.0754 |
| prefrontal    | neural_finetune_lejepa_unguided |           10 |   0.0018 |         6 | 0.4922 |                     0.0068 |                        0.0058 |                           9 |                  0.0137 |                         0.0465 |
| prefrontal    | neural_finetune_mae             |           10 |   0.0042 |         6 | 0.375  |                     0.0221 |                        0.0179 |                           9 |                  0.0098 |                         0.1344 |

## Guided - unguided (difference of drops)

| a                             | b                               | group         | metric              |   mean_diff_of_drop |   n_pos |   n_subjects |   p_wilcoxon |
|:------------------------------|:--------------------------------|:--------------|:--------------------|--------------------:|--------:|-------------:|-------------:|
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | contralateral | r                   |              0.0013 |       4 |           10 |       0.8457 |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | contralateral | residual_r_specific |             -0.0042 |       5 |           10 |       0.5566 |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | frontal       | r                   |              0.0017 |       7 |           10 |       0.375  |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | frontal       | residual_r_specific |              0.0005 |       5 |           10 |       0.9219 |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | ipsilateral   | r                   |             -0.0002 |       7 |           10 |       0.7695 |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | ipsilateral   | residual_r_specific |             -0.002  |       4 |           10 |       0.4922 |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | occipital     | r                   |             -0.0017 |       4 |           10 |       0.4922 |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | occipital     | residual_r_specific |             -0.0064 |       6 |           10 |       1      |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | prefrontal    | r                   |             -0.0003 |       6 |           10 |       1      |
| hybrid_finetune_lejepa_guided | hybrid_finetune_lejepa_unguided | prefrontal    | residual_r_specific |              0.0041 |       7 |           10 |       0.084  |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | contralateral | r                   |              0.0017 |       6 |           10 |       0.5566 |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | contralateral | residual_r_specific |              0.002  |       7 |           10 |       0.4316 |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | frontal       | r                   |             -0.0085 |       4 |           10 |       0.2324 |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | frontal       | residual_r_specific |             -0.0107 |       5 |           10 |       0.625  |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | ipsilateral   | r                   |             -0.0029 |       5 |           10 |       0.4922 |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | ipsilateral   | residual_r_specific |             -0.0032 |       3 |           10 |       0.5566 |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | occipital     | r                   |             -0.0001 |       6 |           10 |       0.6953 |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | occipital     | residual_r_specific |             -0.0041 |       5 |           10 |       0.375  |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | prefrontal    | r                   |             -0.0097 |       3 |           10 |       0.1934 |
| hybrid_frozen_jepa_guided     | hybrid_frozen_jepa_unguided     | prefrontal    | residual_r_specific |             -0.0365 |       2 |           10 |       0.0098 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | contralateral | r                   |             -0.0064 |       1 |           10 |       0.0371 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | contralateral | residual_r_specific |             -0.0095 |       3 |           10 |       0.1602 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | frontal       | r                   |             -0.0004 |       5 |           10 |       0.6953 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | frontal       | residual_r_specific |              0.008  |       6 |           10 |       0.1934 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | ipsilateral   | r                   |             -0.0007 |       5 |           10 |       1      |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | ipsilateral   | residual_r_specific |             -0.013  |       3 |           10 |       0.0645 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | occipital     | r                   |              0.0021 |       5 |           10 |       0.625  |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | occipital     | residual_r_specific |             -0.0074 |       4 |           10 |       0.5566 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | prefrontal    | r                   |              0.001  |       6 |           10 |       0.7695 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_lejepa_unguided   | prefrontal    | residual_r_specific |              0.0011 |       7 |           10 |       0.2324 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | contralateral | r                   |              0.001  |       6 |           10 |       0.5566 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | contralateral | residual_r_specific |             -0.0035 |       4 |           10 |       0.5566 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | frontal       | r                   |              0.0034 |       4 |           10 |       0.8457 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | frontal       | residual_r_specific |             -0.009  |       3 |           10 |       0.1055 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | ipsilateral   | r                   |             -0.0018 |       4 |           10 |       0.4922 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | ipsilateral   | residual_r_specific |             -0.0014 |       5 |           10 |       0.7695 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | occipital     | r                   |              0.0022 |       6 |           10 |       0.2754 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | occipital     | residual_r_specific |             -0.0016 |       6 |           10 |       0.6953 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | prefrontal    | r                   |              0.0005 |       5 |           10 |       0.7695 |
| neural_finetune_lejepa_guided | neural_finetune_lejepa_unguided | prefrontal    | residual_r_specific |              0.0043 |       9 |           10 |       0.0039 |

## Reproducibility vs stage 1

| config                        |   seed |   r_this |   r_stage1 |   rrs_this |   rrs_stage1 |
|:------------------------------|-------:|---------:|-----------:|-----------:|-------------:|
| hybrid_finetune_lejepa_guided |      1 |   0.8142 |     0.8116 |     0.1514 |       0.1439 |
| hybrid_finetune_random0       |      1 |   0.8161 |     0.8117 |     0.1546 |       0.1424 |
| hybrid_frozen_jepa_guided     |      1 |   0.8045 |     0.8045 |     0.1117 |       0.1119 |
| hybrid_frozen_jepa_unguided   |      1 |   0.8093 |     0.8097 |     0.1424 |       0.1463 |
| hybrid_frozen_lejepa_guided   |      1 |   0.8055 |     0.8056 |     0.1119 |       0.1119 |
| hybrid_frozen_lejepa_unguided |      1 |   0.8116 |     0.8116 |     0.1217 |       0.1218 |
| hybrid_frozen_mae             |      1 |   0.8033 |     0.8035 |     0.1386 |       0.1342 |
| hybrid_frozen_random0         |      1 |   0.8102 |     0.8105 |     0.1311 |       0.1322 |
| neural_finetune_lejepa_guided |      1 |   0.8194 |     0.8194 |     0.1462 |       0.1434 |
