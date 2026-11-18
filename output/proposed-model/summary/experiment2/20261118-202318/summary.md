# Experiment 2: physiological branches (RQ2)

Setting fine-tune, proposed head, proposed encoder (lejepa_guided); ocular-only has no encoder. Test scores per LOSO subject, mean over 3 decoder seeds, then mean over 10 subjects. `head_floor`: the warp head at initialization (cue phase prior); `template_cue` is scored on the same samples. `*_seed_sd`: SD over seeds of the subject-mean score.

| branch       |      r |     r2 |   nrmse |   residual_r_specific |   r_seed_sd |   residual_r_specific_seed_sd |
|:-------------|-------:|-------:|--------:|----------------------:|------------:|------------------------------:|
| neural       | 0.8184 | 0.5267 |  0.2143 |                0.1459 |      0.0013 |                        0.0035 |
| hybrid       | 0.815  | 0.5195 |  0.2163 |                0.1401 |      0.003  |                        0.0045 |
| ocular       | 0.8045 | 0.5183 |  0.2182 |                0.1047 |      0.0067 |                        0.015  |
| head_floor   | 0.8039 | 0.5284 |  0.2162 |                0.0333 |    nan      |                      nan      |
| template_cue | 0.8027 | 0.5244 |  0.2164 |              nan      |    nan      |                      nan      |

## Paired comparisons (Wilcoxon over subjects)

| a      | b            | metric              |   mean_diff |   n_pos |   n_subjects |   p_wilcoxon |
|:-------|:-------------|:--------------------|------------:|--------:|-------------:|-------------:|
| hybrid | neural       | r                   |     -0.0034 |       6 |           10 |       0.8457 |
| hybrid | neural       | r2                  |     -0.0072 |       5 |           10 |       0.7695 |
| hybrid | neural       | nrmse               |      0.002  |       7 |           10 |       0.4316 |
| hybrid | neural       | residual_r_specific |     -0.0058 |       5 |           10 |       0.6953 |
| hybrid | ocular       | r                   |      0.0105 |       5 |           10 |       0.5566 |
| hybrid | ocular       | r2                  |      0.0012 |       5 |           10 |       0.7695 |
| hybrid | ocular       | nrmse               |     -0.0018 |       5 |           10 |       1      |
| hybrid | ocular       | residual_r_specific |      0.0354 |       9 |           10 |       0.0371 |
| neural | ocular       | r                   |      0.0139 |       7 |           10 |       0.084  |
| neural | ocular       | r2                  |      0.0084 |       7 |           10 |       0.6953 |
| neural | ocular       | nrmse               |     -0.0039 |       3 |           10 |       0.2754 |
| neural | ocular       | residual_r_specific |      0.0412 |       8 |           10 |       0.0371 |
| neural | template_cue | r                   |      0.0157 |       5 |           10 |       0.4922 |
| neural | template_cue | r2                  |      0.0023 |       4 |           10 |       0.625  |
| neural | template_cue | nrmse               |     -0.0022 |       5 |           10 |       1      |
| hybrid | template_cue | r                   |      0.0123 |       3 |           10 |       0.8457 |
| hybrid | template_cue | r2                  |     -0.0049 |       3 |           10 |       0.4922 |
| hybrid | template_cue | nrmse               |     -0.0001 |       7 |           10 |       0.6953 |
| ocular | template_cue | r                   |      0.0018 |       3 |           10 |       0.6953 |
| ocular | template_cue | r2                  |     -0.0061 |       4 |           10 |       0.4316 |
| ocular | template_cue | nrmse               |      0.0017 |       6 |           10 |       0.4316 |
| neural | head_floor   | r                   |      0.0145 |       5 |           10 |       0.5566 |
| neural | head_floor   | r2                  |     -0.0016 |       4 |           10 |       0.625  |
| neural | head_floor   | nrmse               |     -0.0019 |       5 |           10 |       1      |
| neural | head_floor   | residual_r_specific |      0.1126 |      10 |           10 |       0.002  |
| hybrid | head_floor   | r                   |      0.0111 |       3 |           10 |       0.9219 |
| hybrid | head_floor   | r2                  |     -0.0089 |       3 |           10 |       0.4316 |
| hybrid | head_floor   | nrmse               |      0.0002 |       7 |           10 |       0.625  |
| hybrid | head_floor   | residual_r_specific |      0.1068 |       9 |           10 |       0.0039 |
| ocular | head_floor   | r                   |      0.0006 |       5 |           10 |       0.625  |
| ocular | head_floor   | r2                  |     -0.0101 |       4 |           10 |       0.3223 |
| ocular | head_floor   | nrmse               |      0.002  |       6 |           10 |       0.375  |
| ocular | head_floor   | residual_r_specific |      0.0714 |       9 |           10 |       0.0273 |

## Neural-branch group occlusion (seed 1, `exp1-occl`)

Drop = full - occluded per test subject; the group is zeroed in the neural input only, ocular proxies intact. `p_*`: Wilcoxon of the drop against 0; `rel_drop_*`: mean drop / mean full score.

| group         | branch   |   n_subjects |   drop_residual_r_specific |   n_pos_residual_r_specific |   p_residual_r_specific |   rel_drop_residual_r_specific |   drop_r |   n_pos_r |    p_r |
|:--------------|:---------|-------------:|---------------------------:|----------------------------:|------------------------:|-------------------------------:|---------:|----------:|-------:|
| prefrontal    | hybrid   |           10 |                     0.0077 |                           9 |                  0.0059 |                         0.051  |   0.0013 |         6 | 0.4922 |
| prefrontal    | neural   |           10 |                     0.011  |                           9 |                  0.0059 |                         0.0754 |   0.0023 |         8 | 0.1309 |
| frontal       | hybrid   |           10 |                     0.0094 |                           7 |                  0.1602 |                         0.0619 |   0.0072 |         4 | 1      |
| frontal       | neural   |           10 |                     0.0034 |                           4 |                  0.8457 |                         0.023  |   0.0087 |         5 | 0.9219 |
| occipital     | hybrid   |           10 |                     0.0514 |                          10 |                  0.002  |                         0.3397 |   0.0115 |         9 | 0.0137 |
| occipital     | neural   |           10 |                     0.0633 |                          10 |                  0.002  |                         0.4331 |   0.0161 |        10 | 0.002  |
| contralateral | hybrid   |           10 |                     0.0523 |                          10 |                  0.002  |                         0.3452 |   0.0184 |        10 | 0.002  |
| contralateral | neural   |           10 |                     0.0555 |                          10 |                  0.002  |                         0.3794 |   0.0161 |         9 | 0.0195 |
| ipsilateral   | hybrid   |           10 |                     0.0146 |                           8 |                  0.0195 |                         0.0963 |   0.0034 |         6 | 0.2324 |
| ipsilateral   | neural   |           10 |                     0.0171 |                           8 |                  0.1055 |                         0.1169 |   0.0042 |         6 | 0.3223 |

### Hybrid - neural (difference of residual_r_specific drops)

| group         | metric              |   hybrid_minus_neural_drop |   n_pos |   n_subjects |   p_wilcoxon |
|:--------------|:--------------------|---------------------------:|--------:|-------------:|-------------:|
| prefrontal    | residual_r_specific |                    -0.0033 |       2 |           10 |       0.084  |
| frontal       | residual_r_specific |                     0.006  |       7 |           10 |       0.2324 |
| occipital     | residual_r_specific |                    -0.0119 |       6 |           10 |       1      |
| contralateral | residual_r_specific |                    -0.0032 |       4 |           10 |       0.5566 |
| ipsilateral   | residual_r_specific |                    -0.0025 |       5 |           10 |       0.7695 |

## Reference: diagnostics stage (CNN-LSTM direct, uncleaned, test)

Prefrontal-only control (input Fp1/Fp2 only) and all-channel model:

| model                            |      r |   residual_r_specific |
|:---------------------------------|-------:|----------------------:|
| CNN-LSTM direct, all 32 channels | 0.5877 |                0.1344 |
| CNN-LSTM direct, Fp1/Fp2 only    | 0.3624 |                0.0624 |

Group-occlusion drops of the all-channel model:

| occluded      |   drop_r |   drop_residual_r_specific |
|:--------------|---------:|---------------------------:|
| contralateral |   0.1297 |                     0.05   |
| ipsilateral   |   0.0752 |                     0.0279 |
| occipital     |   0.088  |                     0.0645 |
| prefrontal    |   0.061  |                     0.0439 |

Figure: `experiment2.png`.
