# Experiment 2: physiological branches (RQ2)

Setting fine-tune, proposed head, proposed encoder (lejepa_guided); ocular-only has no encoder. Test scores per LOSO subject, mean over 3 decoder seeds, then mean over 10 subjects. `head_floor`: the warp head at initialization (cue phase prior); `template_cue` is scored on the same samples. `*_seed_sd`: SD over seeds of the subject-mean score.

| branch       |      r |     r2 |   nrmse |   residual_r_specific |   residual_r_matched |   r_seed_sd |   residual_r_matched_seed_sd |   residual_r_specific_seed_sd |
|:-------------|-------:|-------:|--------:|----------------------:|---------------------:|------------:|-----------------------------:|------------------------------:|
| neural       | 0.8184 | 0.5267 |  0.2143 |                0.1459 |               0.1292 |      0.0013 |                       0.0019 |                        0.0035 |
| hybrid       | 0.815  | 0.5195 |  0.2163 |                0.1401 |               0.1214 |      0.003  |                       0.003  |                        0.0045 |
| ocular       | 0.8045 | 0.5183 |  0.2182 |                0.1047 |               0.096  |      0.0067 |                       0.0028 |                        0.015  |
| head_floor   | 0.8039 | 0.5284 |  0.2162 |                0.0333 |               0      |    nan      |                     nan      |                      nan      |
| template_cue | 0.8027 | 0.5244 |  0.2164 |              nan      |               0      |    nan      |                     nan      |                      nan      |

## Paired comparisons (Wilcoxon over subjects)

residual_r_specific is not compared with the rigid floors (window effect, SSL-27); their residual_r_matched is 0.

| a      | b            | metric              |   mean_diff |   n_pos |   n_subjects |   p_wilcoxon |
|:-------|:-------------|:--------------------|------------:|--------:|-------------:|-------------:|
| hybrid | neural       | r                   |     -0.0034 |       6 |           10 |       0.8457 |
| hybrid | neural       | r2                  |     -0.0072 |       5 |           10 |       0.7695 |
| hybrid | neural       | nrmse               |      0.002  |       7 |           10 |       0.4316 |
| hybrid | neural       | residual_r_specific |     -0.0058 |       5 |           10 |       0.6953 |
| hybrid | neural       | residual_r_matched  |     -0.0078 |       3 |           10 |       0.3223 |
| hybrid | ocular       | r                   |      0.0105 |       5 |           10 |       0.5566 |
| hybrid | ocular       | r2                  |      0.0012 |       5 |           10 |       0.7695 |
| hybrid | ocular       | nrmse               |     -0.0018 |       5 |           10 |       1      |
| hybrid | ocular       | residual_r_specific |      0.0354 |       9 |           10 |       0.0371 |
| hybrid | ocular       | residual_r_matched  |      0.0253 |       6 |           10 |       0.1934 |
| neural | ocular       | r                   |      0.0139 |       7 |           10 |       0.084  |
| neural | ocular       | r2                  |      0.0084 |       7 |           10 |       0.6953 |
| neural | ocular       | nrmse               |     -0.0039 |       3 |           10 |       0.2754 |
| neural | ocular       | residual_r_specific |      0.0412 |       8 |           10 |       0.0371 |
| neural | ocular       | residual_r_matched  |      0.0331 |       7 |           10 |       0.0488 |
| neural | template_cue | r                   |      0.0157 |       5 |           10 |       0.4922 |
| neural | template_cue | r2                  |      0.0023 |       4 |           10 |       0.625  |
| neural | template_cue | nrmse               |     -0.0022 |       5 |           10 |       1      |
| neural | template_cue | residual_r_matched  |      0.1292 |      10 |           10 |       0.002  |
| hybrid | template_cue | r                   |      0.0123 |       3 |           10 |       0.8457 |
| hybrid | template_cue | r2                  |     -0.0049 |       3 |           10 |       0.4922 |
| hybrid | template_cue | nrmse               |     -0.0001 |       7 |           10 |       0.6953 |
| hybrid | template_cue | residual_r_matched  |      0.1214 |      10 |           10 |       0.002  |
| ocular | template_cue | r                   |      0.0018 |       3 |           10 |       0.6953 |
| ocular | template_cue | r2                  |     -0.0061 |       4 |           10 |       0.4316 |
| ocular | template_cue | nrmse               |      0.0017 |       6 |           10 |       0.4316 |
| ocular | template_cue | residual_r_matched  |      0.096  |      10 |           10 |       0.002  |
| neural | head_floor   | r                   |      0.0145 |       5 |           10 |       0.5566 |
| neural | head_floor   | r2                  |     -0.0016 |       4 |           10 |       0.625  |
| neural | head_floor   | nrmse               |     -0.0019 |       5 |           10 |       1      |
| neural | head_floor   | residual_r_matched  |      0.1292 |      10 |           10 |       0.002  |
| hybrid | head_floor   | r                   |      0.0111 |       3 |           10 |       0.9219 |
| hybrid | head_floor   | r2                  |     -0.0089 |       3 |           10 |       0.4316 |
| hybrid | head_floor   | nrmse               |      0.0002 |       7 |           10 |       0.625  |
| hybrid | head_floor   | residual_r_matched  |      0.1214 |      10 |           10 |       0.002  |
| ocular | head_floor   | r                   |      0.0006 |       5 |           10 |       0.625  |
| ocular | head_floor   | r2                  |     -0.0101 |       4 |           10 |       0.3223 |
| ocular | head_floor   | nrmse               |      0.002  |       6 |           10 |       0.375  |
| ocular | head_floor   | residual_r_matched  |      0.096  |      10 |           10 |       0.002  |

## Trial-specific readings: residual_r_specific vs residual_r_matched (SSL-27)

Finding only if p < 0.05 on residual_r_matched (uncorrected). Matched from output/proposed-model/rescoring/matched/20261126-083428.

| a      | b      |   mean_diff_matched |   mean_diff_rrs |   n_pos_matched |   n_pos_rrs |   p_wilcoxon_matched |   p_wilcoxon_rrs | reading_rrs   | reading_matched   | changed   |
|:-------|:-------|--------------------:|----------------:|----------------:|------------:|---------------------:|-----------------:|:--------------|:------------------|:----------|
| hybrid | neural |             -0.0078 |         -0.0058 |               3 |           5 |               0.3223 |           0.6953 | not shown     | not shown         | False     |
| hybrid | ocular |              0.0253 |          0.0354 |               6 |           9 |               0.1934 |           0.0371 | finding (+)   | not shown         | True      |
| neural | ocular |              0.0331 |          0.0412 |               7 |           8 |               0.0488 |           0.0371 | finding (+)   | finding (+)       | False     |

## Neural-branch group occlusion (seed 1, `exp1-occl`)

Drop = full - occluded per test subject; the group is zeroed in the neural input only, ocular proxies intact. `p_*`: Wilcoxon of the drop against 0; `rel_drop_*`: mean drop / mean full score. Matched drops from output/proposed-model/rescoring/occlusion/20261127-015811.

| group         | branch   |   n_subjects |   drop_residual_r_matched |   n_pos_residual_r_matched |   p_residual_r_matched |   drop_residual_r_specific |   n_pos_residual_r_specific |   p_residual_r_specific |   rel_drop_residual_r_specific |   drop_r |   n_pos_r |    p_r |
|:--------------|:---------|-------------:|--------------------------:|---------------------------:|-----------------------:|---------------------------:|----------------------------:|------------------------:|-------------------------------:|---------:|----------:|-------:|
| prefrontal    | hybrid   |           10 |                    0.0065 |                          9 |                 0.0098 |                     0.0077 |                           9 |                  0.0059 |                         0.051  |   0.0013 |         6 | 0.4922 |
| prefrontal    | neural   |           10 |                    0.0073 |                          9 |                 0.0039 |                     0.011  |                           9 |                  0.0059 |                         0.0754 |   0.0023 |         8 | 0.1309 |
| frontal       | hybrid   |           10 |                    0.0045 |                          7 |                 0.4922 |                     0.0094 |                           7 |                  0.1602 |                         0.0619 |   0.0072 |         4 | 1      |
| frontal       | neural   |           10 |                    0.0001 |                          4 |                 1      |                     0.0034 |                           4 |                  0.8457 |                         0.023  |   0.0087 |         5 | 0.9219 |
| occipital     | hybrid   |           10 |                    0.035  |                         10 |                 0.002  |                     0.0514 |                          10 |                  0.002  |                         0.3397 |   0.0115 |         9 | 0.0137 |
| occipital     | neural   |           10 |                    0.0448 |                         10 |                 0.002  |                     0.0633 |                          10 |                  0.002  |                         0.4331 |   0.0161 |        10 | 0.002  |
| contralateral | hybrid   |           10 |                    0.0275 |                          8 |                 0.0137 |                     0.0523 |                          10 |                  0.002  |                         0.3452 |   0.0184 |        10 | 0.002  |
| contralateral | neural   |           10 |                    0.0328 |                         10 |                 0.002  |                     0.0555 |                          10 |                  0.002  |                         0.3794 |   0.0161 |         9 | 0.0195 |
| ipsilateral   | hybrid   |           10 |                    0.0071 |                          8 |                 0.2324 |                     0.0146 |                           8 |                  0.0195 |                         0.0963 |   0.0034 |         6 | 0.2324 |
| ipsilateral   | neural   |           10 |                    0.0101 |                          6 |                 0.2324 |                     0.0171 |                           8 |                  0.1055 |                         0.1169 |   0.0042 |         6 | 0.3223 |

### Occlusion readings: residual_r_specific vs residual_r_matched

| branch   | group         |   mean_diff_matched |   mean_diff_rrs |   n_pos_matched |   n_pos_rrs |   p_wilcoxon_matched |   p_wilcoxon_rrs | reading_rrs   | reading_matched   | changed   |
|:---------|:--------------|--------------------:|----------------:|----------------:|------------:|---------------------:|-----------------:|:--------------|:------------------|:----------|
| hybrid   | contralateral |              0.0275 |          0.0523 |               8 |          10 |               0.0137 |           0.002  | finding (+)   | finding (+)       | False     |
| hybrid   | frontal       |              0.0045 |          0.0094 |               7 |           7 |               0.4922 |           0.1602 | not shown     | not shown         | False     |
| hybrid   | ipsilateral   |              0.0071 |          0.0146 |               8 |           8 |               0.2324 |           0.0195 | finding (+)   | not shown         | True      |
| hybrid   | occipital     |              0.035  |          0.0514 |              10 |          10 |               0.002  |           0.002  | finding (+)   | finding (+)       | False     |
| hybrid   | prefrontal    |              0.0065 |          0.0077 |               9 |           9 |               0.0098 |           0.0059 | finding (+)   | finding (+)       | False     |
| neural   | contralateral |              0.0328 |          0.0555 |              10 |          10 |               0.002  |           0.002  | finding (+)   | finding (+)       | False     |
| neural   | frontal       |              0.0001 |          0.0034 |               4 |           4 |               1      |           0.8457 | not shown     | not shown         | False     |
| neural   | ipsilateral   |              0.0101 |          0.0171 |               6 |           8 |               0.2324 |           0.1055 | not shown     | not shown         | False     |
| neural   | occipital     |              0.0448 |          0.0633 |              10 |          10 |               0.002  |           0.002  | finding (+)   | finding (+)       | False     |
| neural   | prefrontal    |              0.0073 |          0.011  |               9 |           9 |               0.0039 |           0.0059 | finding (+)   | finding (+)       | False     |

### Hybrid - neural (difference of drops)

| group         | metric              |   mean_diff |   n_pos |   n_subjects |   p_wilcoxon |
|:--------------|:--------------------|------------:|--------:|-------------:|-------------:|
| prefrontal    | residual_r_specific |     -0.0033 |       2 |           10 |       0.084  |
| prefrontal    | residual_r_matched  |     -0.0009 |       5 |           10 |       0.5566 |
| frontal       | residual_r_specific |      0.006  |       7 |           10 |       0.2324 |
| frontal       | residual_r_matched  |      0.0044 |       6 |           10 |       0.625  |
| occipital     | residual_r_specific |     -0.0119 |       6 |           10 |       1      |
| occipital     | residual_r_matched  |     -0.0099 |       3 |           10 |       0.3223 |
| contralateral | residual_r_specific |     -0.0032 |       4 |           10 |       0.5566 |
| contralateral | residual_r_matched  |     -0.0053 |       4 |           10 |       0.375  |
| ipsilateral   | residual_r_specific |     -0.0025 |       5 |           10 |       0.7695 |
| ipsilateral   | residual_r_matched  |     -0.003  |       5 |           10 |       0.625  |

| group         |   mean_diff_matched |   mean_diff_rrs |   n_pos_matched |   n_pos_rrs |   p_wilcoxon_matched |   p_wilcoxon_rrs | reading_rrs   | reading_matched   | changed   |
|:--------------|--------------------:|----------------:|----------------:|------------:|---------------------:|-----------------:|:--------------|:------------------|:----------|
| contralateral |             -0.0053 |         -0.0032 |               4 |           4 |               0.375  |           0.5566 | not shown     | not shown         | False     |
| frontal       |              0.0044 |          0.006  |               6 |           7 |               0.625  |           0.2324 | not shown     | not shown         | False     |
| ipsilateral   |             -0.003  |         -0.0025 |               5 |           5 |               0.625  |           0.7695 | not shown     | not shown         | False     |
| occipital     |             -0.0099 |         -0.0119 |               3 |           6 |               0.3223 |           1      | not shown     | not shown         | False     |
| prefrontal    |             -0.0009 |         -0.0033 |               5 |           2 |               0.5566 |           0.084  | not shown     | not shown         | False     |

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
