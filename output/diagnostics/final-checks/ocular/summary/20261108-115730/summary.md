# Ocular-cleaned variant

Subjects with a cleaned cache: P1, P2, P4, P5, P7, P8, P9, P10, P11, P12.

## Components removed

| subject   |   rank |   n_removed |   removed_variance_fit_data |   veog_r |   heog_r |
|:----------|-------:|------------:|----------------------------:|---------:|---------:|
| P1        |     32 |           2 |                       0.48  |    0.976 |    0.868 |
| P2        |     31 |           2 |                       0.337 |    0.955 |    0.845 |
| P4        |     32 |           3 |                       0.275 |    0.785 |    0.574 |
| P5        |     32 |           3 |                       0.376 |    0.937 |    0.711 |
| P7        |     32 |           2 |                       0.45  |    0.96  |    0.676 |
| P8        |     32 |           2 |                       0.18  |    0.896 |    0.733 |
| P9        |     32 |           2 |                       0.521 |    0.959 |    0.883 |
| P10       |     32 |           2 |                       0.192 |    0.894 |    0.796 |
| P11       |     32 |           3 |                       0.281 |    0.95  |    0.702 |
| P12       |     32 |           2 |                       0.172 |    0.904 |    0.866 |

## Onset-aligned averages (uV)

| channel   | band          |   mean_0_to_0.5s_uV |   peak_abs_uV |
|:----------|:--------------|--------------------:|--------------:|
| Fp1       | broadband     |             -193.65 |        207.51 |
| Fp1       | broadband_ica |              -21.41 |         36.7  |
| Cz        | broadband     |              -29.79 |         53.49 |
| Cz        | broadband_ica |              -43.07 |         64.28 |

## Cleaned vs uncleaned

| model           | metric              | split     | seeds   |   cleaned |   uncleaned |    diff |   n_up |   n |   wilcoxon_p |
|:----------------|:--------------------|:----------|:--------|----------:|------------:|--------:|-------:|----:|-------------:|
| direct          | r                   | inner_val | 1 vs 3  |     0.54  |       0.644 |  -0.103 |      0 |  10 |        0.002 |
| direct          | r                   | test      | 1 vs 3  |     0.485 |       0.588 |  -0.103 |      1 |  10 |        0.006 |
| direct          | r2                  | inner_val | 1 vs 3  |     0.073 |       0.255 |  -0.182 |      0 |  10 |        0.002 |
| direct          | r2                  | test      | 1 vs 3  |    -0.083 |       0.075 |  -0.158 |      1 |  10 |        0.049 |
| direct          | nrmse               | inner_val | 1 vs 3  |     0.307 |       0.273 |   0.034 |     10 |  10 |        0.002 |
| direct          | nrmse               | test      | 1 vs 3  |     0.33  |       0.3   |   0.03  |      9 |  10 |        0.027 |
| direct          | residual_r_specific | inner_val | 1 vs 3  |     0.062 |       0.161 |  -0.098 |      0 |  10 |        0.002 |
| direct          | residual_r_specific | test      | 1 vs 3  |     0.068 |       0.134 |  -0.067 |      0 |  10 |        0.002 |
| residual        | r                   | inner_val | 1 vs 3  |   nan     |     nan     | nan     |    nan | nan |      nan     |
| residual        | r                   | test      | 1 vs 3  |     0.796 |       0.813 |  -0.017 |      3 |  10 |        0.027 |
| residual        | r2                  | inner_val | 1 vs 3  |   nan     |     nan     | nan     |    nan | nan |      nan     |
| residual        | r2                  | test      | 1 vs 3  |     0.487 |       0.51  |  -0.023 |      5 |  10 |        0.232 |
| residual        | nrmse               | inner_val | 1 vs 3  |   nan     |     nan     | nan     |    nan | nan |      nan     |
| residual        | nrmse               | test      | 1 vs 3  |     0.225 |       0.218 |   0.006 |      8 |  10 |        0.027 |
| residual        | residual_r_specific | inner_val | 1 vs 3  |   nan     |     nan     | nan     |    nan | nan |      nan     |
| residual        | residual_r_specific | test      | 1 vs 3  |     0.08  |       0.192 |  -0.112 |      0 |  10 |        0.002 |
| prefrontal only | r                   | inner_val | 1 vs 3  |     0.181 |       0.414 |  -0.233 |      1 |  10 |        0.004 |
| prefrontal only | r                   | test      | 1 vs 3  |     0.156 |       0.362 |  -0.207 |      1 |  10 |        0.027 |
| prefrontal only | r2                  | inner_val | 1 vs 3  |    -0.111 |       0.042 |  -0.153 |      0 |  10 |        0.002 |
| prefrontal only | r2                  | test      | 1 vs 3  |    -0.12  |      -0.028 |  -0.092 |      2 |  10 |        0.131 |
| prefrontal only | nrmse               | inner_val | 1 vs 3  |     0.349 |       0.321 |   0.028 |     10 |  10 |        0.002 |
| prefrontal only | nrmse               | test      | 1 vs 3  |     0.35  |       0.331 |   0.019 |      9 |  10 |        0.037 |
| prefrontal only | residual_r_specific | inner_val | 1 vs 3  |     0.015 |       0.071 |  -0.055 |      0 |  10 |        0.002 |
| prefrontal only | residual_r_specific | test      | 1 vs 3  |     0.015 |       0.062 |  -0.047 |      0 |  10 |        0.002 |

## Occlusion, group drops (test)

| occluded      |   uncleaned_r |   uncleaned_residual_r_specific |   cleaned_r |   cleaned_residual_r_specific |
|:--------------|--------------:|--------------------------------:|------------:|------------------------------:|
| contralateral |         0.13  |                           0.05  |       0.218 |                         0.033 |
| ipsilateral   |         0.075 |                           0.028 |       0.102 |                         0.022 |
| occipital     |         0.088 |                           0.065 |       0.091 |                         0.039 |
| prefrontal    |         0.061 |                           0.044 |       0.01  |                        -0.004 |

