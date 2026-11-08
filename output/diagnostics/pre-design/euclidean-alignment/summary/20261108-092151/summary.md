# Euclidean alignment

CNN-LSTM, broadband, all channels, 2 s. Seeds: aligned 3, unaligned 3, averaged per fold or subject first. Inner-validation folds share subjects, so p is descriptive. Median best epoch: {'aligned': 9.0, 'unaligned': 8.0}.

| metric              | split     | against      |   aligned |   other |   diff |   diff_sd |   n_up |   n |   wilcoxon_p |
|:--------------------|:----------|:-------------|----------:|--------:|-------:|----------:|-------:|----:|-------------:|
| r                   | inner_val | unaligned    |     0.649 |   0.644 |  0.005 |     0.018 |      6 |  10 |        0.432 |
| r                   | test      | unaligned    |     0.561 |   0.588 | -0.026 |     0.05  |      2 |  10 |        0.084 |
| r                   | test      | template_cue |     0.561 |   0.803 | -0.241 |     0.223 |      1 |  10 |        0.014 |
| r2                  | inner_val | unaligned    |     0.258 |   0.255 |  0.003 |     0.024 |      6 |  10 |        0.846 |
| r2                  | test      | unaligned    |    -0.061 |   0.075 | -0.136 |     0.301 |      2 |  10 |        0.037 |
| r2                  | test      | template_cue |    -0.061 |   0.524 | -0.585 |     0.65  |      1 |  10 |        0.006 |
| nrmse               | inner_val | unaligned    |     0.272 |   0.273 | -0.001 |     0.004 |      3 |  10 |        0.322 |
| nrmse               | test      | unaligned    |     0.317 |   0.3   |  0.017 |     0.034 |      9 |  10 |        0.02  |
| nrmse               | test      | template_cue |     0.317 |   0.216 |  0.101 |     0.096 |      9 |  10 |        0.006 |
| residual_r_specific | inner_val | unaligned    |     0.162 |   0.161 |  0.001 |     0.02  |      3 |  10 |        0.77  |
| residual_r_specific | test      | unaligned    |     0.134 |   0.134 | -0.001 |     0.028 |      4 |  10 |        0.846 |
