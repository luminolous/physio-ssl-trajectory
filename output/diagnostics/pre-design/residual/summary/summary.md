# Residual formulation against direct prediction and the cue-locked template

`a` is the residual run, `b` the comparator. LOSO, 10 subjects, 3 seeds, seed-averaged per subject.

| model    | metric              | against      |     a |     b |   diff |   diff_sd |   n_up |   n |   wilcoxon_p |
|:---------|:--------------------|:-------------|------:|------:|-------:|----------:|-------:|----:|-------------:|
| cnn_lstm | r                   | direct 2 s   | 0.813 | 0.588 |  0.225 |     0.202 |      9 |  10 |        0.014 |
| cnn_lstm | r                   | template_cue | 0.813 | 0.803 |  0.01  |     0.021 |      5 |  10 |        0.232 |
| cnn_lstm | r2                  | direct 2 s   | 0.51  | 0.075 |  0.435 |     0.401 |      9 |  10 |        0.006 |
| cnn_lstm | r2                  | template_cue | 0.51  | 0.524 | -0.014 |     0.047 |      4 |  10 |        0.492 |
| cnn_lstm | nrmse               | direct 2 s   | 0.218 | 0.3   | -0.082 |     0.069 |      1 |  10 |        0.006 |
| cnn_lstm | nrmse               | template_cue | 0.218 | 0.216 |  0.002 |     0.01  |      6 |  10 |        0.695 |
| cnn_lstm | residual_r          | direct 2 s   | 0.262 | 0.237 |  0.025 |     0.154 |      6 |  10 |        0.492 |
| cnn_lstm | residual_r_specific | direct 2 s   | 0.192 | 0.134 |  0.057 |     0.039 |      9 |  10 |        0.01  |
| eegnet   | r                   | direct 2 s   | 0.812 | 0.576 |  0.236 |     0.151 |      9 |  10 |        0.004 |
| eegnet   | r                   | template_cue | 0.812 | 0.803 |  0.01  |     0.018 |      6 |  10 |        0.232 |
| eegnet   | r2                  | direct 2 s   | 0.497 | 0.004 |  0.493 |     0.285 |     10 |  10 |        0.002 |
| eegnet   | r2                  | template_cue | 0.497 | 0.524 | -0.028 |     0.038 |      2 |  10 |        0.064 |
| eegnet   | nrmse               | direct 2 s   | 0.221 | 0.319 | -0.098 |     0.049 |      0 |  10 |        0.002 |
| eegnet   | nrmse               | template_cue | 0.221 | 0.216 |  0.005 |     0.009 |      5 |  10 |        0.275 |
| eegnet   | residual_r          | direct 2 s   | 0.233 | 0.148 |  0.085 |     0.139 |      8 |  10 |        0.084 |
| eegnet   | residual_r_specific | direct 2 s   | 0.161 | 0.09  |  0.07  |     0.047 |      9 |  10 |        0.004 |

Median best epoch: {'cnn_lstm': 1.0, 'eegnet': 2.0}
