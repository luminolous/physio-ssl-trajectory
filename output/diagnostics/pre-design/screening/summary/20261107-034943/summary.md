# Band and channel screening

CNN-LSTM, 2 s, fixed budget. Inner validation is the selection readout (folds share validation subjects, so p is descriptive). LOSO test is descriptive only.

| config                   |   n_seeds |   n_parameters |   median_best_epoch |   wall_min |   val_r |   test_r |   val_r2 |   test_r2 |   val_nrmse |   test_nrmse |   val_residual_r |   test_residual_r |   val_residual_r_specific |   test_residual_r_specific |   test_r_vs_floor |   n_above_floor_r |   test_r2_vs_floor |   n_above_floor_r2 |
|:-------------------------|----------:|---------------:|--------------------:|-----------:|--------:|---------:|---------:|----------:|------------:|-------------:|-----------------:|------------------:|--------------------------:|---------------------------:|------------------:|------------------:|-------------------:|-------------------:|
| broadband__all           |         1 |          10843 |                 8.5 |       29   |   0.666 |    0.601 |    0.256 |     0.049 |       0.27  |        0.301 |            0.325 |             0.264 |                     0.174 |                      0.153 |            -0.202 |                 2 |             -0.475 |                  1 |
| broadband__no_prefrontal |         1 |          10587 |                 5   |       23.8 |   0.648 |    0.575 |    0.245 |     0.041 |       0.275 |        0.307 |            0.279 |             0.216 |                     0.158 |                      0.128 |            -0.228 |                 1 |             -0.483 |                  1 |
| mrcp__all                |         1 |          10843 |                 9.5 |       31.9 |   0.616 |    0.55  |    0.242 |     0.113 |       0.278 |        0.3   |            0.282 |             0.206 |                     0.151 |                      0.127 |            -0.253 |                 1 |             -0.412 |                  1 |
| mrcp_env__all            |         1 |          19035 |                 4   |       74   |   0.677 |    0.647 |    0.263 |     0.196 |       0.269 |        0.28  |            0.292 |             0.235 |                     0.144 |                      0.139 |            -0.156 |                 1 |             -0.328 |                  1 |
| broadband__sensorimotor  |         1 |           8155 |                 4.5 |       11.4 |   0.58  |    0.537 |    0.125 |     0.004 |       0.299 |        0.316 |            0.169 |             0.128 |                     0.083 |                      0.082 |            -0.266 |                 1 |             -0.521 |                  1 |
| mrcp__contralateral      |         1 |           7515 |                 9   |        9.1 |   0.472 |    0.432 |    0.093 |     0.046 |       0.311 |        0.319 |            0.125 |             0.105 |                     0.076 |                      0.069 |            -0.371 |                 0 |             -0.478 |                  0 |
| broadband__prefrontal    |         1 |           7003 |                 7   |        5.3 |   0.417 |    0.383 |    0.044 |    -0.001 |       0.321 |        0.327 |            0.128 |             0.104 |                     0.073 |                      0.069 |            -0.42  |                 0 |             -0.525 |                  0 |
| broadband__contralateral |         1 |           7515 |                 3.5 |        6.5 |   0.532 |    0.479 |    0.131 |    -0.064 |       0.302 |        0.328 |            0.14  |             0.098 |                     0.071 |                      0.058 |            -0.324 |                 1 |             -0.589 |                  0 |
| mrcp__prefrontal         |         1 |           7003 |                10   |        6.6 |   0.422 |    0.353 |    0.05  |    -0.004 |       0.32  |        0.329 |            0.119 |             0.091 |                     0.069 |                      0.06  |            -0.449 |                 0 |             -0.529 |                  0 |
| mrcp__ipsilateral        |         1 |           7515 |                 6.5 |        7.6 |   0.465 |    0.417 |    0.086 |     0.017 |       0.313 |        0.324 |            0.102 |             0.075 |                     0.066 |                      0.059 |            -0.386 |                 0 |             -0.507 |                  0 |
| broadband__ipsilateral   |         1 |           7515 |                 2   |        5.9 |   0.503 |    0.467 |    0.089 |    -0.019 |       0.308 |        0.325 |            0.111 |             0.085 |                     0.065 |                      0.058 |            -0.335 |                 1 |             -0.543 |                  1 |
| env__all                 |         1 |          14939 |                 1   |       32.2 |   0.5   |    0.51  |    0.043 |     0.044 |       0.314 |        0.313 |            0.076 |             0.055 |                     0.022 |                      0.018 |            -0.293 |                 1 |             -0.481 |                  0 |
| env__ipsilateral         |         1 |           8283 |                 1   |        7.9 |   0.405 |    0.414 |   -0.002 |    -0.004 |       0.326 |        0.326 |            0.024 |             0.01  |                    -0.002 |                     -0.003 |            -0.389 |                 0 |             -0.528 |                  0 |
| env__contralateral       |         1 |           8283 |                 2   |        8.1 |   0.423 |    0.427 |    0.013 |     0     |       0.323 |        0.324 |            0.028 |             0.007 |                    -0.007 |                     -0.007 |            -0.375 |                 0 |             -0.524 |                  0 |

| contrast                              | metric              | split     |      a |      b |   diff |   diff_sd |   n_up |   n |   wilcoxon_p |
|:--------------------------------------|:--------------------|:----------|-------:|-------:|-------:|----------:|-------:|----:|-------------:|
| contra vs ipsi, broadband             | r                   | inner_val |  0.532 |  0.503 |  0.028 |     0.066 |      7 |  10 |        0.275 |
| contra vs ipsi, broadband             | r                   | test      |  0.479 |  0.467 |  0.011 |     0.072 |      5 |  10 |        0.846 |
| contra vs ipsi, broadband             | r2                  | inner_val |  0.131 |  0.089 |  0.041 |     0.04  |      9 |  10 |        0.004 |
| contra vs ipsi, broadband             | r2                  | test      | -0.064 | -0.019 | -0.046 |     0.242 |      5 |  10 |        1     |
| contra vs ipsi, broadband             | residual_r_specific | inner_val |  0.071 |  0.065 |  0.006 |     0.02  |      6 |  10 |        0.492 |
| contra vs ipsi, broadband             | residual_r_specific | test      |  0.058 |  0.058 |  0     |     0.025 |      6 |  10 |        0.625 |
| contra vs ipsi, MRCP                  | r                   | inner_val |  0.472 |  0.465 |  0.007 |     0.028 |      4 |  10 |        0.695 |
| contra vs ipsi, MRCP                  | r                   | test      |  0.432 |  0.417 |  0.015 |     0.069 |      6 |  10 |        0.625 |
| contra vs ipsi, MRCP                  | r2                  | inner_val |  0.093 |  0.086 |  0.007 |     0.03  |      5 |  10 |        0.492 |
| contra vs ipsi, MRCP                  | r2                  | test      |  0.046 |  0.017 |  0.029 |     0.052 |      6 |  10 |        0.16  |
| contra vs ipsi, MRCP                  | residual_r_specific | inner_val |  0.076 |  0.066 |  0.01  |     0.009 |      8 |  10 |        0.02  |
| contra vs ipsi, MRCP                  | residual_r_specific | test      |  0.069 |  0.059 |  0.01  |     0.012 |      9 |  10 |        0.02  |
| contra vs ipsi, envelope              | r                   | inner_val |  0.423 |  0.405 |  0.018 |     0.044 |      8 |  10 |        0.16  |
| contra vs ipsi, envelope              | r                   | test      |  0.427 |  0.414 |  0.014 |     0.037 |      6 |  10 |        0.322 |
| contra vs ipsi, envelope              | r2                  | inner_val |  0.013 | -0.002 |  0.015 |     0.031 |      7 |  10 |        0.16  |
| contra vs ipsi, envelope              | r2                  | test      |  0     | -0.004 |  0.004 |     0.061 |      6 |  10 |        0.846 |
| contra vs ipsi, envelope              | residual_r_specific | inner_val | -0.007 | -0.002 | -0.005 |     0.007 |      2 |  10 |        0.064 |
| contra vs ipsi, envelope              | residual_r_specific | test      | -0.007 | -0.003 | -0.004 |     0.011 |      4 |  10 |        0.322 |
| MRCP vs broadband, all                | r                   | inner_val |  0.616 |  0.666 | -0.05  |     0.036 |      1 |  10 |        0.004 |
| MRCP vs broadband, all                | r                   | test      |  0.55  |  0.601 | -0.051 |     0.038 |      2 |  10 |        0.01  |
| MRCP vs broadband, all                | r2                  | inner_val |  0.242 |  0.256 | -0.014 |     0.07  |      4 |  10 |        0.77  |
| MRCP vs broadband, all                | r2                  | test      |  0.113 |  0.049 |  0.064 |     0.364 |      3 |  10 |        0.432 |
| MRCP vs broadband, all                | residual_r_specific | inner_val |  0.151 |  0.174 | -0.023 |     0.035 |      3 |  10 |        0.084 |
| MRCP vs broadband, all                | residual_r_specific | test      |  0.127 |  0.153 | -0.026 |     0.051 |      3 |  10 |        0.105 |
| envelope vs broadband, all            | r                   | inner_val |  0.5   |  0.666 | -0.166 |     0.094 |      1 |  10 |        0.004 |
| envelope vs broadband, all            | r                   | test      |  0.51  |  0.601 | -0.091 |     0.158 |      1 |  10 |        0.064 |
| envelope vs broadband, all            | r2                  | inner_val |  0.043 |  0.256 | -0.213 |     0.165 |      1 |  10 |        0.01  |
| envelope vs broadband, all            | r2                  | test      |  0.044 |  0.049 | -0.005 |     0.397 |      4 |  10 |        0.625 |
| envelope vs broadband, all            | residual_r_specific | inner_val |  0.022 |  0.174 | -0.152 |     0.038 |      0 |  10 |        0.002 |
| envelope vs broadband, all            | residual_r_specific | test      |  0.018 |  0.153 | -0.135 |     0.058 |      0 |  10 |        0.002 |
| MRCP + envelope vs broadband, all     | r                   | inner_val |  0.677 |  0.666 |  0.01  |     0.044 |      5 |  10 |        0.625 |
| MRCP + envelope vs broadband, all     | r                   | test      |  0.647 |  0.601 |  0.046 |     0.077 |      6 |  10 |        0.193 |
| MRCP + envelope vs broadband, all     | r2                  | inner_val |  0.263 |  0.256 |  0.008 |     0.078 |      5 |  10 |        1     |
| MRCP + envelope vs broadband, all     | r2                  | test      |  0.196 |  0.049 |  0.147 |     0.351 |      5 |  10 |        0.232 |
| MRCP + envelope vs broadband, all     | residual_r_specific | inner_val |  0.144 |  0.174 | -0.03  |     0.027 |      1 |  10 |        0.014 |
| MRCP + envelope vs broadband, all     | residual_r_specific | test      |  0.139 |  0.153 | -0.014 |     0.042 |      5 |  10 |        0.492 |
| MRCP + envelope vs MRCP, all          | r                   | inner_val |  0.677 |  0.616 |  0.061 |     0.063 |      7 |  10 |        0.027 |
| MRCP + envelope vs MRCP, all          | r                   | test      |  0.647 |  0.55  |  0.097 |     0.068 |     10 |  10 |        0.002 |
| MRCP + envelope vs MRCP, all          | r2                  | inner_val |  0.263 |  0.242 |  0.022 |     0.063 |      6 |  10 |        0.322 |
| MRCP + envelope vs MRCP, all          | r2                  | test      |  0.196 |  0.113 |  0.083 |     0.139 |      8 |  10 |        0.064 |
| MRCP + envelope vs MRCP, all          | residual_r_specific | inner_val |  0.144 |  0.151 | -0.006 |     0.04  |      5 |  10 |        0.846 |
| MRCP + envelope vs MRCP, all          | residual_r_specific | test      |  0.139 |  0.127 |  0.012 |     0.039 |      7 |  10 |        0.375 |
| MRCP vs envelope, all                 | r                   | inner_val |  0.616 |  0.5   |  0.116 |     0.118 |      9 |  10 |        0.02  |
| MRCP vs envelope, all                 | r                   | test      |  0.55  |  0.51  |  0.04  |     0.155 |      7 |  10 |        0.193 |
| MRCP vs envelope, all                 | r2                  | inner_val |  0.242 |  0.043 |  0.198 |     0.144 |      9 |  10 |        0.006 |
| MRCP vs envelope, all                 | r2                  | test      |  0.113 |  0.044 |  0.069 |     0.297 |      8 |  10 |        0.193 |
| MRCP vs envelope, all                 | residual_r_specific | inner_val |  0.151 |  0.022 |  0.129 |     0.048 |     10 |  10 |        0.002 |
| MRCP vs envelope, all                 | residual_r_specific | test      |  0.127 |  0.018 |  0.109 |     0.04  |     10 |  10 |        0.002 |
| no prefrontal vs all, broadband       | r                   | inner_val |  0.648 |  0.666 | -0.019 |     0.045 |      3 |  10 |        0.322 |
| no prefrontal vs all, broadband       | r                   | test      |  0.575 |  0.601 | -0.026 |     0.066 |      4 |  10 |        0.375 |
| no prefrontal vs all, broadband       | r2                  | inner_val |  0.245 |  0.256 | -0.011 |     0.053 |      4 |  10 |        0.557 |
| no prefrontal vs all, broadband       | r2                  | test      |  0.041 |  0.049 | -0.008 |     0.223 |      4 |  10 |        0.625 |
| no prefrontal vs all, broadband       | residual_r_specific | inner_val |  0.158 |  0.174 | -0.016 |     0.029 |      3 |  10 |        0.084 |
| no prefrontal vs all, broadband       | residual_r_specific | test      |  0.128 |  0.153 | -0.025 |     0.04  |      2 |  10 |        0.105 |
| sensorimotor vs all, broadband        | r                   | inner_val |  0.58  |  0.666 | -0.087 |     0.038 |      0 |  10 |        0.002 |
| sensorimotor vs all, broadband        | r                   | test      |  0.537 |  0.601 | -0.064 |     0.059 |      1 |  10 |        0.01  |
| sensorimotor vs all, broadband        | r2                  | inner_val |  0.125 |  0.256 | -0.131 |     0.104 |      1 |  10 |        0.004 |
| sensorimotor vs all, broadband        | r2                  | test      |  0.004 |  0.049 | -0.045 |     0.244 |      2 |  10 |        0.432 |
| sensorimotor vs all, broadband        | residual_r_specific | inner_val |  0.083 |  0.174 | -0.09  |     0.03  |      0 |  10 |        0.002 |
| sensorimotor vs all, broadband        | residual_r_specific | test      |  0.082 |  0.153 | -0.071 |     0.038 |      0 |  10 |        0.002 |
| sensorimotor vs prefrontal, broadband | r                   | inner_val |  0.58  |  0.417 |  0.162 |     0.049 |     10 |  10 |        0.002 |
| sensorimotor vs prefrontal, broadband | r                   | test      |  0.537 |  0.383 |  0.154 |     0.067 |     10 |  10 |        0.002 |
| sensorimotor vs prefrontal, broadband | r2                  | inner_val |  0.125 |  0.044 |  0.081 |     0.098 |      7 |  10 |        0.064 |
| sensorimotor vs prefrontal, broadband | r2                  | test      |  0.004 | -0.001 |  0.005 |     0.218 |      7 |  10 |        0.922 |
| sensorimotor vs prefrontal, broadband | residual_r_specific | inner_val |  0.083 |  0.073 |  0.01  |     0.018 |      8 |  10 |        0.105 |
| sensorimotor vs prefrontal, broadband | residual_r_specific | test      |  0.082 |  0.069 |  0.013 |     0.045 |      7 |  10 |        0.375 |
| prefrontal vs contra, broadband       | r                   | inner_val |  0.417 |  0.532 | -0.114 |     0.063 |      0 |  10 |        0.002 |
| prefrontal vs contra, broadband       | r                   | test      |  0.383 |  0.479 | -0.096 |     0.085 |      1 |  10 |        0.004 |
| prefrontal vs contra, broadband       | r2                  | inner_val |  0.044 |  0.131 | -0.086 |     0.065 |      1 |  10 |        0.01  |
| prefrontal vs contra, broadband       | r2                  | test      | -0.001 | -0.064 |  0.064 |     0.276 |      5 |  10 |        1     |
| prefrontal vs contra, broadband       | residual_r_specific | inner_val |  0.073 |  0.071 |  0.002 |     0.027 |      6 |  10 |        0.846 |
| prefrontal vs contra, broadband       | residual_r_specific | test      |  0.069 |  0.058 |  0.011 |     0.034 |      6 |  10 |        0.432 |
| prefrontal vs contra, MRCP            | r                   | inner_val |  0.422 |  0.472 | -0.049 |     0.051 |      1 |  10 |        0.014 |
| prefrontal vs contra, MRCP            | r                   | test      |  0.353 |  0.432 | -0.078 |     0.113 |      2 |  10 |        0.084 |
| prefrontal vs contra, MRCP            | r2                  | inner_val |  0.05  |  0.093 | -0.044 |     0.051 |      2 |  10 |        0.037 |
| prefrontal vs contra, MRCP            | r2                  | test      | -0.004 |  0.046 | -0.05  |     0.092 |      3 |  10 |        0.105 |
| prefrontal vs contra, MRCP            | residual_r_specific | inner_val |  0.069 |  0.076 | -0.007 |     0.02  |      4 |  10 |        0.375 |
| prefrontal vs contra, MRCP            | residual_r_specific | test      |  0.06  |  0.069 | -0.009 |     0.026 |      3 |  10 |        0.232 |
