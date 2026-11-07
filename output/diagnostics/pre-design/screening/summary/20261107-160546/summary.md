# Band and channel screening

CNN-LSTM, 2 s, fixed budget. Inner validation is the selection readout (folds share validation subjects, so p is descriptive). LOSO test is descriptive only.

| config                   |   n_seeds |   n_parameters |   median_best_epoch |   wall_min |   val_r |   test_r |   val_r2 |   test_r2 |   val_nrmse |   test_nrmse |   val_residual_r |   test_residual_r |   val_residual_r_specific |   test_residual_r_specific |   test_r_vs_floor |   n_above_floor_r |   test_r2_vs_floor |   n_above_floor_r2 |
|:-------------------------|----------:|---------------:|--------------------:|-----------:|--------:|---------:|---------:|----------:|------------:|-------------:|-----------------:|------------------:|--------------------------:|---------------------------:|------------------:|------------------:|-------------------:|-------------------:|
| broadband__all           |         3 |          10843 |                 8   |       29.7 |   0.644 |    0.588 |    0.255 |     0.075 |       0.273 |        0.3   |            0.297 |             0.237 |                     0.161 |                      0.134 |            -0.215 |                 1 |             -0.449 |                  1 |
| broadband__no_prefrontal |         3 |          10587 |                 5   |       24.1 |   0.635 |    0.57  |    0.244 |     0.037 |       0.276 |        0.307 |            0.282 |             0.211 |                     0.155 |                      0.127 |            -0.232 |                 1 |             -0.488 |                  1 |
| mrcp__all                |         3 |          10843 |                10.5 |       36   |   0.621 |    0.554 |    0.237 |     0.104 |       0.278 |        0.3   |            0.278 |             0.21  |                     0.153 |                      0.129 |            -0.249 |                 1 |             -0.42  |                  1 |
| mrcp_env__all            |         3 |          19035 |                 3   |       76.5 |   0.676 |    0.646 |    0.262 |     0.202 |       0.269 |        0.28  |            0.285 |             0.24  |                     0.142 |                      0.135 |            -0.157 |                 1 |             -0.323 |                  1 |
| broadband__sensorimotor  |         3 |           8155 |                 3.5 |       12.3 |   0.574 |    0.526 |    0.142 |    -0.013 |       0.297 |        0.319 |            0.161 |             0.131 |                     0.081 |                      0.073 |            -0.277 |                 1 |             -0.537 |                  1 |
| mrcp__contralateral      |         3 |           7515 |                11   |       13.9 |   0.477 |    0.441 |    0.097 |     0.051 |       0.31  |        0.317 |            0.129 |             0.108 |                     0.077 |                      0.072 |            -0.361 |                 0 |             -0.474 |                  0 |
| broadband__prefrontal    |         3 |           7003 |                 7   |        8   |   0.414 |    0.362 |    0.042 |    -0.028 |       0.321 |        0.331 |            0.125 |             0.097 |                     0.071 |                      0.062 |            -0.44  |                 0 |             -0.552 |                  0 |
| broadband__contralateral |         3 |           7515 |                 3   |        9   |   0.537 |    0.489 |    0.117 |    -0.025 |       0.303 |        0.324 |            0.137 |             0.104 |                     0.07  |                      0.064 |            -0.314 |                 1 |             -0.549 |                  0 |
| mrcp__ipsilateral        |         3 |           7515 |                 6.5 |        9.4 |   0.464 |    0.419 |    0.079 |     0.021 |       0.314 |        0.323 |            0.099 |             0.08  |                     0.065 |                      0.061 |            -0.384 |                 0 |             -0.504 |                  0 |
| mrcp__prefrontal         |         3 |           7003 |                 9   |        9.4 |   0.414 |    0.357 |    0.046 |    -0.006 |       0.321 |        0.329 |            0.114 |             0.092 |                     0.065 |                      0.059 |            -0.446 |                 0 |             -0.53  |                  0 |
| broadband__ipsilateral   |         3 |           7515 |                 3   |        7.7 |   0.509 |    0.456 |    0.087 |    -0.058 |       0.309 |        0.33  |            0.11  |             0.082 |                     0.065 |                      0.057 |            -0.347 |                 1 |             -0.582 |                  1 |
| env__all                 |         1 |          14939 |                 1   |       32.2 |   0.5   |    0.51  |    0.043 |     0.044 |       0.314 |        0.313 |            0.076 |             0.055 |                     0.022 |                      0.018 |            -0.293 |                 1 |             -0.481 |                  0 |
| env__ipsilateral         |         1 |           8283 |                 1   |        7.9 |   0.405 |    0.414 |   -0.002 |    -0.004 |       0.326 |        0.326 |            0.024 |             0.01  |                    -0.002 |                     -0.003 |            -0.389 |                 0 |             -0.528 |                  0 |
| env__contralateral       |         1 |           8283 |                 2   |        8.1 |   0.423 |    0.427 |    0.013 |     0     |       0.323 |        0.324 |            0.028 |             0.007 |                    -0.007 |                     -0.007 |            -0.375 |                 0 |             -0.524 |                  0 |

| contrast                              | metric              | split     |      a |      b |   diff |   diff_sd |   n_up |   n |   wilcoxon_p |
|:--------------------------------------|:--------------------|:----------|-------:|-------:|-------:|----------:|-------:|----:|-------------:|
| contra vs ipsi, broadband             | r                   | inner_val |  0.537 |  0.509 |  0.028 |     0.045 |      8 |  10 |        0.131 |
| contra vs ipsi, broadband             | r                   | test      |  0.489 |  0.456 |  0.033 |     0.061 |      6 |  10 |        0.131 |
| contra vs ipsi, broadband             | r2                  | inner_val |  0.117 |  0.087 |  0.029 |     0.034 |      9 |  10 |        0.01  |
| contra vs ipsi, broadband             | r2                  | test      | -0.025 | -0.058 |  0.033 |     0.194 |      6 |  10 |        0.557 |
| contra vs ipsi, broadband             | residual_r_specific | inner_val |  0.07  |  0.065 |  0.006 |     0.012 |      8 |  10 |        0.16  |
| contra vs ipsi, broadband             | residual_r_specific | test      |  0.064 |  0.057 |  0.007 |     0.02  |      7 |  10 |        0.232 |
| contra vs ipsi, MRCP                  | r                   | inner_val |  0.477 |  0.464 |  0.013 |     0.025 |      8 |  10 |        0.064 |
| contra vs ipsi, MRCP                  | r                   | test      |  0.441 |  0.419 |  0.023 |     0.058 |      6 |  10 |        0.275 |
| contra vs ipsi, MRCP                  | r2                  | inner_val |  0.097 |  0.079 |  0.017 |     0.024 |      9 |  10 |        0.027 |
| contra vs ipsi, MRCP                  | r2                  | test      |  0.051 |  0.021 |  0.03  |     0.036 |      7 |  10 |        0.037 |
| contra vs ipsi, MRCP                  | residual_r_specific | inner_val |  0.077 |  0.065 |  0.012 |     0.01  |      9 |  10 |        0.006 |
| contra vs ipsi, MRCP                  | residual_r_specific | test      |  0.072 |  0.061 |  0.011 |     0.01  |      9 |  10 |        0.014 |
| contra vs ipsi, envelope              | r                   | inner_val |  0.423 |  0.405 |  0.018 |     0.044 |      8 |  10 |        0.16  |
| contra vs ipsi, envelope              | r                   | test      |  0.427 |  0.414 |  0.014 |     0.037 |      6 |  10 |        0.322 |
| contra vs ipsi, envelope              | r2                  | inner_val |  0.013 | -0.002 |  0.015 |     0.031 |      7 |  10 |        0.16  |
| contra vs ipsi, envelope              | r2                  | test      |  0     | -0.004 |  0.004 |     0.061 |      6 |  10 |        0.846 |
| contra vs ipsi, envelope              | residual_r_specific | inner_val | -0.007 | -0.002 | -0.005 |     0.007 |      2 |  10 |        0.064 |
| contra vs ipsi, envelope              | residual_r_specific | test      | -0.007 | -0.003 | -0.004 |     0.011 |      4 |  10 |        0.322 |
| MRCP vs broadband, all                | r                   | inner_val |  0.621 |  0.644 | -0.023 |     0.027 |      3 |  10 |        0.064 |
| MRCP vs broadband, all                | r                   | test      |  0.554 |  0.588 | -0.034 |     0.038 |      3 |  10 |        0.027 |
| MRCP vs broadband, all                | r2                  | inner_val |  0.237 |  0.255 | -0.018 |     0.041 |      3 |  10 |        0.275 |
| MRCP vs broadband, all                | r2                  | test      |  0.104 |  0.075 |  0.029 |     0.267 |      4 |  10 |        0.275 |
| MRCP vs broadband, all                | residual_r_specific | inner_val |  0.153 |  0.161 | -0.007 |     0.028 |      4 |  10 |        0.625 |
| MRCP vs broadband, all                | residual_r_specific | test      |  0.129 |  0.134 | -0.006 |     0.033 |      3 |  10 |        0.492 |
| envelope vs broadband, all            | r                   | inner_val |  0.5   |  0.644 | -0.143 |     0.101 |      1 |  10 |        0.004 |
| envelope vs broadband, all            | r                   | test      |  0.51  |  0.588 | -0.078 |     0.153 |      3 |  10 |        0.16  |
| envelope vs broadband, all            | r2                  | inner_val |  0.043 |  0.255 | -0.212 |     0.151 |      1 |  10 |        0.006 |
| envelope vs broadband, all            | r2                  | test      |  0.044 |  0.075 | -0.032 |     0.311 |      4 |  10 |        0.77  |
| envelope vs broadband, all            | residual_r_specific | inner_val |  0.022 |  0.161 | -0.139 |     0.039 |      0 |  10 |        0.002 |
| envelope vs broadband, all            | residual_r_specific | test      |  0.018 |  0.134 | -0.117 |     0.046 |      0 |  10 |        0.002 |
| MRCP + envelope vs broadband, all     | r                   | inner_val |  0.676 |  0.644 |  0.032 |     0.05  |      7 |  10 |        0.105 |
| MRCP + envelope vs broadband, all     | r                   | test      |  0.646 |  0.588 |  0.058 |     0.067 |      8 |  10 |        0.02  |
| MRCP + envelope vs broadband, all     | r2                  | inner_val |  0.262 |  0.255 |  0.007 |     0.062 |      6 |  10 |        0.77  |
| MRCP + envelope vs broadband, all     | r2                  | test      |  0.202 |  0.075 |  0.126 |     0.237 |      7 |  10 |        0.084 |
| MRCP + envelope vs broadband, all     | residual_r_specific | inner_val |  0.142 |  0.161 | -0.019 |     0.017 |      1 |  10 |        0.01  |
| MRCP + envelope vs broadband, all     | residual_r_specific | test      |  0.135 |  0.134 |  0     |     0.03  |      5 |  10 |        1     |
| MRCP + envelope vs MRCP, all          | r                   | inner_val |  0.676 |  0.621 |  0.055 |     0.055 |      8 |  10 |        0.02  |
| MRCP + envelope vs MRCP, all          | r                   | test      |  0.646 |  0.554 |  0.092 |     0.07  |     10 |  10 |        0.002 |
| MRCP + envelope vs MRCP, all          | r2                  | inner_val |  0.262 |  0.237 |  0.025 |     0.06  |      7 |  10 |        0.232 |
| MRCP + envelope vs MRCP, all          | r2                  | test      |  0.202 |  0.104 |  0.098 |     0.153 |      7 |  10 |        0.064 |
| MRCP + envelope vs MRCP, all          | residual_r_specific | inner_val |  0.142 |  0.153 | -0.012 |     0.033 |      3 |  10 |        0.375 |
| MRCP + envelope vs MRCP, all          | residual_r_specific | test      |  0.135 |  0.129 |  0.006 |     0.038 |      6 |  10 |        0.77  |
| MRCP vs envelope, all                 | r                   | inner_val |  0.621 |  0.5   |  0.12  |     0.113 |      9 |  10 |        0.02  |
| MRCP vs envelope, all                 | r                   | test      |  0.554 |  0.51  |  0.043 |     0.162 |      7 |  10 |        0.275 |
| MRCP vs envelope, all                 | r2                  | inner_val |  0.237 |  0.043 |  0.194 |     0.147 |      9 |  10 |        0.006 |
| MRCP vs envelope, all                 | r2                  | test      |  0.104 |  0.044 |  0.06  |     0.308 |      6 |  10 |        0.322 |
| MRCP vs envelope, all                 | residual_r_specific | inner_val |  0.153 |  0.022 |  0.131 |     0.039 |     10 |  10 |        0.002 |
| MRCP vs envelope, all                 | residual_r_specific | test      |  0.129 |  0.018 |  0.111 |     0.037 |     10 |  10 |        0.002 |
| no prefrontal vs all, broadband       | r                   | inner_val |  0.635 |  0.644 | -0.009 |     0.019 |      2 |  10 |        0.16  |
| no prefrontal vs all, broadband       | r                   | test      |  0.57  |  0.588 | -0.017 |     0.036 |      4 |  10 |        0.232 |
| no prefrontal vs all, broadband       | r2                  | inner_val |  0.244 |  0.255 | -0.011 |     0.024 |      5 |  10 |        0.232 |
| no prefrontal vs all, broadband       | r2                  | test      |  0.037 |  0.075 | -0.039 |     0.075 |      3 |  10 |        0.193 |
| no prefrontal vs all, broadband       | residual_r_specific | inner_val |  0.155 |  0.161 | -0.005 |     0.02  |      3 |  10 |        0.432 |
| no prefrontal vs all, broadband       | residual_r_specific | test      |  0.127 |  0.134 | -0.007 |     0.021 |      5 |  10 |        0.375 |
| sensorimotor vs all, broadband        | r                   | inner_val |  0.574 |  0.644 | -0.069 |     0.032 |      0 |  10 |        0.002 |
| sensorimotor vs all, broadband        | r                   | test      |  0.526 |  0.588 | -0.062 |     0.051 |      1 |  10 |        0.01  |
| sensorimotor vs all, broadband        | r2                  | inner_val |  0.142 |  0.255 | -0.113 |     0.063 |      1 |  10 |        0.004 |
| sensorimotor vs all, broadband        | r2                  | test      | -0.013 |  0.075 | -0.088 |     0.128 |      1 |  10 |        0.084 |
| sensorimotor vs all, broadband        | residual_r_specific | inner_val |  0.081 |  0.161 | -0.079 |     0.026 |      0 |  10 |        0.002 |
| sensorimotor vs all, broadband        | residual_r_specific | test      |  0.073 |  0.134 | -0.062 |     0.034 |      0 |  10 |        0.002 |
| sensorimotor vs prefrontal, broadband | r                   | inner_val |  0.574 |  0.414 |  0.161 |     0.051 |     10 |  10 |        0.002 |
| sensorimotor vs prefrontal, broadband | r                   | test      |  0.526 |  0.362 |  0.164 |     0.099 |     10 |  10 |        0.002 |
| sensorimotor vs prefrontal, broadband | r2                  | inner_val |  0.142 |  0.042 |  0.1   |     0.068 |      9 |  10 |        0.004 |
| sensorimotor vs prefrontal, broadband | r2                  | test      | -0.013 | -0.028 |  0.015 |     0.267 |      7 |  10 |        0.432 |
| sensorimotor vs prefrontal, broadband | residual_r_specific | inner_val |  0.081 |  0.071 |  0.011 |     0.019 |      7 |  10 |        0.105 |
| sensorimotor vs prefrontal, broadband | residual_r_specific | test      |  0.073 |  0.062 |  0.01  |     0.044 |      6 |  10 |        0.492 |
| prefrontal vs contra, broadband       | r                   | inner_val |  0.414 |  0.537 | -0.124 |     0.07  |      1 |  10 |        0.004 |
| prefrontal vs contra, broadband       | r                   | test      |  0.362 |  0.489 | -0.126 |     0.091 |      0 |  10 |        0.002 |
| prefrontal vs contra, broadband       | r2                  | inner_val |  0.042 |  0.117 | -0.075 |     0.065 |      2 |  10 |        0.02  |
| prefrontal vs contra, broadband       | r2                  | test      | -0.028 | -0.025 | -0.003 |     0.181 |      3 |  10 |        0.695 |
| prefrontal vs contra, broadband       | residual_r_specific | inner_val |  0.071 |  0.07  |  0     |     0.017 |      6 |  10 |        0.846 |
| prefrontal vs contra, broadband       | residual_r_specific | test      |  0.062 |  0.064 | -0.001 |     0.037 |      4 |  10 |        0.625 |
| prefrontal vs contra, MRCP            | r                   | inner_val |  0.414 |  0.477 | -0.063 |     0.04  |      0 |  10 |        0.002 |
| prefrontal vs contra, MRCP            | r                   | test      |  0.357 |  0.441 | -0.084 |     0.097 |      2 |  10 |        0.02  |
| prefrontal vs contra, MRCP            | r2                  | inner_val |  0.046 |  0.097 | -0.051 |     0.04  |      2 |  10 |        0.01  |
| prefrontal vs contra, MRCP            | r2                  | test      | -0.006 |  0.051 | -0.057 |     0.085 |      3 |  10 |        0.027 |
| prefrontal vs contra, MRCP            | residual_r_specific | inner_val |  0.065 |  0.077 | -0.012 |     0.018 |      2 |  10 |        0.049 |
| prefrontal vs contra, MRCP            | residual_r_specific | test      |  0.059 |  0.072 | -0.012 |     0.025 |      3 |  10 |        0.16  |
