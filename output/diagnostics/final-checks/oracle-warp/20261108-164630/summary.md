# Oracle-warp ceiling

ANALYSIS ONLY: warp parameters are fitted to each test trial's own trajectory. Base shape: the fold's cue-locked template. Per-axis A and c do not change per-trial r. Mean over test subjects; d_* and n_better_* are paired against template_cue. EEG decoders: mean over seeds per subject.

| method                   | kind             |   n_params |   r_mean |   r_sd |   r2_mean |   r2_sd |   nrmse_mean |   nrmse_sd |   residual_r_specific_mean |   residual_r_specific_sd |   d_r_vs_cue |   n_better_r |   d_r2_vs_cue |   n_better_r2 |
|:-------------------------|:-----------------|-----------:|---------:|-------:|----------:|--------:|-------------:|-----------:|---------------------------:|-------------------------:|-------------:|-------------:|--------------:|--------------:|
| template_cue             | floor            |          0 |    0.803 |  0.1   |     0.524 |   0.162 |        0.216 |      0.033 |                    nan     |                  nan     |        0     |            0 |         0     |             0 |
| template                 | oracle_reference |        nan |    0.864 |  0.042 |     0.622 |   0.092 |        0.193 |      0.022 |                      0.195 |                    0.121 |        0.061 |           10 |         0.097 |            10 |
| tau                      | oracle_warp      |          1 |    0.879 |  0.024 |     0.65  |   0.066 |        0.187 |      0.016 |                      0.171 |                    0.159 |        0.076 |           10 |         0.126 |            10 |
| tau_scale                | oracle_warp      |          2 |    0.916 |  0.018 |     0.734 |   0.055 |        0.163 |      0.013 |                      0.291 |                    0.173 |        0.114 |           10 |         0.21  |            10 |
| tau_scale_amp            | oracle_warp      |          5 |    0.918 |  0.019 |     0.84  |   0.034 |        0.129 |      0.013 |                      0.32  |                    0.174 |        0.115 |           10 |         0.316 |            10 |
| tau_scale_amp_offset     | oracle_warp      |          8 |    0.918 |  0.018 |     0.846 |   0.032 |        0.127 |      0.012 |                      0.324 |                    0.174 |        0.116 |           10 |         0.322 |            10 |
| piecewise_landmark       | oracle_warp      |          6 |    0.914 |  0.02  |     0.837 |   0.036 |        0.132 |      0.015 |                      0.264 |                    0.163 |        0.112 |           10 |         0.313 |            10 |
| piecewise_fit            | oracle_warp      |         11 |    0.923 |  0.02  |     0.854 |   0.037 |        0.123 |      0.015 |                      0.324 |                    0.179 |        0.121 |           10 |         0.33  |            10 |
| min_jerk                 | oracle_warp      |         23 |    0.977 |  0.009 |     0.957 |   0.017 |        0.047 |      0.008 |                      0.568 |                    0.214 |        0.175 |           10 |         0.433 |            10 |
| cnn_lstm_direct_2s       | eeg_decoder      |        nan |    0.588 |  0.153 |     0.075 |   0.322 |        0.3   |      0.051 |                      0.134 |                    0.064 |       -0.215 |            1 |        -0.449 |             1 |
| cnn_lstm_residual_2s     | eeg_decoder      |        nan |    0.813 |  0.084 |     0.51  |   0.132 |        0.218 |      0.026 |                      0.192 |                    0.062 |        0.01  |            5 |        -0.014 |             4 |
| cnn_lstm_direct_2s_ica   | eeg_decoder      |        nan |    0.485 |  0.128 |    -0.083 |   0.31  |        0.33  |      0.045 |                      0.068 |                    0.068 |       -0.318 |            1 |        -0.607 |             1 |
| cnn_lstm_residual_2s_ica | eeg_decoder      |        nan |    0.796 |  0.098 |     0.487 |   0.152 |        0.225 |      0.03  |                      0.08  |                    0.047 |       -0.006 |            4 |        -0.037 |             1 |

## Median over subjects of the per-subject Spearman rho

| variant              |   rho_tau_rt |   rho_scale_duration |
|:---------------------|-------------:|---------------------:|
| tau                  |        0.864 |              nan     |
| tau_scale            |        0.83  |                0.887 |
| tau_scale_amp        |        0.879 |                0.963 |
| tau_scale_amp_offset |        0.862 |                0.963 |

## Fits at a bound

| variant              | parameter         |   n_trials |   frac |
|:---------------------|:------------------|-----------:|-------:|
| tau                  | tau_s             |         10 | 0.0034 |
| tau_scale            | scale             |         14 | 0.0048 |
| tau_scale            | tau_s             |          3 | 0.001  |
| tau_scale_amp        | scale             |         12 | 0.0041 |
| tau_scale_amp        | tau_s             |          3 | 0.001  |
| tau_scale_amp        | amp_z             |          1 | 0.0003 |
| tau_scale_amp_offset | scale             |         12 | 0.0041 |
| tau_scale_amp_offset | tau_s             |          3 | 0.001  |
| tau_scale_amp_offset | amp_z             |          2 | 0.0007 |
| tau_scale_amp_offset | offset_y          |          2 | 0.0007 |
| tau_scale_amp_offset | amp_x             |          1 | 0.0003 |
| tau_scale_amp_offset | offset_x          |          1 | 0.0003 |
| piecewise_fit        | bp_handstop       |       2685 | 0.9133 |
| piecewise_fit        | bp_replace        |       1607 | 0.5466 |
| piecewise_fit        | bp_touch          |        588 | 0.2    |
| piecewise_fit        | bp_liftoff        |         66 | 0.0224 |
| piecewise_fit        | bp_onset          |         21 | 0.0071 |
| min_jerk             | lower_start_s     |        165 | 0.0561 |
| min_jerk             | return_start_s    |        160 | 0.0544 |
| min_jerk             | lift_start_s      |         52 | 0.0177 |
| min_jerk             | return_duration_s |         26 | 0.0088 |
| min_jerk             | reach_start_s     |          3 | 0.001  |
| min_jerk             | lower_duration_s  |          1 | 0.0003 |
