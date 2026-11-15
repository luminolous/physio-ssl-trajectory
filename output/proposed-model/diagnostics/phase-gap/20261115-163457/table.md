# Phase gap diagnostic (screening folds P3, P6)

| row                                                           | kind    |      r |   r_P3 |   r_P6 |   residual_r_specific |   nrmse |     r2 |
|:--------------------------------------------------------------|:--------|-------:|-------:|-------:|----------------------:|--------:|-------:|
| template_cue (floor)                                          | floor   | 0.8626 | 0.8294 | 0.8958 |              nan      |  0.199  | 0.6374 |
| 1. oracle phi, true condition, A=1, Delta=0                   | oracle  | 0.9829 | 0.9792 | 0.9866 |                0.4921 |  0.1087 | 0.8679 |
| 1b. oracle phi, uniform mixing                                | oracle  | 0.983  | 0.9794 | 0.9866 |                0.4897 |  0.1083 | 0.8686 |
| 2. cue phi = head floor (median RT, median duration), uniform | no EEG  | 0.8495 | 0.7997 | 0.8993 |               -0.0196 |  0.2061 | 0.5873 |
| 2b. cue phi with runner s0, uniform                           | no EEG  | 0.8489 | 0.7978 | 0.9001 |               -0.0231 |  0.2063 | 0.5874 |
| 4. proposed (20261115-162238)                                 | decoder | 0.7999 | 0.7515 | 0.8483 |                0.0219 |  0.2314 | 0.5004 |
| 4. proposed: phi -> oracle phi (own A, mix, Delta)            | oracle  | 0.9752 | 0.973  | 0.9774 |                0.4551 |  0.1227 | 0.8397 |
| 4. proposed: phi -> cue phi (own A, mix, Delta)               | no EEG  | 0.8636 | 0.8328 | 0.8945 |                0.066  |  0.1978 | 0.6238 |
| 4. proposed: own phi, A=1, uniform, Delta=0                   | decoder | 0.7703 | 0.7072 | 0.8334 |               -0.0429 |  0.2498 | 0.4128 |
| 4. global_warp (20261115-162634)                              | decoder | 0.8107 | 0.7743 | 0.8471 |                0.0548 |  0.226  | 0.5205 |
| 4. global_warp: phi -> oracle phi (own A, mix, Delta)         | oracle  | 0.9654 | 0.9652 | 0.9656 |                0.437  |  0.1356 | 0.8036 |
| 4. global_warp: phi -> cue phi (own A, mix, Delta)            | no EEG  | 0.865  | 0.843  | 0.887  |                0.106  |  0.2047 | 0.5969 |
| 4. global_warp: own phi, A=1, uniform, Delta=0                | decoder | 0.7588 | 0.7119 | 0.8057 |               -0.0814 |  0.2509 | 0.4095 |
| 4. residual (20261115-163006)                                 | decoder | 0.869  | 0.834  | 0.9039 |                0.1133 |  0.1979 | 0.6379 |

## Phase timing (s; lag = predicted - landmark crossing; trial_corr = across-lift correlation of crossing times)

| head        | subject   |   phi_mae |   phi_pred_at_onset |   phi_pred_pre_onset_max |   phi_pred_at_stop |   frac_reach_0.95 |   lag_0.05_median_s |   lag_0.05_iqr_s |   trial_corr_0.05 |   lag_0.5_median_s |   lag_0.5_iqr_s |   trial_corr_0.5 |   lag_0.95_median_s |   lag_0.95_iqr_s |   trial_corr_0.95 |
|:------------|:----------|----------:|--------------------:|-------------------------:|-------------------:|------------------:|--------------------:|-----------------:|------------------:|-------------------:|----------------:|-----------------:|--------------------:|-----------------:|------------------:|
| cue_phi     | P3        |     0.101 |               0     |                    0     |              0.821 |             0.024 |               0.144 |            0.063 |            -0     |              0.669 |           0.218 |          nan     |               0.172 |            0.267 |           nan     |
| cue_phi     | P6        |     0.039 |               0.002 |                    0.001 |              1     |             0.827 |              -0.004 |            0.151 |             0.959 |             -0.023 |           0.436 |            0.79  |              -0.137 |            0.49  |             0.81  |
| global_warp | P3        |     0.199 |               0     |                    0     |              0.565 |             0.007 |               0.178 |            0.18  |             0.065 |              1.904 |           0.87  |           -0.118 |               0.133 |            0.016 |           nan     |
| global_warp | P6        |     0.101 |               0.012 |                    0.011 |              0.789 |             0.218 |              -0.004 |            0.158 |             0.944 |              0.686 |           1.163 |            0.014 |              -0.023 |            0.462 |             0.511 |
| proposed    | P3        |     0.161 |               0     |                    0     |              0.699 |             0     |               0.281 |            0.236 |             0.122 |              1.19  |           0.425 |            0.044 |             nan     |          nan     |           nan     |
| proposed    | P6        |     0.064 |               0     |                    0     |              0.923 |             0.408 |               0.108 |            0.269 |             0.86  |              0.276 |           0.682 |            0.351 |              -0.037 |            0.49  |             0.426 |

## Fold info

```
{
 "P3": {
  "fold": "screening_P3",
  "median_rt_s": 0.3719999999999999,
  "median_duration_s": 5.59,
  "runner_s0_s": 5.612,
  "fractions": [
   0.0,
   0.1800152313981135,
   0.26004191727231735,
   0.8332367792978727,
   1.0
  ],
  "n_fallback_lifts": 0,
  "lifts_without_condition": 0
 },
 "P6": {
  "fold": "screening_P6",
  "median_rt_s": 0.39400000000000013,
  "median_duration_s": 5.8309999999999995,
  "runner_s0_s": 5.859,
  "fractions": [
   0.0,
   0.18637823284266353,
   0.27004957711928035,
   0.8282160464602907,
   1.0
  ],
  "n_fallback_lifts": 0,
  "lifts_without_condition": 0
 },
 "timing": [
  {
   "head": "cue_phi",
   "subject": "P3",
   "phi_mae": 0.10074232715749848,
   "phi_pred_at_onset": 0.0,
   "phi_pred_pre_onset_max": 0.0,
   "phi_pred_at_stop": 0.8207513416815743,
   "frac_reach_0.95": 0.023809523809523808,
   "lag_0.05_median_s": 0.14407327700996353,
   "lag_0.05_iqr_s": 0.06344618368532795,
   "trial_corr_0.05": -7.142658278442398e-16,
   "lag_0.5_median_s": 0.6694063422906882,
   "lag_0.5_iqr_s": 0.21772171413510577,
   "trial_corr_0.5": NaN,
   "lag_0.95_median_s": 0.17233325122832444,
   "lag_0.95_iqr_s": 0.2673354341139811,
   "trial_corr_0.95": NaN
  },
  {
   "head": "cue_phi",
   "subject": "P6",
   "phi_mae": 0.03860907146630782,
   "phi_pred_at_onset": 0.0020579660435602835,
   "phi_pred_pre_onset_max": 0.0010289830217801036,
   "phi_pred_at_stop": 1.0,
   "frac_reach_0.95": 0.826530612244898,
   "lag_0.05_median_s": -0.0035024111455784213,
   "lag_0.05_iqr_s": 0.1513103914415932,
   "trial_corr_0.05": 0.9589856630749709,
   "lag_0.5_median_s": -0.02268478476714919,
   "lag_0.5_iqr_s": 0.43571768360415053,
   "trial_corr_0.5": 0.7896837691273192,
   "lag_0.95_median_s": -0.13670890439744277,
   "lag_0.95_iqr_s": 0.4899547944734943,
   "trial_corr_0.95": 0.8103159637492858
  },
  {
   "head": "global_warp",
   "subject": "P3",
   "phi_mae": 0.19874537162428,
   "phi_pred_at_onset": 0.00019015531870536447,
   "phi_pred_pre_onset_max": 0.0,
   "phi_pred_at_stop": 0.5650210678577423,
   "frac_reach_0.95": 0.006802721088435374,
   "lag_0.05_median_s": 0.1783732397587142,
   "lag_0.05_iqr_s": 0.17959841671355609,
   "trial_corr_0.05": 0.06474555957032269,
   "lag_0.5_median_s": 1.9036685949833299,
   "lag_0.5_iqr_s": 0.8696298722206939,
   "trial_corr_0.5": -0.11757000223050072,
   "lag_0.95_median_s": 0.13260254875662847,
   "lag_0.95_iqr_s": 0.015950694859137915,
   "trial_corr_0.95": NaN
  },
  {
   "head": "global_warp",
   "subject": "P6",
   "phi_mae": 0.1006908500729691,
   "phi_pred_at_onset": 0.011664409190416353,
   "phi_pred_pre_onset_max": 0.010816830210387707,
   "phi_pred_at_stop": 0.7894971072673798,
   "frac_reach_0.95": 0.21768707482993196,
   "lag_0.05_median_s": -0.0038311838750903493,
   "lag_0.05_iqr_s": 0.1576851765310916,
   "trial_corr_0.05": 0.9442052137029694,
   "lag_0.5_median_s": 0.6855063556976,
   "lag_0.5_iqr_s": 1.1630953158876176,
   "trial_corr_0.5": 0.013965008309758834,
   "lag_0.95_median_s": -0.023352801592206873,
   "lag_0.95_iqr_s": 0.46177396947733573,
   "trial_corr_0.95": 0.5109405364109979
  },
  {
   "head": "proposed",
   "subject": "P3",
   "phi_mae": 0.16062539749584714,
   "phi_pred_at_onset": 0.0,
   "phi_pred_pre_onset_max": 0.0,
   "phi_pred_at_stop": 0.6989215314388275,
   "frac_reach_0.95": 0.0,
   "lag_0.05_median_s": 0.28135245727813496,
   "lag_0.05_iqr_s": 0.23614009012647713,
   "trial_corr_0.05": 0.12177330601142344,
   "lag_0.5_median_s": 1.1899482609603993,
   "lag_0.5_iqr_s": 0.42510174894898844,
   "trial_corr_0.5": 0.044059992101360725,
   "lag_0.95_median_s": NaN,
   "lag_0.95_iqr_s": NaN,
   "trial_corr_0.95": NaN
  },
  {
   "head": "proposed",
   "subject": "P6",
   "phi_mae": 0.06397787439760773,
   "phi_pred_at_onset": 0.0,
   "phi_pred_pre_onset_max": 0.0,
   "phi_pred_at_stop": 0.9227203726768494,
   "frac_reach_0.95": 0.40816326530612246,
   "lag_0.05_median_s": 0.10822425153766718,
   "lag_0.05_iqr_s": 0.2693466172507192,
   "trial_corr_0.05": 0.8598456098625523,
   "lag_0.5_median_s": 0.2762730147537269,
   "lag_0.5_iqr_s": 0.6822106969954019,
   "trial_corr_0.5": 0.35066685765207667,
   "lag_0.95_median_s": -0.03720928511973387,
   "lag_0.95_iqr_s": 0.4899325636278167,
   "trial_corr_0.95": 0.4263084940568952
  }
 ]
}
```
