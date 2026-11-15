# Phase gap diagnostic (screening folds P3, P6)

| row                                                            | kind    |      r |   r_P3 |   r_P6 |   residual_r_specific |   nrmse |     r2 |
|:---------------------------------------------------------------|:--------|-------:|-------:|-------:|----------------------:|--------:|-------:|
| template_cue (floor)                                           | floor   | 0.8626 | 0.8294 | 0.8958 |              nan      |  0.199  | 0.6374 |
| 1. oracle phi, true condition, A=1, Delta=0                    | oracle  | 0.9829 | 0.9792 | 0.9866 |                0.4921 |  0.1087 | 0.8679 |
| 1b. oracle phi, uniform mixing                                 | oracle  | 0.983  | 0.9794 | 0.9866 |                0.4897 |  0.1083 | 0.8686 |
| 2. cue phi (median RT, median duration), uniform               | no EEG  | 0.8495 | 0.7997 | 0.8993 |               -0.0196 |  0.2061 | 0.5873 |
| 2b. cue phi with runner s0, uniform                            | no EEG  | 0.8489 | 0.7978 | 0.9001 |               -0.0231 |  0.2063 | 0.5874 |
| 2c. head floor: phase prior N(mu(t), sd(t)), uniform (SSL-20)  | no EEG  | 0.8613 | 0.8295 | 0.8932 |                0.0004 |  0.1988 | 0.6401 |
| 4. proposed (20261115-165121)                                  | decoder | 0.8599 | 0.8275 | 0.8923 |                0.1153 |  0.202  | 0.6187 |
| 4. proposed: phi -> oracle phi, sigma 0 (own A, mix, Delta)    | oracle  | 0.9764 | 0.9731 | 0.9798 |                0.4706 |  0.1186 | 0.8456 |
| 4. proposed: phi -> cue phi, sigma 0 (own A, mix, Delta)       | no EEG  | 0.8566 | 0.8176 | 0.8957 |                0.0366 |  0.2059 | 0.5891 |
| 4. proposed: phi -> phase prior (own A, mix, Delta)            | no EEG  | 0.8709 | 0.846  | 0.8957 |                0.0866 |  0.198  | 0.6412 |
| 4. proposed: own phi and sigma, A=1, uniform, Delta=0          | decoder | 0.8504 | 0.8103 | 0.8904 |                0.0792 |  0.2035 | 0.6138 |
| 4. global_warp (20261115-165537)                               | decoder | 0.8484 | 0.8071 | 0.8897 |                0.1299 |  0.2099 | 0.5866 |
| 4. global_warp: phi -> oracle phi, sigma 0 (own A, mix, Delta) | oracle  | 0.9763 | 0.9727 | 0.9799 |                0.4701 |  0.1216 | 0.8379 |
| 4. global_warp: phi -> cue phi, sigma 0 (own A, mix, Delta)    | no EEG  | 0.8518 | 0.8092 | 0.8944 |                0.0242 |  0.2091 | 0.5764 |
| 4. global_warp: phi -> phase prior (own A, mix, Delta)         | no EEG  | 0.866  | 0.8376 | 0.8943 |                0.0548 |  0.2016 | 0.6273 |
| 4. global_warp: own phi and sigma, A=1, uniform, Delta=0       | decoder | 0.8435 | 0.7982 | 0.8888 |                0.114  |  0.2085 | 0.5941 |
| 4. residual (20261115-170139)                                  | decoder | 0.869  | 0.834  | 0.9039 |                0.1134 |  0.1979 | 0.6379 |

## Phase timing (s; lag = predicted - landmark crossing; trial_corr = across-lift correlation of crossing times)

| head        | subject   |   phi_mae |   phi_pred_at_onset |   phi_pred_pre_onset_max |   phi_pred_at_stop |   frac_reach_0.95 |   lag_0.05_median_s |   lag_0.05_iqr_s |   trial_corr_0.05 |   lag_0.5_median_s |   lag_0.5_iqr_s |   trial_corr_0.5 |   lag_0.95_median_s |   lag_0.95_iqr_s |   trial_corr_0.95 |
|:------------|:----------|----------:|--------------------:|-------------------------:|-------------------:|------------------:|--------------------:|-----------------:|------------------:|-------------------:|----------------:|-----------------:|--------------------:|-----------------:|------------------:|
| cue_phi     | P3        |     0.101 |               0     |                    0     |              0.821 |             0.024 |               0.144 |            0.063 |            -0     |              0.669 |           0.218 |          nan     |               0.172 |            0.267 |           nan     |
| cue_phi     | P6        |     0.039 |               0.002 |                    0.001 |              1     |             0.827 |              -0.004 |            0.151 |             0.959 |             -0.023 |           0.436 |            0.79  |              -0.137 |            0.49  |             0.81  |
| global_warp | P3        |     0.127 |               0     |                    0     |              0.83  |             0     |               0.22  |            0.156 |             0.016 |              0.907 |           0.85  |           -0.082 |             nan     |          nan     |           nan     |
| global_warp | P6        |     0.06  |               0.003 |                    0.002 |              0.909 |             0.024 |              -0.008 |            0.19  |             0.928 |             -0.062 |           0.826 |            0.294 |               0.059 |            0.507 |             0.952 |
| proposed    | P3        |     0.113 |               0     |                    0     |              0.794 |             0     |               0.157 |            0.164 |             0.151 |              0.767 |           0.334 |            0.135 |             nan     |          nan     |           nan     |
| proposed    | P6        |     0.051 |               0     |                    0     |              0.894 |             0.003 |              -0.011 |            0.228 |             0.903 |              0.045 |           0.619 |            0.455 |              -4.499 |            0     |           nan     |

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
 "sigma": {
  "proposed_P3": {
   "median_sigma": 0.047899095341563225,
   "median_prior_sd": 0.09072445333003999,
   "median_sigma_over_prior": 0.6087560762754529
  },
  "global_warp_P3": {
   "median_sigma": 0.043682465329766273,
   "median_prior_sd": 0.09072445333003999,
   "median_sigma_over_prior": 0.5416772250644215
  },
  "proposed_P6": {
   "median_sigma": 0.05492076277732849,
   "median_prior_sd": 0.09455687552690506,
   "median_sigma_over_prior": 0.6408425701800045
  },
  "global_warp_P6": {
   "median_sigma": 0.046258047223091125,
   "median_prior_sd": 0.09455687552690506,
   "median_sigma_over_prior": 0.5577580536510854
  }
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
   "phi_mae": 0.12710208146061358,
   "phi_pred_at_onset": 0.0,
   "phi_pred_pre_onset_max": 0.0,
   "phi_pred_at_stop": 0.8300122618675232,
   "frac_reach_0.95": 0.0,
   "lag_0.05_median_s": 0.21998254166312223,
   "lag_0.05_iqr_s": 0.15648261924957255,
   "trial_corr_0.05": 0.016318112046113876,
   "lag_0.5_median_s": 0.9066174116933268,
   "lag_0.5_iqr_s": 0.849796872999211,
   "trial_corr_0.5": -0.08190961157113524,
   "lag_0.95_median_s": NaN,
   "lag_0.95_iqr_s": NaN,
   "trial_corr_0.95": NaN
  },
  {
   "head": "global_warp",
   "subject": "P6",
   "phi_mae": 0.059837839557183484,
   "phi_pred_at_onset": 0.0034868381917476654,
   "phi_pred_pre_onset_max": 0.00233356689568609,
   "phi_pred_at_stop": 0.9089224934577942,
   "frac_reach_0.95": 0.023809523809523808,
   "lag_0.05_median_s": -0.008142980891779783,
   "lag_0.05_iqr_s": 0.19011869615724925,
   "trial_corr_0.05": 0.9284297563506185,
   "lag_0.5_median_s": -0.06213326478349135,
   "lag_0.5_iqr_s": 0.8260404979930331,
   "trial_corr_0.5": 0.2942615716866449,
   "lag_0.95_median_s": 0.059086320025738104,
   "lag_0.95_iqr_s": 0.5069477996572713,
   "trial_corr_0.95": 0.9515786329030711
  },
  {
   "head": "proposed",
   "subject": "P3",
   "phi_mae": 0.11270337537514082,
   "phi_pred_at_onset": 0.0,
   "phi_pred_pre_onset_max": 0.0,
   "phi_pred_at_stop": 0.7938293218612671,
   "frac_reach_0.95": 0.0,
   "lag_0.05_median_s": 0.15705068700607694,
   "lag_0.05_iqr_s": 0.16355106654053211,
   "trial_corr_0.05": 0.15071561644044126,
   "lag_0.5_median_s": 0.7673580488832337,
   "lag_0.5_iqr_s": 0.333943268906273,
   "trial_corr_0.5": 0.13521055732325588,
   "lag_0.95_median_s": NaN,
   "lag_0.95_iqr_s": NaN,
   "trial_corr_0.95": NaN
  },
  {
   "head": "proposed",
   "subject": "P6",
   "phi_mae": 0.05065678019905223,
   "phi_pred_at_onset": 0.0,
   "phi_pred_pre_onset_max": 0.0,
   "phi_pred_at_stop": 0.8935722708702087,
   "frac_reach_0.95": 0.003401360544217687,
   "lag_0.05_median_s": -0.011137262723439456,
   "lag_0.05_iqr_s": 0.22840935081464653,
   "trial_corr_0.05": 0.9032922128332695,
   "lag_0.5_median_s": 0.04466615146674169,
   "lag_0.5_iqr_s": 0.6191548772861186,
   "trial_corr_0.5": 0.4551741729482329,
   "lag_0.95_median_s": -4.499018039451675,
   "lag_0.95_iqr_s": 0.0,
   "trial_corr_0.95": NaN
  }
 ]
}
```
