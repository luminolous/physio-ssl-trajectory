# Phase gap diagnostic (screening folds P3, P6)

| row                                                | kind    |      r |   r_P3 |   r_P6 |   residual_r_specific |   nrmse |     r2 |
|:---------------------------------------------------|:--------|-------:|-------:|-------:|----------------------:|--------:|-------:|
| template_cue (floor)                               | floor   | 0.8626 | 0.8294 | 0.8958 |              nan      |  0.199  | 0.6374 |
| 1. oracle phi, true condition, A=1, Delta=0        | oracle  | 0.9829 | 0.9792 | 0.9866 |                0.4921 |  0.1087 | 0.8679 |
| 1b. oracle phi, uniform mixing                     | oracle  | 0.983  | 0.9794 | 0.9866 |                0.4897 |  0.1083 | 0.8686 |
| 2. cue phi (median RT, median duration), uniform   | no EEG  | 0.8495 | 0.7997 | 0.8993 |               -0.0196 |  0.2061 | 0.5873 |
| 2b. cue phi with runner s0, uniform                | no EEG  | 0.8489 | 0.7978 | 0.9001 |               -0.0231 |  0.2063 | 0.5874 |
| 4. proposed (20261115-144504)                      | decoder | 0.7033 | 0.6498 | 0.7567 |                0.0655 |  0.2613 | 0.3753 |
| 4. proposed: phi -> oracle phi (own A, mix, Delta) | oracle  | 0.9478 | 0.9421 | 0.9536 |                0.3996 |  0.1967 | 0.4399 |
| 4. proposed: phi -> cue phi (own A, mix, Delta)    | no EEG  | 0.8505 | 0.8126 | 0.8884 |                0.1074 |  0.2684 | 0.1817 |
| 4. proposed: own phi, A=1, uniform, Delta=0        | decoder | 0.4513 | 0.4585 | 0.4442 |               -0.0354 |  0.3071 | 0.1274 |

## Phase timing (s; lag = predicted - landmark crossing; trial_corr = across-lift correlation of crossing times)

| head     | subject   |   phi_mae |   phi_pred_at_onset |   phi_pred_pre_onset_max |   phi_pred_at_stop |   frac_reach_0.95 |   lag_0.05_median_s |   lag_0.05_iqr_s |   trial_corr_0.05 |   lag_0.5_median_s |   lag_0.5_iqr_s |   trial_corr_0.5 |   lag_0.95_median_s |   lag_0.95_iqr_s |   trial_corr_0.95 |
|:---------|:----------|----------:|--------------------:|-------------------------:|-------------------:|------------------:|--------------------:|-----------------:|------------------:|-------------------:|----------------:|-----------------:|--------------------:|-----------------:|------------------:|
| cue_phi  | P3        |     0.101 |               0     |                    0     |              0.821 |             0.024 |               0.144 |            0.063 |            -0     |              0.669 |           0.217 |           nan    |               0.172 |            0.26  |             -0    |
| cue_phi  | P6        |     0.039 |               0.002 |                    0.001 |              1     |             0.827 |              -0.004 |            0.151 |             0.959 |             -0.023 |           0.435 |             0.79 |              -0.137 |            0.489 |              0.81 |
| proposed | P3        |     0.36  |               0.051 |                    0.051 |              0.205 |             0     |              -0.239 |            0.437 |             0.149 |            nan     |         nan     |           nan    |             nan     |          nan     |            nan    |
| proposed | P6        |     0.312 |               0.05  |                    0.05  |              0.276 |             0     |              -0.3   |            0.495 |             0.78  |              1.967 |           0     |           nan    |             nan     |          nan     |            nan    |

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
   "lag_0.05_iqr_s": 0.06319417415700981,
   "trial_corr_0.05": -7.951261102417007e-16,
   "lag_0.5_median_s": 0.6694063422906882,
   "lag_0.5_iqr_s": 0.21657077531656477,
   "trial_corr_0.5": NaN,
   "lag_0.95_median_s": 0.17233325122832444,
   "lag_0.95_iqr_s": 0.25967955305981993,
   "trial_corr_0.95": -3.551098326808594e-15
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
   "lag_0.05_iqr_s": 0.15117840930799545,
   "trial_corr_0.05": 0.9589856630749716,
   "lag_0.5_median_s": -0.02268478476714919,
   "lag_0.5_iqr_s": 0.43507731940810646,
   "trial_corr_0.5": 0.7896837691273187,
   "lag_0.95_median_s": -0.13670890439744277,
   "lag_0.95_iqr_s": 0.48868245265939736,
   "trial_corr_0.95": 0.8103159637492853
  },
  {
   "head": "proposed",
   "subject": "P3",
   "phi_mae": 0.35960372046653793,
   "phi_pred_at_onset": 0.05086155533790588,
   "phi_pred_pre_onset_max": 0.05051416903734207,
   "phi_pred_at_stop": 0.20513418316841125,
   "frac_reach_0.95": 0.0,
   "lag_0.05_median_s": -0.23890440199727672,
   "lag_0.05_iqr_s": 0.43668206277790783,
   "trial_corr_0.05": 0.14924855030673495,
   "lag_0.5_median_s": NaN,
   "lag_0.5_iqr_s": NaN,
   "trial_corr_0.5": NaN,
   "lag_0.95_median_s": NaN,
   "lag_0.95_iqr_s": NaN,
   "trial_corr_0.95": NaN
  },
  {
   "head": "proposed",
   "subject": "P6",
   "phi_mae": 0.3115942258267746,
   "phi_pred_at_onset": 0.050467068701982504,
   "phi_pred_pre_onset_max": 0.05038517154753208,
   "phi_pred_at_stop": 0.27649354934692383,
   "frac_reach_0.95": 0.0,
   "lag_0.05_median_s": -0.2995290159249464,
   "lag_0.05_iqr_s": 0.4951185113537204,
   "trial_corr_0.05": 0.7798740807198328,
   "lag_0.5_median_s": 1.9669140990834482,
   "lag_0.5_iqr_s": 0.0,
   "trial_corr_0.5": NaN,
   "lag_0.95_median_s": NaN,
   "lag_0.95_iqr_s": NaN,
   "trial_corr_0.95": NaN
  }
 ]
}
```
