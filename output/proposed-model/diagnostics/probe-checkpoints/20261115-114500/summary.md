# Probe of pretraining checkpoints on P3/P6

`raw`: broadband CAR input (training input); `ica`: ICA-cleaned cache. Selection score `within_rrs` = within-subject (leave-one-series-out) probe `residual_r_specific`, mean of P3 and P6.

## Random-init encoders (mean and SD over seeds [0, 1, 2])

| input   |   within_rrs_mean |   within_rrs_std |   within_rrs_P3_mean |   within_rrs_P3_std |   within_rrs_P6_mean |   within_rrs_P6_std |   within_r_mean |   within_r_std |   cross_rrs_mean |   cross_rrs_std |   cross_r_mean |   cross_r_std |   emb_sd_min_mean |   emb_sd_min_std |   eff_rank_mean |   eff_rank_std |
|:--------|------------------:|-----------------:|---------------------:|--------------------:|---------------------:|--------------------:|----------------:|---------------:|-----------------:|----------------:|---------------:|--------------:|------------------:|-----------------:|----------------:|---------------:|
| ica     |            0.02   |           0.0052 |               0.0146 |              0.0078 |               0.0253 |              0.0045 |          0.5069 |         0.0153 |          -0.0197 |          0.0011 |         0.1308 |        0.0148 |            0.3106 |           0.0383 |         50.2434 |         0.5911 |
| raw     |            0.0501 |           0.0041 |               0.0283 |              0.0037 |               0.0718 |              0.0055 |          0.6319 |         0.0142 |          -0.0138 |          0.0048 |         0.2742 |        0.0395 |            0.2856 |           0.0332 |         50.1303 |         0.9402 |

## Checkpoints

|                                    |   cross_rrs_ica |   cross_rrs_raw |   within_r_ica |   within_r_raw |   within_rrs_ica |   within_rrs_raw |
|:-----------------------------------|----------------:|----------------:|---------------:|---------------:|-----------------:|-----------------:|
| ('lejepa_unguided', 'online', 40)  |         -0.0162 |         -0.014  |         0.5129 |         0.6302 |           0.0228 |           0.0468 |
| ('lejepa_unguided', 'online', 80)  |         -0.0164 |         -0.0151 |         0.513  |         0.628  |           0.0229 |           0.0483 |
| ('lejepa_unguided', 'online', 120) |         -0.0159 |         -0.015  |         0.5129 |         0.6275 |           0.0239 |           0.0487 |

## All rows

| input   | encoder         | kind   |   step |   within_rrs |   within_rrs_P3 |   within_rrs_P6 |   within_r |   cross_rrs |   cross_r |   emb_sd_min |   eff_rank |
|:--------|:----------------|:-------|-------:|-------------:|----------------:|----------------:|-----------:|------------:|----------:|-------------:|-----------:|
| raw     | random_seed0    | init   |      0 |       0.0477 |          0.0245 |          0.0708 |     0.6382 |     -0.0152 |    0.2396 |       0.3197 |    49.323  |
| raw     | random_seed1    | init   |      0 |       0.0477 |          0.0286 |          0.0669 |     0.6156 |     -0.0085 |    0.3172 |       0.2838 |    51.1625 |
| raw     | random_seed2    | init   |      0 |       0.0548 |          0.0319 |          0.0778 |     0.6419 |     -0.0178 |    0.2658 |       0.2534 |    49.9055 |
| raw     | lejepa_unguided | online |     40 |       0.0468 |          0.0265 |          0.0671 |     0.6302 |     -0.014  |    0.23   |       0.601  |    58.8493 |
| raw     | lejepa_unguided | online |     80 |       0.0483 |          0.0276 |          0.0689 |     0.628  |     -0.0151 |    0.2246 |       0.5492 |    64.1712 |
| raw     | lejepa_unguided | online |    120 |       0.0487 |          0.0287 |          0.0688 |     0.6275 |     -0.015  |    0.2291 |       0.5473 |    65.2887 |
| ica     | random_seed0    | init   |      0 |       0.0188 |          0.0096 |          0.0281 |     0.5222 |     -0.0184 |    0.1385 |       0.3535 |    49.6495 |
| ica     | random_seed1    | init   |      0 |       0.0154 |          0.0107 |          0.0201 |     0.4917 |     -0.0206 |    0.1403 |       0.2981 |    50.8315 |
| ica     | random_seed2    | init   |      0 |       0.0257 |          0.0237 |          0.0278 |     0.507  |     -0.0201 |    0.1137 |       0.2801 |    50.2492 |
| ica     | lejepa_unguided | online |     40 |       0.0228 |          0.018  |          0.0275 |     0.5129 |     -0.0162 |    0.1382 |       0.6259 |    58.422  |
| ica     | lejepa_unguided | online |     80 |       0.0229 |          0.0195 |          0.0262 |     0.513  |     -0.0164 |    0.1459 |       0.5784 |    63.6074 |
| ica     | lejepa_unguided | online |    120 |       0.0239 |          0.021  |          0.0268 |     0.5129 |     -0.0159 |    0.1472 |       0.5772 |    64.6318 |
