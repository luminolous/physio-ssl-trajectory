# Probe of pretraining checkpoints on P3/P6

`raw`: broadband CAR input (training input); `ica`: ICA-cleaned cache. Selection score `within_rrs` = within-subject (leave-one-series-out) probe `residual_r_specific`, mean of P3 and P6. `within_matched`: the same probe scored with `residual_r_matched` (design.md SSL-27).

## Random-init encoders (mean and SD over seeds [0])

| input   |   within_matched_mean |   within_matched_std |   within_matched_P3_mean |   within_matched_P3_std |   within_matched_P6_mean |   within_matched_P6_std |   within_rrs_mean |   within_rrs_std |   within_rrs_P3_mean |   within_rrs_P3_std |   within_rrs_P6_mean |   within_rrs_P6_std |   within_r_mean |   within_r_std |   cross_matched_mean |   cross_matched_std |   cross_rrs_mean |   cross_rrs_std |   cross_r_mean |   cross_r_std |   emb_sd_min_mean |   emb_sd_min_std |   eff_rank_mean |   eff_rank_std |
|:--------|----------------------:|---------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|------------------:|-----------------:|---------------------:|--------------------:|---------------------:|--------------------:|----------------:|---------------:|---------------------:|--------------------:|-----------------:|----------------:|---------------:|--------------:|------------------:|-----------------:|----------------:|---------------:|
| ica     |                0.0281 |                  nan |                   0.0264 |                     nan |                   0.0299 |                     nan |            0.0188 |              nan |               0.0096 |                 nan |               0.0281 |                 nan |          0.5222 |            nan |              -0.0026 |                 nan |          -0.0184 |             nan |         0.1385 |           nan |            0.3535 |              nan |         49.6495 |            nan |
| raw     |                0.0537 |                  nan |                   0.0384 |                     nan |                   0.0689 |                     nan |            0.0477 |              nan |               0.0245 |                 nan |               0.0708 |                 nan |          0.6382 |            nan |               0.0005 |                 nan |          -0.0152 |             nan |         0.2396 |           nan |            0.3197 |              nan |         49.323  |            nan |

## Checkpoints



## All rows

| input   | encoder      | run   | kind   |   step |   within_matched |   within_matched_P3 |   within_matched_P6 |   within_rrs |   within_rrs_P3 |   within_rrs_P6 |   within_r |   cross_matched |   cross_rrs |   cross_r |   emb_sd_min |   eff_rank |
|:--------|:-------------|:------|:-------|-------:|-----------------:|--------------------:|--------------------:|-------------:|----------------:|----------------:|-----------:|----------------:|------------:|----------:|-------------:|-----------:|
| raw     | random_seed0 | -     | init   |      0 |           0.0537 |              0.0384 |              0.0689 |       0.0477 |          0.0245 |          0.0708 |     0.6382 |          0.0005 |     -0.0152 |    0.2396 |       0.3197 |    49.323  |
| ica     | random_seed0 | -     | init   |      0 |           0.0281 |              0.0264 |              0.0299 |       0.0188 |          0.0096 |          0.0281 |     0.5222 |         -0.0026 |     -0.0184 |    0.1385 |       0.3535 |    49.6495 |
