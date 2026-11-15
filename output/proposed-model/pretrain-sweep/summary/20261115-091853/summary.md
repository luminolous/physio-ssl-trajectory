# Pretraining budget sweep (lejepa_guided)

Selection score: `within_rrs`, the within-subject (leave-one-series-out) linear-probe `residual_r_specific` on P3 and P6, final online encoder. Rows sorted by it. `within_rrs_step0`: the same probe on the random-init encoder (reference for the probe noise level).

|   lam |   steps |   within_rrs |   within_rrs_P3 |   within_rrs_P6 |   within_rrs_step0 |   within_r |   cross_rrs |   cross_r |   emb_sd_min |   eff_rank |   final_pred |   final_sigreg |   train_minutes | run                               |
|------:|--------:|-------------:|----------------:|----------------:|-------------------:|-----------:|------------:|----------:|-------------:|-----------:|-------------:|---------------:|----------------:|:----------------------------------|
|   0.1 |     200 |       0.0486 |          0.0307 |          0.0666 |             0.0477 |     0.6181 |     -0.0158 |    0.2243 |       0.5739 |     72.617 |       0.6672 |        12.9522 |             0.4 | 20261115-091600_smoke_lam0.1_s200 |

## within_rrs by lam x steps

|   lam |    200 |
|------:|-------:|
|   0.1 | 0.0486 |
