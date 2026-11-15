# Decoder loss-weight sweep (screening folds P3, P6)

Selection score: mean r over P3 and P6 (per trial, per axis). Tie rule (decided 2026-11-15): differences < 0.005 keep the base / default value. `floor_r`: template_cue on the same samples. `head_floor_r`: cue phi (u = 0: landmark templates at clip((t - t_LED - tau0) / s0, 0, 1), uniform mixing, A = 1, Delta = 0); NaN for runs before SSL-19.

| head        |   aux |   delta |   amp |   gate |      r |   r_P3 |   r_P6 |    rrs |   rrs_P3 |   rrs_P6 |   nrmse |     r2 |   floor_r |   head_floor_r | lr           | best_epochs   |   minutes | run                                                             |
|:------------|------:|--------:|------:|-------:|-------:|-------:|-------:|-------:|---------:|---------:|--------:|-------:|----------:|---------------:|:-------------|:--------------|----------:|:----------------------------------------------------------------|
| residual    |   0.1 |    0.01 |   0.1 |    0.1 | 0.869  | 0.834  | 0.9039 | 0.1134 |   0.0612 |   0.1655 |  0.1979 | 0.6379 |    0.8626 |         0.8613 | 0.0003 0.001 | 18 2          |       3.3 | 20261115-170139_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening |
| proposed    |   0.1 |    0.01 |   0.1 |    0.1 | 0.8599 | 0.8275 | 0.8923 | 0.1153 |   0.0636 |   0.167  |  0.202  | 0.6187 |    0.8626 |         0.8613 | 0.001 0.001  | 4 6           |       4.2 | 20261115-165121_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening |
| global_warp |   0.1 |    0.01 |   0.1 |    0.1 | 0.8484 | 0.8071 | 0.8897 | 0.1299 |   0.0649 |   0.1949 |  0.2099 | 0.5866 |    0.8626 |         0.8613 | 0.001 0.001  | 6 8           |       5.9 | 20261115-165537_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening |
