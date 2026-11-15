# Decoder loss-weight sweep (screening folds P3, P6)

Selection score: mean r over P3 and P6 (per trial, per axis). Tie rule (decided 2026-11-15): differences < 0.005 keep the base / default value. `floor_r`: template_cue on the same samples. `head_floor_r`: cue phi (u = 0: landmark templates at clip((t - t_LED - tau0) / s0, 0, 1), uniform mixing, A = 1, Delta = 0); NaN for runs before SSL-19.

| head        |   aux |   delta |   amp |   gate |      r |   r_P3 |   r_P6 |    rrs |   rrs_P3 |   rrs_P6 |   nrmse |     r2 |   floor_r |   head_floor_r | lr           | best_epochs   |   minutes | run                                                             |
|:------------|------:|--------:|------:|-------:|-------:|-------:|-------:|-------:|---------:|---------:|--------:|-------:|----------:|---------------:|:-------------|:--------------|----------:|:----------------------------------------------------------------|
| residual    |   0.1 |    0.01 |   0.1 |    0.1 | 0.869  | 0.834  | 0.9039 | 0.1133 |   0.0611 |   0.1655 |  0.1979 | 0.6379 |    0.8626 |         0.8495 | 0.0003 0.001 | 18 2          |       3.3 | 20261115-163006_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening |
| global_warp |   0.1 |    0.01 |   0.1 |    0.1 | 0.8107 | 0.7743 | 0.8471 | 0.0548 |   0.0417 |   0.0679 |  0.226  | 0.5205 |    0.8626 |         0.8495 | 0.001 0.001  | 5 4           |       3.4 | 20261115-162634_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening |
| proposed    |   0.1 |    0.01 |   0.1 |    0.1 | 0.7999 | 0.7515 | 0.8483 | 0.0219 |  -0.0124 |   0.0562 |  0.2314 | 0.5004 |    0.8626 |         0.8495 | 0.001 0.001  | 2 5           |       3.8 | 20261115-162238_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening |
