# Decoder loss-weight sweep (screening folds P3, P6)

Selection score: mean r over P3 and P6 (per trial, per axis). Tie rule (decided 2026-11-15): differences < 0.005 keep the base / default value. `floor_r`: template_cue on the same samples. `head_floor_r`: cue phi (u = 0: landmark templates at clip((t - t_LED - tau0) / s0, 0, 1), uniform mixing, A = 1, Delta = 0); NaN for runs before SSL-19.

| head        |   aux |   delta |   amp |   gate |      r |   r_P3 |   r_P6 |     rrs |   rrs_P3 |   rrs_P6 |   nrmse |     r2 |   floor_r |   head_floor_r | lr          | best_epochs   |   minutes | run                                                                   |
|:------------|------:|--------:|------:|-------:|-------:|-------:|-------:|--------:|---------:|---------:|--------:|-------:|----------:|---------------:|:------------|:--------------|----------:|:----------------------------------------------------------------------|
| global_warp |   0.1 |    0.01 |   0.1 |    0.1 | 0.7241 | 0.6663 | 0.7819 | -0.0837 |  -0.0345 |  -0.1329 |  0.2867 | 0.2447 |     0.874 |         0.8087 | 0.001 0.001 | 1 1           |       0.2 | 20261115-161300_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening_smoke |
| proposed    |   0.1 |    0.01 |   0.1 |    0.1 | 0.6797 | 0.6061 | 0.7533 | -0.0614 |  -0.0304 |  -0.0923 |  0.3094 | 0.1299 |     0.874 |         0.8087 | 0.001 0.001 | 1 1           |       0.2 | 20261115-161237_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening_smoke |
