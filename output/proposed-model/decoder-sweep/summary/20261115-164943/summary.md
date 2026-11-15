# Decoder loss-weight sweep (screening folds P3, P6)

Selection score: mean r over P3 and P6 (per trial, per axis). Tie rule (decided 2026-11-15): differences < 0.005 keep the base / default value. `floor_r`: template_cue on the same samples. `head_floor_r`: cue phi (u = 0: landmark templates at clip((t - t_LED - tau0) / s0, 0, 1), uniform mixing, A = 1, Delta = 0); NaN for runs before SSL-19.

| head        |   aux |   delta |   amp |   gate |      r |   r_P3 |   r_P6 |     rrs |   rrs_P3 |   rrs_P6 |   nrmse |     r2 |   floor_r |   head_floor_r | lr           | best_epochs   |   minutes | run                                                                   |
|:------------|------:|--------:|------:|-------:|-------:|-------:|-------:|--------:|---------:|---------:|--------:|-------:|----------:|---------------:|:-------------|:--------------|----------:|:----------------------------------------------------------------------|
| residual    |   0.1 |    0.01 |   0.1 |    0.1 | 0.8764 | 0.8788 | 0.8739 |  0.047  |   0.0917 |   0.0024 |  0.2249 | 0.5033 |     0.874 |         0.8722 | 0.001 0.001  | 2 2           |       0.2 | 20261115-164758_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening_smoke |
| global_warp |   0.1 |    0.01 |   0.1 |    0.1 | 0.8261 | 0.7689 | 0.8833 | -0.0298 |   0.0196 |  -0.0792 |  0.2419 | 0.44   |     0.874 |         0.8722 | 0.001 0.001  | 2 1           |       0.2 | 20261115-164741_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening_smoke |
| proposed    |   0.1 |    0.01 |   0.1 |    0.1 | 0.8255 | 0.7543 | 0.8966 | -0.0502 |  -0.0119 |  -0.0885 |  0.2433 | 0.4263 |     0.874 |         0.8722 | 0.001 0.0003 | 2 2           |       0.2 | 20261115-164724_seed1_aux0.1_delta0.01_amp0.1_gate0.1_screening_smoke |
