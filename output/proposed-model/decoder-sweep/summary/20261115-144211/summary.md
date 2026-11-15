# Decoder loss-weight sweep (screening folds P3, P6)

Selection score: mean r over P3 and P6 (per trial, per axis). Tie rule (decided 2026-11-15): differences < 0.005 keep the base / default value. `floor_r`: template_cue on the same samples.

|   aux |   delta |   amp |   gate |      r |   r_P3 |   r_P6 |     rrs |   rrs_P3 |   rrs_P6 |   nrmse |     r2 |   floor_r | lr          | best_epochs   |   minutes | run                                                                 |
|------:|--------:|------:|-------:|-------:|-------:|-------:|--------:|---------:|---------:|--------:|-------:|----------:|:------------|:--------------|----------:|:--------------------------------------------------------------------|
|     1 |    0.01 |   0.1 |    0.1 | 0.4439 | 0.5063 | 0.3814 | -0.0605 |  -0.0572 |  -0.0639 |   0.324 | 0.0461 |     0.874 | 0.001 0.001 | 2 2           |       0.2 | 20261115-144157_seed1_aux1_delta0.01_amp0.1_gate0.1_screening_smoke |
