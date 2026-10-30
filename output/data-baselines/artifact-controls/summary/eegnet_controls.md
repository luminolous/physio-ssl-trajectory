# Artifact controls (eegnet)

Per-trial Pearson r (D7), mean +- SD across the 10 LOSO test subjects (per-subject values averaged over seeds when n_runs > 1; r_seed_sd = SD of the aggregate across seeds). delta_r_vs_all: mean over subjects of (control r - all-channel r) under the same reference.

| control   | reference   |   n_runs |   r_mean |   r_sd |   r_seed_sd |   r2_mean |   nrmse_mean |   delta_r_vs_all |   r_ratio_vs_all | run                                                                                                                                                                                      |
|:----------|:------------|---------:|---------:|-------:|------------:|----------:|-------------:|-----------------:|-----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| all (car) | car         |        3 |    0.449 |  0.075 |       0.008 |     0.021 |        0.323 |            0.000 |            1.000 | output/data-baselines/baselines/eegnet/20261030-042902_seed1, output/data-baselines/baselines/eegnet/20261030-043700_seed2, output/data-baselines/baselines/eegnet/20261030-044818_seed3 |
