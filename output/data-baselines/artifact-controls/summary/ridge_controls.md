# Artifact controls (ridge)

Per-trial Pearson r (D7), mean +- SD across the 10 LOSO test subjects. delta_r_vs_all: mean over subjects of (control r - all-channel r) under the same reference.

| control             | reference         |   r_mean |   r_sd |   r2_mean |   nrmse_mean |   delta_r_vs_all |   r_ratio_vs_all |   n_subjects_above_all | run                                                                             |
|:--------------------|:------------------|---------:|-------:|----------:|-------------:|-----------------:|-----------------:|-----------------------:|:--------------------------------------------------------------------------------|
| all (car)           | car               |    0.331 |  0.068 |    -0.002 |        0.330 |            0.000 |            1.000 |                nan     | output/data-baselines/baselines/ridge_lag/20261030-034531                       |
| sensorimotor (car)  | car               |    0.299 |  0.063 |    -0.032 |        0.336 |           -0.032 |            0.903 |                  3.000 | output/data-baselines/artifact-controls/ridge_sensorimotor_car/20261030-062227  |
| frontal (car)       | car               |    0.184 |  0.091 |    -0.082 |        0.345 |           -0.147 |            0.557 |                  1.000 | output/data-baselines/artifact-controls/ridge_frontal_car/20261030-061705       |
| temporal (car)      | car               |    0.143 |  0.074 |    -0.091 |        0.346 |           -0.189 |            0.430 |                  0.000 | output/data-baselines/artifact-controls/ridge_temporal_car/20261030-064652      |
| emg                 | n/a (vs all car)  |    0.870 |  0.026 |     0.553 |        0.203 |            0.539 |            2.626 |                 10.000 | output/data-baselines/artifact-controls/ridge_emg/20261030-061433               |
| all (orig)          | orig              |    0.324 |  0.078 |    -0.005 |        0.331 |            0.000 |            1.000 |                nan     | output/data-baselines/artifact-controls/ridge_all_orig/20261030-060504          |
| sensorimotor (orig) | orig              |    0.284 |  0.056 |    -0.027 |        0.335 |           -0.041 |            0.875 |                  2.000 | output/data-baselines/artifact-controls/ridge_sensorimotor_orig/20261030-062534 |
| frontal (orig)      | orig              |    0.229 |  0.071 |    -0.067 |        0.342 |           -0.095 |            0.707 |                  0.000 | output/data-baselines/artifact-controls/ridge_frontal_orig/20261030-061945      |
| temporal (orig)     | orig              |    0.225 |  0.057 |    -0.063 |        0.341 |           -0.099 |            0.695 |                  2.000 | output/data-baselines/artifact-controls/ridge_temporal_orig/20261030-065247     |
| emg                 | n/a (vs all orig) |    0.870 |  0.026 |     0.553 |        0.203 |            0.546 |            2.685 |                 10.000 | output/data-baselines/artifact-controls/ridge_emg/20261030-061433               |
