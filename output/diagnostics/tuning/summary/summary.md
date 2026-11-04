# Deep baseline tuning

## Tuned vs untuned (delta = tuned - untuned; n_up = subjects with a higher value, and lower is better for NRMSE)

| model          |   n_seeds_tuned |   r_untuned |   r_tuned |   r_delta |   r_delta_sd |   r_n_up |   r2_untuned |   r2_tuned |   r2_delta |   r2_delta_sd |   r2_n_up |   nrmse_untuned |   nrmse_tuned |   nrmse_delta |   nrmse_delta_sd |   nrmse_n_up |   n_subjects |
|:---------------|----------------:|------------:|----------:|----------:|-------------:|---------:|-------------:|-----------:|-----------:|--------------:|----------:|----------------:|--------------:|--------------:|-----------------:|-------------:|-------------:|
| eegnet         |               3 |       0.449 |     0.456 |     0.007 |        0.022 |        6 |        0.021 |      0.017 |     -0.004 |         0.026 |         5 |           0.323 |         0.323 |         0.000 |            0.004 |            3 |           10 |
| shallowconvnet |               3 |       0.454 |     0.451 |    -0.003 |        0.031 |        6 |        0.002 |     -0.002 |     -0.004 |         0.026 |         5 |           0.327 |         0.327 |         0.001 |            0.004 |            3 |           10 |
| cnn_lstm       |               3 |       0.499 |     0.478 |    -0.021 |        0.026 |        2 |        0.039 |      0.030 |     -0.009 |         0.043 |         3 |           0.317 |         0.319 |         0.002 |            0.007 |            7 |           10 |

## Folds selecting each lr (capped = candidates that ran all 30 epochs)

| model          | lr        |   n_folds |   mean_val_mse |   capped |
|:---------------|:----------|----------:|---------------:|---------:|
| cnn_lstm       | lr 0.0001 |         3 |          0.827 |        3 |
| cnn_lstm       | lr 0.0003 |         1 |          0.820 |        0 |
| cnn_lstm       | lr 0.001  |         6 |          0.817 |        0 |
| eegnet         | lr 0.0001 |         0 |          0.877 |        2 |
| eegnet         | lr 0.0003 |         0 |          0.867 |        0 |
| eegnet         | lr 0.001  |        10 |          0.847 |        0 |
| shallowconvnet | lr 0.0001 |         2 |          0.928 |        0 |
| shallowconvnet | lr 0.0003 |         3 |          0.925 |        0 |
| shallowconvnet | lr 0.001  |         5 |          0.923 |        0 |

## Selection gain vs initial-weight noise in val MSE

| model          |   init_noise |   selection_gain |
|:---------------|-------------:|-----------------:|
| eegnet         |        0.011 |            0.000 |
| shallowconvnet |        0.007 |            0.007 |
| cnn_lstm       |        0.007 |            0.007 |

## Per subject

| model          | subject   | selected_lr   |   r_untuned |   r_tuned |   r_delta |
|:---------------|:----------|:--------------|------------:|----------:|----------:|
| eegnet         | P1        | lr 0.001      |       0.501 |     0.501 |     0.000 |
| eegnet         | P2        | lr 0.001      |       0.343 |     0.386 |     0.043 |
| eegnet         | P4        | lr 0.001      |       0.449 |     0.466 |     0.016 |
| eegnet         | P5        | lr 0.001      |       0.433 |     0.405 |    -0.028 |
| eegnet         | P7        | lr 0.001      |       0.625 |     0.630 |     0.006 |
| eegnet         | P8        | lr 0.001      |       0.409 |     0.436 |     0.026 |
| eegnet         | P9        | lr 0.001      |       0.446 |     0.441 |    -0.005 |
| eegnet         | P10       | lr 0.001      |       0.441 |     0.466 |     0.025 |
| eegnet         | P11       | lr 0.001      |       0.389 |     0.394 |     0.006 |
| eegnet         | P12       | lr 0.001      |       0.454 |     0.435 |    -0.019 |
| shallowconvnet | P1        | lr 0.001      |       0.462 |     0.462 |     0.000 |
| shallowconvnet | P2        | lr 0.001      |       0.359 |     0.368 |     0.008 |
| shallowconvnet | P4        | lr 0.0003     |       0.428 |     0.457 |     0.029 |
| shallowconvnet | P5        | lr 0.0001     |       0.457 |     0.463 |     0.006 |
| shallowconvnet | P7        | lr 0.001      |       0.572 |     0.578 |     0.007 |
| shallowconvnet | P8        | lr 0.0001     |       0.509 |     0.517 |     0.008 |
| shallowconvnet | P9        | lr 0.001      |       0.403 |     0.397 |    -0.006 |
| shallowconvnet | P10       | lr 0.0003     |       0.504 |     0.419 |    -0.085 |
| shallowconvnet | P11       | lr 0.001      |       0.406 |     0.420 |     0.014 |
| shallowconvnet | P12       | lr 0.0003     |       0.436 |     0.429 |    -0.007 |
| cnn_lstm       | P1        | lr 0.0001     |       0.571 |     0.529 |    -0.042 |
| cnn_lstm       | P2        | lr 0.001      |       0.475 |     0.471 |    -0.004 |
| cnn_lstm       | P4        | lr 0.001      |       0.542 |     0.526 |    -0.015 |
| cnn_lstm       | P5        | lr 0.001      |       0.462 |     0.467 |     0.005 |
| cnn_lstm       | P7        | lr 0.001      |       0.676 |     0.668 |    -0.008 |
| cnn_lstm       | P8        | lr 0.0001     |       0.420 |     0.358 |    -0.062 |
| cnn_lstm       | P9        | lr 0.0001     |       0.436 |     0.370 |    -0.066 |
| cnn_lstm       | P10       | lr 0.001      |       0.581 |     0.569 |    -0.012 |
| cnn_lstm       | P11       | lr 0.001      |       0.377 |     0.369 |    -0.008 |
| cnn_lstm       | P12       | lr 0.0003     |       0.451 |     0.454 |     0.003 |

## Best epoch of the final models

| model          | untuned                    | tuned                       |
|:---------------|:---------------------------|:----------------------------|
| eegnet         | 3/30 at epoch 1, median 6  | 3/30 at epoch 1, median 5.5 |
| shallowconvnet | 17/30 at epoch 1, median 1 | 10/30 at epoch 1, median 4  |
| cnn_lstm       | 2/30 at epoch 1, median 4  | 4/30 at epoch 1, median 4   |
