# Channel importance

Drop = score with nothing occluded minus score with the input occluded (set to its normalizer mean), per LOSO test subject, mean over subjects. rank_share = fraction of the region's channels among the 8 largest drops. CAR spreads each channel into all others, so single-channel drops understate regions; see the group occlusions.

## Groups

| config             | occluded      |   n_subjects |   drop_r |   n_positive_r |   wilcoxon_p_r |   drop_residual_r_specific |   n_positive_residual_r_specific |   wilcoxon_p_residual_r_specific |
|:-------------------|:--------------|-------------:|---------:|---------------:|---------------:|---------------------------:|---------------------------------:|---------------------------------:|
| cnn_lstm_broadband | contralateral |           10 |    0.13  |             10 |          0.002 |                      0.05  |                               10 |                            0.002 |
| cnn_lstm_broadband | ipsilateral   |           10 |    0.075 |              9 |          0.004 |                      0.028 |                                9 |                            0.006 |
| cnn_lstm_broadband | occipital     |           10 |    0.088 |              8 |          0.01  |                      0.065 |                               10 |                            0.002 |
| cnn_lstm_broadband | prefrontal    |           10 |    0.061 |              9 |          0.006 |                      0.044 |                               10 |                            0.002 |
| cnn_lstm_mrcp      | contralateral |           10 |    0.109 |             10 |          0.002 |                      0.045 |                               10 |                            0.002 |
| cnn_lstm_mrcp      | ipsilateral   |           10 |    0.031 |              7 |          0.027 |                      0.013 |                                9 |                            0.004 |
| cnn_lstm_mrcp      | occipital     |           10 |    0.09  |              9 |          0.006 |                      0.047 |                               10 |                            0.002 |
| cnn_lstm_mrcp      | prefrontal    |           10 |    0.066 |             10 |          0.002 |                      0.022 |                               10 |                            0.002 |
| eegnet_broadband   | contralateral |           10 |    0.121 |             10 |          0.002 |                      0.029 |                                9 |                            0.004 |
| eegnet_broadband   | ipsilateral   |           10 |    0.038 |              9 |          0.027 |                      0.005 |                                7 |                            0.432 |
| eegnet_broadband   | occipital     |           10 |    0.064 |             10 |          0.002 |                      0.041 |                               10 |                            0.002 |
| eegnet_broadband   | prefrontal    |           10 |    0.03  |              9 |          0.004 |                      0.01  |                                5 |                            0.322 |
| ridge_lag          | contralateral |           10 |    0.182 |             10 |          0.002 |                      0.026 |                               10 |                            0.002 |
| ridge_lag          | ipsilateral   |           10 |    0.024 |              7 |          0.064 |                      0.007 |                                9 |                            0.049 |
| ridge_lag          | occipital     |           10 |    0.033 |              9 |          0.014 |                      0.006 |                                9 |                            0.064 |
| ridge_lag          | prefrontal    |           10 |    0.01  |              9 |          0.027 |                      0.003 |                                5 |                            0.846 |

## Regions (single-channel drops)

| config             | region        |   n_channels |   mean_channel_drop_r |   rank_share_r |   mean_channel_drop_residual_r_specific |   rank_share_residual_r_specific |
|:-------------------|:--------------|-------------:|----------------------:|---------------:|----------------------------------------:|---------------------------------:|
| cnn_lstm_broadband | prefrontal    |            2 |                 0.02  |          0.5   |                                   0.014 |                            1     |
| cnn_lstm_broadband | contralateral |            6 |                 0.019 |          0.333 |                                   0.008 |                            0.333 |
| cnn_lstm_broadband | ipsilateral   |            6 |                 0.014 |          0.167 |                                   0.004 |                            0.167 |
| cnn_lstm_broadband | occipital     |            5 |                 0.009 |          0.2   |                                   0.008 |                            0.4   |
| cnn_lstm_broadband | temporal      |            4 |                 0.007 |          0.25  |                                   0.002 |                            0     |
| cnn_lstm_mrcp      | prefrontal    |            2 |                 0.02  |          1     |                                   0.007 |                            0.5   |
| cnn_lstm_mrcp      | contralateral |            6 |                 0.012 |          0.333 |                                   0.006 |                            0.333 |
| cnn_lstm_mrcp      | ipsilateral   |            6 |                 0.006 |          0.167 |                                   0.003 |                            0.167 |
| cnn_lstm_mrcp      | occipital     |            5 |                 0.008 |          0.4   |                                   0.006 |                            0.6   |
| cnn_lstm_mrcp      | temporal      |            4 |                 0.003 |          0.25  |                                   0     |                            0     |
| eegnet_broadband   | prefrontal    |            2 |                 0.012 |          0.5   |                                   0.004 |                            0.5   |
| eegnet_broadband   | contralateral |            6 |                 0.014 |          0.667 |                                   0.004 |                            0.333 |
| eegnet_broadband   | ipsilateral   |            6 |                 0.007 |          0.167 |                                   0     |                            0.167 |
| eegnet_broadband   | occipital     |            5 |                 0.006 |          0     |                                   0.007 |                            0.8   |
| eegnet_broadband   | temporal      |            4 |                 0.001 |          0     |                                   0.001 |                            0     |
| ridge_lag          | prefrontal    |            2 |                 0.004 |          0     |                                   0.001 |                            0.5   |
| ridge_lag          | contralateral |            6 |                 0.012 |          0.5   |                                   0.004 |                            0.667 |
| ridge_lag          | ipsilateral   |            6 |                 0.005 |          0.167 |                                   0.001 |                            0.167 |
| ridge_lag          | occipital     |            5 |                 0.003 |          0.4   |                                   0.001 |                            0.4   |
| ridge_lag          | temporal      |            4 |                 0     |          0     |                                   0     |                            0     |

## Top channels

- `cnn_lstm_broadband`: top 8 channels by drop in residual_r_specific: Cz (0.023), F7 (0.020), Fp2 (0.017), O1 (0.012), Oz (0.011), Fp1 (0.011), F3 (0.010), FC1 (0.010)
- `cnn_lstm_mrcp`: top 8 channels by drop in residual_r_specific: Cz (0.019), Fp2 (0.011), F7 (0.011), O1 (0.010), FC1 (0.009), Oz (0.009), O2 (0.006), P7 (0.005)
- `eegnet_broadband`: top 8 channels by drop in residual_r_specific: F7 (0.013), Cz (0.012), O1 (0.009), Oz (0.008), PO9 (0.007), O2 (0.007), FC1 (0.007), Fp2 (0.006)
- `ridge_lag`: top 8 channels by drop in residual_r_specific: Cz (0.013), FC1 (0.008), O1 (0.003), Oz (0.003), F7 (0.003), C3 (0.003), CP1 (0.002), Fp1 (0.002)
- `ridge_lag` Haufe pattern norm (mean over lags), top 8: Fz (18.128), T7 (13.682), PO9 (12.541), F3 (11.483), F4 (10.417), TP9 (10.119), Oz (9.115), P7 (8.296)
