# Calibration within subject

Sources: uncleaned `output/diagnostics/final-checks/calibration/uncleaned/20261108-132049_seed1`, cleaned `output/diagnostics/final-checks/calibration/cleaned/20261108-134754_seed1`, prefrontal `output/diagnostics/final-checks/calibration/prefrontal/20261108-141449_seed1`

## Means per k (over subjects; draws averaged first)

| data       | condition      | k   |     r |    r2 |   nrmse |   residual_r_specific |
|:-----------|:---------------|:----|------:|------:|--------:|----------------------:|
| cleaned    | finetune       | 10  | 0.862 | 0.622 |   0.195 |                 0.07  |
| cleaned    | finetune       | 30  | 0.868 | 0.641 |   0.19  |                 0.088 |
| cleaned    | finetune       | all | 0.876 | 0.651 |   0.186 |                 0.158 |
| cleaned    | no_calibration | 10  | 0.833 | 0.499 |   0.227 |                 0.009 |
| cleaned    | no_calibration | 30  | 0.833 | 0.499 |   0.227 |                 0.014 |
| cleaned    | no_calibration | all | 0.833 | 0.499 |   0.227 |                 0.007 |
| cleaned    | scratch        | 10  | 0.862 | 0.633 |   0.192 |                 0.055 |
| cleaned    | scratch        | 30  | 0.868 | 0.643 |   0.19  |                 0.081 |
| cleaned    | scratch        | all | 0.873 | 0.649 |   0.187 |                 0.165 |
| cleaned    | template       | 10  | 0.859 | 0.63  |   0.193 |                 0     |
| cleaned    | template       | 30  | 0.863 | 0.645 |   0.189 |                 0     |
| cleaned    | template       | all | 0.864 | 0.648 |   0.189 |                 0     |
| prefrontal | finetune       | 30  | 0.878 | 0.663 |   0.183 |                 0.173 |
| prefrontal | finetune       | all | 0.885 | 0.681 |   0.177 |                 0.207 |
| prefrontal | no_calibration | 30  | 0.85  | 0.536 |   0.218 |                 0.028 |
| prefrontal | no_calibration | all | 0.85  | 0.536 |   0.218 |                 0.021 |
| uncleaned  | finetune       | 10  | 0.87  | 0.628 |   0.191 |                 0.164 |
| uncleaned  | finetune       | 30  | 0.88  | 0.66  |   0.183 |                 0.196 |
| uncleaned  | finetune       | all | 0.887 | 0.667 |   0.179 |                 0.258 |
| uncleaned  | no_calibration | 10  | 0.843 | 0.515 |   0.222 |                 0.062 |
| uncleaned  | no_calibration | 30  | 0.843 | 0.515 |   0.222 |                 0.067 |
| uncleaned  | no_calibration | all | 0.843 | 0.515 |   0.222 |                 0.062 |
| uncleaned  | scratch        | 10  | 0.865 | 0.637 |   0.191 |                 0.08  |
| uncleaned  | scratch        | 30  | 0.876 | 0.66  |   0.184 |                 0.139 |
| uncleaned  | scratch        | all | 0.885 | 0.67  |   0.179 |                 0.248 |
| uncleaned  | template       | 10  | 0.859 | 0.63  |   0.193 |                 0     |
| uncleaned  | template       | 30  | 0.863 | 0.645 |   0.189 |                 0     |
| uncleaned  | template       | all | 0.864 | 0.648 |   0.189 |                 0     |

## Each condition vs the subject cue template

| k   | a                         | b                  | metric              |   a_mean |   b_mean |   diff |   n_a_better |   n |   wilcoxon_p |
|:----|:--------------------------|:-------------------|:--------------------|---------:|---------:|-------:|-------------:|----:|-------------:|
| 10  | uncleaned/no_calibration  | uncleaned/template | r                   |    0.843 |    0.859 | -0.016 |            5 |  10 |        0.557 |
| 10  | uncleaned/no_calibration  | uncleaned/template | r2                  |    0.515 |    0.63  | -0.115 |            3 |  10 |        0.037 |
| 10  | uncleaned/no_calibration  | uncleaned/template | nrmse               |    0.222 |    0.193 |  0.03  |            3 |  10 |        0.037 |
| 10  | uncleaned/no_calibration  | uncleaned/template | residual_r_specific |    0.062 |    0     |  0.062 |            8 |  10 |        0.01  |
| 10  | uncleaned/finetune        | uncleaned/template | r                   |    0.87  |    0.859 |  0.011 |            7 |  10 |        0.064 |
| 10  | uncleaned/finetune        | uncleaned/template | r2                  |    0.628 |    0.63  | -0.002 |            6 |  10 |        0.846 |
| 10  | uncleaned/finetune        | uncleaned/template | nrmse               |    0.191 |    0.193 | -0.001 |            6 |  10 |        0.846 |
| 10  | uncleaned/finetune        | uncleaned/template | residual_r_specific |    0.164 |    0     |  0.164 |           10 |  10 |        0.002 |
| 10  | uncleaned/scratch         | uncleaned/template | r                   |    0.865 |    0.859 |  0.006 |            7 |  10 |        0.027 |
| 10  | uncleaned/scratch         | uncleaned/template | r2                  |    0.637 |    0.63  |  0.007 |            7 |  10 |        0.16  |
| 10  | uncleaned/scratch         | uncleaned/template | nrmse               |    0.191 |    0.193 | -0.002 |            7 |  10 |        0.131 |
| 10  | uncleaned/scratch         | uncleaned/template | residual_r_specific |    0.08  |    0     |  0.08  |           10 |  10 |        0.002 |
| 10  | cleaned/no_calibration    | uncleaned/template | r                   |    0.833 |    0.859 | -0.026 |            5 |  10 |        0.557 |
| 10  | cleaned/no_calibration    | uncleaned/template | r2                  |    0.499 |    0.63  | -0.131 |            1 |  10 |        0.006 |
| 10  | cleaned/no_calibration    | uncleaned/template | nrmse               |    0.227 |    0.193 |  0.035 |            2 |  10 |        0.014 |
| 10  | cleaned/no_calibration    | uncleaned/template | residual_r_specific |    0.009 |    0     |  0.009 |            6 |  10 |        0.695 |
| 10  | cleaned/finetune          | uncleaned/template | r                   |    0.862 |    0.859 |  0.003 |            5 |  10 |        0.432 |
| 10  | cleaned/finetune          | uncleaned/template | r2                  |    0.622 |    0.63  | -0.009 |            4 |  10 |        0.275 |
| 10  | cleaned/finetune          | uncleaned/template | nrmse               |    0.195 |    0.193 |  0.002 |            4 |  10 |        0.375 |
| 10  | cleaned/finetune          | uncleaned/template | residual_r_specific |    0.07  |    0     |  0.07  |           10 |  10 |        0.002 |
| 10  | cleaned/scratch           | uncleaned/template | r                   |    0.862 |    0.859 |  0.003 |            6 |  10 |        0.193 |
| 10  | cleaned/scratch           | uncleaned/template | r2                  |    0.633 |    0.63  |  0.003 |            4 |  10 |        0.695 |
| 10  | cleaned/scratch           | uncleaned/template | nrmse               |    0.192 |    0.193 | -0.001 |            4 |  10 |        0.77  |
| 10  | cleaned/scratch           | uncleaned/template | residual_r_specific |    0.055 |    0     |  0.055 |            9 |  10 |        0.004 |
| 30  | uncleaned/no_calibration  | uncleaned/template | r                   |    0.843 |    0.863 | -0.02  |            5 |  10 |        0.492 |
| 30  | uncleaned/no_calibration  | uncleaned/template | r2                  |    0.515 |    0.645 | -0.13  |            1 |  10 |        0.006 |
| 30  | uncleaned/no_calibration  | uncleaned/template | nrmse               |    0.222 |    0.189 |  0.033 |            2 |  10 |        0.014 |
| 30  | uncleaned/no_calibration  | uncleaned/template | residual_r_specific |    0.067 |    0     |  0.067 |            9 |  10 |        0.014 |
| 30  | uncleaned/finetune        | uncleaned/template | r                   |    0.88  |    0.863 |  0.017 |            8 |  10 |        0.01  |
| 30  | uncleaned/finetune        | uncleaned/template | r2                  |    0.66  |    0.645 |  0.015 |            7 |  10 |        0.275 |
| 30  | uncleaned/finetune        | uncleaned/template | nrmse               |    0.183 |    0.189 | -0.006 |            7 |  10 |        0.16  |
| 30  | uncleaned/finetune        | uncleaned/template | residual_r_specific |    0.196 |    0     |  0.196 |           10 |  10 |        0.002 |
| 30  | uncleaned/scratch         | uncleaned/template | r                   |    0.876 |    0.863 |  0.013 |            8 |  10 |        0.01  |
| 30  | uncleaned/scratch         | uncleaned/template | r2                  |    0.66  |    0.645 |  0.014 |            8 |  10 |        0.027 |
| 30  | uncleaned/scratch         | uncleaned/template | nrmse               |    0.184 |    0.189 | -0.005 |            8 |  10 |        0.02  |
| 30  | uncleaned/scratch         | uncleaned/template | residual_r_specific |    0.139 |    0     |  0.139 |            9 |  10 |        0.004 |
| 30  | cleaned/no_calibration    | uncleaned/template | r                   |    0.833 |    0.863 | -0.03  |            4 |  10 |        0.432 |
| 30  | cleaned/no_calibration    | uncleaned/template | r2                  |    0.499 |    0.645 | -0.146 |            0 |  10 |        0.002 |
| 30  | cleaned/no_calibration    | uncleaned/template | nrmse               |    0.227 |    0.189 |  0.038 |            1 |  10 |        0.004 |
| 30  | cleaned/no_calibration    | uncleaned/template | residual_r_specific |    0.014 |    0     |  0.014 |            6 |  10 |        0.625 |
| 30  | cleaned/finetune          | uncleaned/template | r                   |    0.868 |    0.863 |  0.004 |            8 |  10 |        0.037 |
| 30  | cleaned/finetune          | uncleaned/template | r2                  |    0.641 |    0.645 | -0.004 |            3 |  10 |        0.322 |
| 30  | cleaned/finetune          | uncleaned/template | nrmse               |    0.19  |    0.189 |  0.001 |            4 |  10 |        0.625 |
| 30  | cleaned/finetune          | uncleaned/template | residual_r_specific |    0.088 |    0     |  0.088 |           10 |  10 |        0.002 |
| 30  | cleaned/scratch           | uncleaned/template | r                   |    0.868 |    0.863 |  0.004 |            6 |  10 |        0.105 |
| 30  | cleaned/scratch           | uncleaned/template | r2                  |    0.643 |    0.645 | -0.002 |            4 |  10 |        0.492 |
| 30  | cleaned/scratch           | uncleaned/template | nrmse               |    0.19  |    0.189 |  0     |            4 |  10 |        0.846 |
| 30  | cleaned/scratch           | uncleaned/template | residual_r_specific |    0.081 |    0     |  0.081 |            9 |  10 |        0.004 |
| 30  | prefrontal/no_calibration | uncleaned/template | r                   |    0.85  |    0.863 | -0.013 |            5 |  10 |        0.77  |
| 30  | prefrontal/no_calibration | uncleaned/template | r2                  |    0.536 |    0.645 | -0.109 |            2 |  10 |        0.049 |
| 30  | prefrontal/no_calibration | uncleaned/template | nrmse               |    0.218 |    0.189 |  0.028 |            2 |  10 |        0.084 |
| 30  | prefrontal/no_calibration | uncleaned/template | residual_r_specific |    0.028 |    0     |  0.028 |            7 |  10 |        0.232 |
| 30  | prefrontal/finetune       | uncleaned/template | r                   |    0.878 |    0.863 |  0.015 |            9 |  10 |        0.006 |
| 30  | prefrontal/finetune       | uncleaned/template | r2                  |    0.663 |    0.645 |  0.018 |            8 |  10 |        0.064 |
| 30  | prefrontal/finetune       | uncleaned/template | nrmse               |    0.183 |    0.189 | -0.006 |            9 |  10 |        0.027 |
| 30  | prefrontal/finetune       | uncleaned/template | residual_r_specific |    0.173 |    0     |  0.173 |           10 |  10 |        0.002 |
| all | uncleaned/no_calibration  | uncleaned/template | r                   |    0.843 |    0.864 | -0.021 |            4 |  10 |        0.432 |
| all | uncleaned/no_calibration  | uncleaned/template | r2                  |    0.515 |    0.648 | -0.133 |            1 |  10 |        0.004 |
| all | uncleaned/no_calibration  | uncleaned/template | nrmse               |    0.222 |    0.189 |  0.034 |            2 |  10 |        0.02  |
| all | uncleaned/no_calibration  | uncleaned/template | residual_r_specific |    0.062 |    0     |  0.062 |            9 |  10 |        0.014 |
| all | uncleaned/finetune        | uncleaned/template | r                   |    0.887 |    0.864 |  0.023 |            9 |  10 |        0.006 |
| all | uncleaned/finetune        | uncleaned/template | r2                  |    0.667 |    0.648 |  0.018 |            7 |  10 |        0.16  |
| all | uncleaned/finetune        | uncleaned/template | nrmse               |    0.179 |    0.189 | -0.009 |            8 |  10 |        0.084 |
| all | uncleaned/finetune        | uncleaned/template | residual_r_specific |    0.258 |    0     |  0.258 |           10 |  10 |        0.002 |
| all | uncleaned/scratch         | uncleaned/template | r                   |    0.885 |    0.864 |  0.021 |            9 |  10 |        0.006 |
| all | uncleaned/scratch         | uncleaned/template | r2                  |    0.67  |    0.648 |  0.022 |            8 |  10 |        0.16  |
| all | uncleaned/scratch         | uncleaned/template | nrmse               |    0.179 |    0.189 | -0.01  |            8 |  10 |        0.049 |
| all | uncleaned/scratch         | uncleaned/template | residual_r_specific |    0.248 |    0     |  0.248 |           10 |  10 |        0.002 |
| all | cleaned/no_calibration    | uncleaned/template | r                   |    0.833 |    0.864 | -0.031 |            4 |  10 |        0.432 |
| all | cleaned/no_calibration    | uncleaned/template | r2                  |    0.499 |    0.648 | -0.149 |            0 |  10 |        0.002 |
| all | cleaned/no_calibration    | uncleaned/template | nrmse               |    0.227 |    0.189 |  0.039 |            0 |  10 |        0.002 |
| all | cleaned/no_calibration    | uncleaned/template | residual_r_specific |    0.007 |    0     |  0.007 |            5 |  10 |        1     |
| all | cleaned/finetune          | uncleaned/template | r                   |    0.876 |    0.864 |  0.012 |            9 |  10 |        0.004 |
| all | cleaned/finetune          | uncleaned/template | r2                  |    0.651 |    0.648 |  0.002 |            5 |  10 |        0.922 |
| all | cleaned/finetune          | uncleaned/template | nrmse               |    0.186 |    0.189 | -0.002 |            6 |  10 |        0.322 |
| all | cleaned/finetune          | uncleaned/template | residual_r_specific |    0.158 |    0     |  0.158 |           10 |  10 |        0.002 |
| all | cleaned/scratch           | uncleaned/template | r                   |    0.873 |    0.864 |  0.009 |            8 |  10 |        0.027 |
| all | cleaned/scratch           | uncleaned/template | r2                  |    0.649 |    0.648 |  0.001 |            4 |  10 |        0.922 |
| all | cleaned/scratch           | uncleaned/template | nrmse               |    0.187 |    0.189 | -0.002 |            6 |  10 |        0.557 |
| all | cleaned/scratch           | uncleaned/template | residual_r_specific |    0.165 |    0     |  0.165 |           10 |  10 |        0.002 |
| all | prefrontal/no_calibration | uncleaned/template | r                   |    0.85  |    0.864 | -0.014 |            5 |  10 |        0.846 |
| all | prefrontal/no_calibration | uncleaned/template | r2                  |    0.536 |    0.648 | -0.113 |            2 |  10 |        0.037 |
| all | prefrontal/no_calibration | uncleaned/template | nrmse               |    0.218 |    0.189 |  0.029 |            2 |  10 |        0.064 |
| all | prefrontal/no_calibration | uncleaned/template | residual_r_specific |    0.021 |    0     |  0.021 |            6 |  10 |        0.432 |
| all | prefrontal/finetune       | uncleaned/template | r                   |    0.885 |    0.864 |  0.021 |           10 |  10 |        0.002 |
| all | prefrontal/finetune       | uncleaned/template | r2                  |    0.681 |    0.648 |  0.032 |            9 |  10 |        0.006 |
| all | prefrontal/finetune       | uncleaned/template | nrmse               |    0.177 |    0.189 | -0.011 |            9 |  10 |        0.004 |
| all | prefrontal/finetune       | uncleaned/template | residual_r_specific |    0.207 |    0     |  0.207 |           10 |  10 |        0.002 |

## Other comparisons

| k   | a                      | b                        | metric              |   a_mean |   b_mean |   diff |   n_a_better |   n |   wilcoxon_p |
|:----|:-----------------------|:-------------------------|:--------------------|---------:|---------:|-------:|-------------:|----:|-------------:|
| 10  | uncleaned/finetune     | uncleaned/scratch        | r                   |    0.87  |    0.865 |  0.005 |            6 |  10 |        0.557 |
| 10  | uncleaned/finetune     | uncleaned/scratch        | r2                  |    0.628 |    0.637 | -0.008 |            4 |  10 |        0.432 |
| 10  | uncleaned/finetune     | uncleaned/scratch        | nrmse               |    0.191 |    0.191 |  0.001 |            5 |  10 |        0.77  |
| 10  | uncleaned/finetune     | uncleaned/scratch        | residual_r_specific |    0.164 |    0.08  |  0.084 |           10 |  10 |        0.002 |
| 10  | cleaned/finetune       | cleaned/scratch          | r                   |    0.862 |    0.862 |  0.001 |            5 |  10 |        0.922 |
| 10  | cleaned/finetune       | cleaned/scratch          | r2                  |    0.622 |    0.633 | -0.012 |            3 |  10 |        0.105 |
| 10  | cleaned/finetune       | cleaned/scratch          | nrmse               |    0.195 |    0.192 |  0.003 |            4 |  10 |        0.131 |
| 10  | cleaned/finetune       | cleaned/scratch          | residual_r_specific |    0.07  |    0.055 |  0.015 |            7 |  10 |        0.16  |
| 10  | cleaned/finetune       | uncleaned/finetune       | r                   |    0.862 |    0.87  | -0.008 |            4 |  10 |        0.084 |
| 10  | cleaned/finetune       | uncleaned/finetune       | r2                  |    0.622 |    0.628 | -0.007 |            3 |  10 |        0.375 |
| 10  | cleaned/finetune       | uncleaned/finetune       | nrmse               |    0.195 |    0.191 |  0.003 |            4 |  10 |        0.375 |
| 10  | cleaned/finetune       | uncleaned/finetune       | residual_r_specific |    0.07  |    0.164 | -0.094 |            0 |  10 |        0.002 |
| 10  | cleaned/scratch        | uncleaned/scratch        | r                   |    0.862 |    0.865 | -0.003 |            2 |  10 |        0.014 |
| 10  | cleaned/scratch        | uncleaned/scratch        | r2                  |    0.633 |    0.637 | -0.004 |            3 |  10 |        0.064 |
| 10  | cleaned/scratch        | uncleaned/scratch        | nrmse               |    0.192 |    0.191 |  0.001 |            3 |  10 |        0.049 |
| 10  | cleaned/scratch        | uncleaned/scratch        | residual_r_specific |    0.055 |    0.08  | -0.025 |            1 |  10 |        0.02  |
| 10  | cleaned/no_calibration | uncleaned/no_calibration | r                   |    0.833 |    0.843 | -0.01  |            4 |  10 |        0.193 |
| 10  | cleaned/no_calibration | uncleaned/no_calibration | r2                  |    0.499 |    0.515 | -0.016 |            3 |  10 |        0.375 |
| 10  | cleaned/no_calibration | uncleaned/no_calibration | nrmse               |    0.227 |    0.222 |  0.005 |            3 |  10 |        0.193 |
| 10  | cleaned/no_calibration | uncleaned/no_calibration | residual_r_specific |    0.009 |    0.062 | -0.053 |            1 |  10 |        0.01  |
| 30  | uncleaned/finetune     | uncleaned/scratch        | r                   |    0.88  |    0.876 |  0.004 |            4 |  10 |        0.557 |
| 30  | uncleaned/finetune     | uncleaned/scratch        | r2                  |    0.66  |    0.66  |  0     |            4 |  10 |        1     |
| 30  | uncleaned/finetune     | uncleaned/scratch        | nrmse               |    0.183 |    0.184 | -0.001 |            5 |  10 |        0.77  |
| 30  | uncleaned/finetune     | uncleaned/scratch        | residual_r_specific |    0.196 |    0.139 |  0.057 |            9 |  10 |        0.01  |
| 30  | cleaned/finetune       | cleaned/scratch          | r                   |    0.868 |    0.868 |  0     |            4 |  10 |        0.922 |
| 30  | cleaned/finetune       | cleaned/scratch          | r2                  |    0.641 |    0.643 | -0.002 |            4 |  10 |        0.77  |
| 30  | cleaned/finetune       | cleaned/scratch          | nrmse               |    0.19  |    0.19  |  0.001 |            3 |  10 |        0.695 |
| 30  | cleaned/finetune       | cleaned/scratch          | residual_r_specific |    0.088 |    0.081 |  0.007 |            7 |  10 |        0.557 |
| 30  | cleaned/finetune       | uncleaned/finetune       | r                   |    0.868 |    0.88  | -0.012 |            1 |  10 |        0.01  |
| 30  | cleaned/finetune       | uncleaned/finetune       | r2                  |    0.641 |    0.66  | -0.019 |            2 |  10 |        0.027 |
| 30  | cleaned/finetune       | uncleaned/finetune       | nrmse               |    0.19  |    0.183 |  0.007 |            2 |  10 |        0.02  |
| 30  | cleaned/finetune       | uncleaned/finetune       | residual_r_specific |    0.088 |    0.196 | -0.108 |            0 |  10 |        0.002 |
| 30  | cleaned/scratch        | uncleaned/scratch        | r                   |    0.868 |    0.876 | -0.009 |            1 |  10 |        0.006 |
| 30  | cleaned/scratch        | uncleaned/scratch        | r2                  |    0.643 |    0.66  | -0.017 |            1 |  10 |        0.006 |
| 30  | cleaned/scratch        | uncleaned/scratch        | nrmse               |    0.19  |    0.184 |  0.005 |            1 |  10 |        0.004 |
| 30  | cleaned/scratch        | uncleaned/scratch        | residual_r_specific |    0.081 |    0.139 | -0.058 |            0 |  10 |        0.002 |
| 30  | cleaned/no_calibration | uncleaned/no_calibration | r                   |    0.833 |    0.843 | -0.01  |            4 |  10 |        0.193 |
| 30  | cleaned/no_calibration | uncleaned/no_calibration | r2                  |    0.499 |    0.515 | -0.016 |            3 |  10 |        0.375 |
| 30  | cleaned/no_calibration | uncleaned/no_calibration | nrmse               |    0.227 |    0.222 |  0.005 |            3 |  10 |        0.193 |
| 30  | cleaned/no_calibration | uncleaned/no_calibration | residual_r_specific |    0.014 |    0.067 | -0.054 |            1 |  10 |        0.01  |
| 30  | prefrontal/finetune    | uncleaned/finetune       | r                   |    0.878 |    0.88  | -0.002 |            4 |  10 |        0.375 |
| 30  | prefrontal/finetune    | uncleaned/finetune       | r2                  |    0.663 |    0.66  |  0.003 |            7 |  10 |        0.432 |
| 30  | prefrontal/finetune    | uncleaned/finetune       | nrmse               |    0.183 |    0.183 | -0     |            6 |  10 |        0.77  |
| 30  | prefrontal/finetune    | uncleaned/finetune       | residual_r_specific |    0.173 |    0.196 | -0.024 |            5 |  10 |        0.432 |
| all | uncleaned/finetune     | uncleaned/scratch        | r                   |    0.887 |    0.885 |  0.002 |            7 |  10 |        0.105 |
| all | uncleaned/finetune     | uncleaned/scratch        | r2                  |    0.667 |    0.67  | -0.004 |            6 |  10 |        1     |
| all | uncleaned/finetune     | uncleaned/scratch        | nrmse               |    0.179 |    0.179 |  0.001 |            6 |  10 |        1     |
| all | uncleaned/finetune     | uncleaned/scratch        | residual_r_specific |    0.258 |    0.248 |  0.01  |            6 |  10 |        0.375 |
| all | cleaned/finetune       | cleaned/scratch          | r                   |    0.876 |    0.873 |  0.003 |            5 |  10 |        0.232 |
| all | cleaned/finetune       | cleaned/scratch          | r2                  |    0.651 |    0.649 |  0.002 |            6 |  10 |        0.695 |
| all | cleaned/finetune       | cleaned/scratch          | nrmse               |    0.186 |    0.187 | -0.001 |            4 |  10 |        0.922 |
| all | cleaned/finetune       | cleaned/scratch          | residual_r_specific |    0.158 |    0.165 | -0.007 |            5 |  10 |        0.922 |
| all | cleaned/finetune       | uncleaned/finetune       | r                   |    0.876 |    0.887 | -0.011 |            2 |  10 |        0.084 |
| all | cleaned/finetune       | uncleaned/finetune       | r2                  |    0.651 |    0.667 | -0.016 |            2 |  10 |        0.105 |
| all | cleaned/finetune       | uncleaned/finetune       | nrmse               |    0.186 |    0.179 |  0.007 |            2 |  10 |        0.084 |
| all | cleaned/finetune       | uncleaned/finetune       | residual_r_specific |    0.158 |    0.258 | -0.1   |            0 |  10 |        0.002 |
| all | cleaned/scratch        | uncleaned/scratch        | r                   |    0.873 |    0.885 | -0.012 |            2 |  10 |        0.084 |
| all | cleaned/scratch        | uncleaned/scratch        | r2                  |    0.649 |    0.67  | -0.021 |            2 |  10 |        0.16  |
| all | cleaned/scratch        | uncleaned/scratch        | nrmse               |    0.187 |    0.179 |  0.008 |            2 |  10 |        0.131 |
| all | cleaned/scratch        | uncleaned/scratch        | residual_r_specific |    0.165 |    0.248 | -0.083 |            1 |  10 |        0.006 |
| all | cleaned/no_calibration | uncleaned/no_calibration | r                   |    0.833 |    0.843 | -0.01  |            4 |  10 |        0.193 |
| all | cleaned/no_calibration | uncleaned/no_calibration | r2                  |    0.499 |    0.515 | -0.016 |            3 |  10 |        0.375 |
| all | cleaned/no_calibration | uncleaned/no_calibration | nrmse               |    0.227 |    0.222 |  0.005 |            3 |  10 |        0.193 |
| all | cleaned/no_calibration | uncleaned/no_calibration | residual_r_specific |    0.007 |    0.062 | -0.055 |            1 |  10 |        0.006 |
| all | prefrontal/finetune    | uncleaned/finetune       | r                   |    0.885 |    0.887 | -0.002 |            1 |  10 |        0.084 |
| all | prefrontal/finetune    | uncleaned/finetune       | r2                  |    0.681 |    0.667 |  0.014 |            5 |  10 |        1     |
| all | prefrontal/finetune    | uncleaned/finetune       | nrmse               |    0.177 |    0.179 | -0.002 |            5 |  10 |        0.695 |
| all | prefrontal/finetune    | uncleaned/finetune       | residual_r_specific |    0.207 |    0.258 | -0.051 |            3 |  10 |        0.049 |

