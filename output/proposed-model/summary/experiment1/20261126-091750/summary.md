# Experiment 1 (RQ1): encoder x branches x setting

LOSO, 10 test subjects; subject scores averaged over decoder seeds, then mean / SD over subjects; `r_seed_sd`: SD over seeds of the aggregate. Paired tests: Wilcoxon signed-rank over subjects (two-sided), `n_pos` = subjects with a positive difference.

## Configurations

| setting   | branches   | encoder         |   n_seeds |   r_mean |   r_sd |   r_seed_sd |   r2_mean |   nrmse_mean |   residual_r_matched_mean |   residual_r_matched_sd |   residual_r_specific_mean |   residual_r_specific_sd |   floor_r_mean |   headfloor_r_mean |
|:----------|:-----------|:----------------|----------:|---------:|-------:|------------:|----------:|-------------:|--------------------------:|------------------------:|---------------------------:|-------------------------:|---------------:|-------------------:|
| frozen    | hybrid     | lejepa_guided   |         3 |   0.8074 | 0.0696 |      0.0047 |    0.5141 |       0.2182 |                    0.0942 |                  0.0489 |                     0.1087 |                   0.0427 |         0.8027 |             0.8039 |
| frozen    | hybrid     | lejepa_unguided |         3 |   0.8099 | 0.0676 |      0.0015 |    0.5168 |       0.2178 |                    0.1005 |                  0.0458 |                     0.1153 |                   0.0389 |         0.8027 |             0.8039 |
| frozen    | hybrid     | jepa_guided     |         3 |   0.8067 | 0.0696 |      0.002  |    0.5098 |       0.2191 |                    0.0996 |                  0.0545 |                     0.1099 |                   0.051  |         0.8027 |             0.8039 |
| frozen    | hybrid     | jepa_unguided   |         3 |   0.8155 | 0.0699 |      0.0055 |    0.5248 |       0.2152 |                    0.1342 |                  0.0599 |                     0.1586 |                   0.0589 |         0.8027 |             0.8039 |
| frozen    | hybrid     | mae             |         3 |   0.8088 | 0.0668 |      0.0045 |    0.5124 |       0.2181 |                    0.11   |                  0.0557 |                     0.1301 |                   0.05   |         0.8027 |             0.8039 |
| frozen    | hybrid     | random0         |         3 |   0.8102 | 0.0713 |      0.001  |    0.5214 |       0.2168 |                    0.107  |                  0.0502 |                     0.1294 |                   0.044  |         0.8027 |             0.8039 |
| frozen    | neural     | lejepa_guided   |         3 |   0.8051 | 0.0705 |      0.0011 |    0.5094 |       0.2193 |                    0.078  |                  0.039  |                     0.0843 |                   0.043  |         0.8027 |             0.8039 |
| frozen    | neural     | lejepa_unguided |         3 |   0.8074 | 0.0714 |      0.0041 |    0.5147 |       0.218  |                    0.0788 |                  0.0384 |                     0.0827 |                   0.0402 |         0.8027 |             0.8039 |
| frozen    | neural     | jepa_guided     |         3 |   0.8029 | 0.0742 |      0.0031 |    0.5058 |       0.2205 |                    0.0514 |                  0.0427 |                     0.0508 |                   0.0564 |         0.8027 |             0.8039 |
| frozen    | neural     | jepa_unguided   |         3 |   0.812  | 0.0677 |      0.0054 |    0.518  |       0.2169 |                    0.1195 |                  0.0527 |                     0.1396 |                   0.0528 |         0.8027 |             0.8039 |
| frozen    | neural     | mae             |         3 |   0.8078 | 0.0709 |      0.0068 |    0.5171 |       0.2172 |                    0.1029 |                  0.0502 |                     0.13   |                   0.0482 |         0.8027 |             0.8039 |
| frozen    | neural     | random0         |         3 |   0.808  | 0.0772 |      0.0025 |    0.5192 |       0.2175 |                    0.0752 |                  0.0404 |                     0.0898 |                   0.0504 |         0.8027 |             0.8039 |
| frozen    | ocular     | noenc           |         3 |   0.8045 | 0.0809 |      0.0067 |    0.5183 |       0.2182 |                    0.096  |                  0.0612 |                     0.1047 |                   0.0619 |         0.8027 |             0.8039 |
| finetune  | hybrid     | lejepa_guided   |         3 |   0.815  | 0.0584 |      0.003  |    0.5195 |       0.2163 |                    0.1214 |                  0.0624 |                     0.1401 |                   0.0561 |         0.8027 |             0.8039 |
| finetune  | hybrid     | lejepa_unguided |         3 |   0.8148 | 0.063  |      0.002  |    0.5156 |       0.2167 |                    0.1243 |                  0.0578 |                     0.1427 |                   0.056  |         0.8027 |             0.8039 |
| finetune  | hybrid     | jepa_unguided   |         3 |   0.82   | 0.0659 |      0.0052 |    0.5268 |       0.2137 |                    0.16   |                  0.0761 |                     0.1826 |                   0.0729 |         0.8027 |             0.8039 |
| finetune  | hybrid     | mae             |         3 |   0.819  | 0.0652 |      0.0045 |    0.5244 |       0.2144 |                    0.1472 |                  0.0657 |                     0.1754 |                   0.0643 |         0.8027 |             0.8039 |
| finetune  | hybrid     | random0         |         3 |   0.8171 | 0.0616 |      0.0052 |    0.5247 |       0.2148 |                    0.1308 |                  0.0547 |                     0.1494 |                   0.047  |         0.8027 |             0.8039 |
| finetune  | neural     | lejepa_guided   |         3 |   0.8184 | 0.0613 |      0.0013 |    0.5267 |       0.2143 |                    0.1292 |                  0.0595 |                     0.1459 |                   0.061  |         0.8027 |             0.8039 |
| finetune  | neural     | lejepa_unguided |         3 |   0.8153 | 0.061  |      0.0007 |    0.5171 |       0.2166 |                    0.1275 |                  0.063  |                     0.1452 |                   0.0576 |         0.8027 |             0.8039 |
| finetune  | neural     | jepa_unguided   |         3 |   0.8227 | 0.0632 |      0.0054 |    0.5329 |       0.212  |                    0.1623 |                  0.0759 |                     0.1893 |                   0.0769 |         0.8027 |             0.8039 |
| finetune  | neural     | mae             |         3 |   0.8161 | 0.0597 |      0.0068 |    0.5151 |       0.2167 |                    0.1439 |                  0.0741 |                     0.1642 |                   0.0676 |         0.8027 |             0.8039 |
| finetune  | neural     | random0         |         3 |   0.8203 | 0.0615 |      0.0046 |    0.5313 |       0.2131 |                    0.1377 |                  0.0581 |                     0.1644 |                   0.052  |         0.8027 |             0.8039 |

## Paired comparisons (r, R2, NRMSE)

| comparison                        | a                               | b                             | metric   |   mean_diff |   sd_diff |   n_subjects |   n_pos |   p_wilcoxon |
|:----------------------------------|:--------------------------------|:------------------------------|:---------|------------:|----------:|-------------:|--------:|-------------:|
| hybrid - ocular-only              | hybrid/finetune/jepa_unguided   | ocular/frozen/noenc           | r        |      0.0155 |    0.0235 |           10 |       7 |       0.1309 |
| hybrid - ocular-only              | hybrid/finetune/jepa_unguided   | ocular/frozen/noenc           | r2       |      0.0085 |    0.0453 |           10 |       5 |       0.4922 |
| hybrid - ocular-only              | hybrid/finetune/jepa_unguided   | ocular/frozen/noenc           | nrmse    |     -0.0044 |    0.0101 |           10 |       4 |       0.1934 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/jepa_unguided   | hybrid/finetune/random0       | r        |      0.0029 |    0.0092 |           10 |       6 |       0.5566 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/jepa_unguided   | hybrid/finetune/random0       | r2       |      0.0021 |    0.0247 |           10 |       4 |       0.9219 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/jepa_unguided   | hybrid/finetune/random0       | nrmse    |     -0.001  |    0.0052 |           10 |       6 |       0.8457 |
| fine-tune - frozen                | hybrid/finetune/jepa_unguided   | hybrid/frozen/jepa_unguided   | r        |      0.0044 |    0.0098 |           10 |       6 |       0.1934 |
| fine-tune - frozen                | hybrid/finetune/jepa_unguided   | hybrid/frozen/jepa_unguided   | r2       |      0.002  |    0.0219 |           10 |       6 |       0.9219 |
| fine-tune - frozen                | hybrid/finetune/jepa_unguided   | hybrid/frozen/jepa_unguided   | nrmse    |     -0.0014 |    0.0049 |           10 |       4 |       0.375  |
| hybrid - ocular-only              | hybrid/finetune/lejepa_guided   | ocular/frozen/noenc           | r        |      0.0105 |    0.0284 |           10 |       5 |       0.5566 |
| hybrid - ocular-only              | hybrid/finetune/lejepa_guided   | ocular/frozen/noenc           | r2       |      0.0012 |    0.0477 |           10 |       5 |       0.7695 |
| hybrid - ocular-only              | hybrid/finetune/lejepa_guided   | ocular/frozen/noenc           | nrmse    |     -0.0018 |    0.0096 |           10 |       5 |       1      |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_guided   | hybrid/finetune/random0       | r        |     -0.0021 |    0.0136 |           10 |       4 |       0.625  |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_guided   | hybrid/finetune/random0       | r2       |     -0.0052 |    0.0286 |           10 |       5 |       0.6953 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_guided   | hybrid/finetune/random0       | nrmse    |      0.0016 |    0.0064 |           10 |       6 |       0.375  |
| fine-tune - frozen                | hybrid/finetune/lejepa_guided   | hybrid/frozen/lejepa_guided   | r        |      0.0076 |    0.0157 |           10 |       7 |       0.2324 |
| fine-tune - frozen                | hybrid/finetune/lejepa_guided   | hybrid/frozen/lejepa_guided   | r2       |      0.0053 |    0.0227 |           10 |       6 |       0.5566 |
| fine-tune - frozen                | hybrid/finetune/lejepa_guided   | hybrid/frozen/lejepa_guided   | nrmse    |     -0.0019 |    0.005  |           10 |       4 |       0.2754 |
| hybrid - ocular-only              | hybrid/finetune/lejepa_unguided | ocular/frozen/noenc           | r        |      0.0104 |    0.0234 |           10 |       6 |       0.2754 |
| hybrid - ocular-only              | hybrid/finetune/lejepa_unguided | ocular/frozen/noenc           | r2       |     -0.0027 |    0.039  |           10 |       6 |       0.9219 |
| hybrid - ocular-only              | hybrid/finetune/lejepa_unguided | ocular/frozen/noenc           | nrmse    |     -0.0015 |    0.0086 |           10 |       4 |       0.7695 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_unguided | hybrid/finetune/random0       | r        |     -0.0022 |    0.0077 |           10 |       2 |       0.1309 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_unguided | hybrid/finetune/random0       | r2       |     -0.0091 |    0.0226 |           10 |       3 |       0.1309 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_unguided | hybrid/finetune/random0       | nrmse    |      0.0019 |    0.0049 |           10 |       8 |       0.1309 |
| fine-tune - frozen                | hybrid/finetune/lejepa_unguided | hybrid/frozen/lejepa_unguided | r        |      0.0049 |    0.012  |           10 |       7 |       0.2754 |
| fine-tune - frozen                | hybrid/finetune/lejepa_unguided | hybrid/frozen/lejepa_unguided | r2       |     -0.0012 |    0.0272 |           10 |       6 |       0.8457 |
| fine-tune - frozen                | hybrid/finetune/lejepa_unguided | hybrid/frozen/lejepa_unguided | nrmse    |     -0.0011 |    0.0061 |           10 |       4 |       0.9219 |
| hybrid - ocular-only              | hybrid/finetune/mae             | ocular/frozen/noenc           | r        |      0.0145 |    0.026  |           10 |       5 |       0.2324 |
| hybrid - ocular-only              | hybrid/finetune/mae             | ocular/frozen/noenc           | r2       |      0.0061 |    0.0535 |           10 |       5 |       0.8457 |
| hybrid - ocular-only              | hybrid/finetune/mae             | ocular/frozen/noenc           | nrmse    |     -0.0037 |    0.0119 |           10 |       5 |       0.2324 |
| pretrained - scratch (fine-tune)  | hybrid/finetune/mae             | hybrid/finetune/random0       | r        |      0.0019 |    0.0063 |           10 |       6 |       0.625  |
| pretrained - scratch (fine-tune)  | hybrid/finetune/mae             | hybrid/finetune/random0       | r2       |     -0.0003 |    0.0168 |           10 |       5 |       1      |
| pretrained - scratch (fine-tune)  | hybrid/finetune/mae             | hybrid/finetune/random0       | nrmse    |     -0.0004 |    0.0038 |           10 |       5 |       0.7695 |
| fine-tune - frozen                | hybrid/finetune/mae             | hybrid/frozen/mae             | r        |      0.0102 |    0.0209 |           10 |       6 |       0.2754 |
| fine-tune - frozen                | hybrid/finetune/mae             | hybrid/frozen/mae             | r2       |      0.012  |    0.0572 |           10 |       5 |       1      |
| fine-tune - frozen                | hybrid/finetune/mae             | hybrid/frozen/mae             | nrmse    |     -0.0037 |    0.013  |           10 |       4 |       0.5566 |
| hybrid - ocular-only              | hybrid/finetune/random0         | ocular/frozen/noenc           | r        |      0.0126 |    0.0264 |           10 |       5 |       0.2324 |
| hybrid - ocular-only              | hybrid/finetune/random0         | ocular/frozen/noenc           | r2       |      0.0064 |    0.0519 |           10 |       5 |       1      |
| hybrid - ocular-only              | hybrid/finetune/random0         | ocular/frozen/noenc           | nrmse    |     -0.0034 |    0.0109 |           10 |       5 |       0.3223 |
| fine-tune - frozen                | hybrid/finetune/random0         | hybrid/frozen/random0         | r        |      0.0069 |    0.0151 |           10 |       6 |       0.1602 |
| fine-tune - frozen                | hybrid/finetune/random0         | hybrid/frozen/random0         | r2       |      0.0034 |    0.0342 |           10 |       3 |       0.9219 |
| fine-tune - frozen                | hybrid/finetune/random0         | hybrid/frozen/random0         | nrmse    |     -0.002  |    0.0073 |           10 |       4 |       0.625  |
| hybrid - ocular-only              | hybrid/frozen/jepa_guided       | ocular/frozen/noenc           | r        |      0.0022 |    0.0174 |           10 |       5 |       0.8457 |
| hybrid - ocular-only              | hybrid/frozen/jepa_guided       | ocular/frozen/noenc           | r2       |     -0.0085 |    0.0288 |           10 |       3 |       0.3223 |
| hybrid - ocular-only              | hybrid/frozen/jepa_guided       | ocular/frozen/noenc           | nrmse    |      0.001  |    0.0065 |           10 |       6 |       0.625  |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_guided       | hybrid/frozen/random0         | r        |     -0.0034 |    0.0064 |           10 |       3 |       0.1934 |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_guided       | hybrid/frozen/random0         | r2       |     -0.0116 |    0.0155 |           10 |       2 |       0.0488 |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_guided       | hybrid/frozen/random0         | nrmse    |      0.0024 |    0.0037 |           10 |       8 |       0.084  |
| hybrid - ocular-only              | hybrid/frozen/jepa_unguided     | ocular/frozen/noenc           | r        |      0.0111 |    0.0155 |           10 |       6 |       0.084  |
| hybrid - ocular-only              | hybrid/frozen/jepa_unguided     | ocular/frozen/noenc           | r2       |      0.0065 |    0.0336 |           10 |       5 |       0.625  |
| hybrid - ocular-only              | hybrid/frozen/jepa_unguided     | ocular/frozen/noenc           | nrmse    |     -0.003  |    0.0071 |           10 |       4 |       0.3223 |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_unguided     | hybrid/frozen/random0         | r        |      0.0054 |    0.0121 |           10 |       7 |       0.2324 |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_unguided     | hybrid/frozen/random0         | r2       |      0.0034 |    0.0266 |           10 |       5 |       0.7695 |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_unguided     | hybrid/frozen/random0         | nrmse    |     -0.0016 |    0.0061 |           10 |       5 |       0.625  |
| hybrid - ocular-only              | hybrid/frozen/lejepa_guided     | ocular/frozen/noenc           | r        |      0.003  |    0.0179 |           10 |       5 |       0.8457 |
| hybrid - ocular-only              | hybrid/frozen/lejepa_guided     | ocular/frozen/noenc           | r2       |     -0.0042 |    0.0354 |           10 |       4 |       0.5566 |
| hybrid - ocular-only              | hybrid/frozen/lejepa_guided     | ocular/frozen/noenc           | nrmse    |      0.0001 |    0.0076 |           10 |       5 |       0.8457 |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_guided     | hybrid/frozen/random0         | r        |     -0.0027 |    0.0066 |           10 |       2 |       0.1934 |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_guided     | hybrid/frozen/random0         | r2       |     -0.0072 |    0.017  |           10 |       3 |       0.2754 |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_guided     | hybrid/frozen/random0         | nrmse    |      0.0014 |    0.0037 |           10 |       7 |       0.2754 |
| hybrid - ocular-only              | hybrid/frozen/lejepa_unguided   | ocular/frozen/noenc           | r        |      0.0054 |    0.0207 |           10 |       5 |       0.7695 |
| hybrid - ocular-only              | hybrid/frozen/lejepa_unguided   | ocular/frozen/noenc           | r2       |     -0.0015 |    0.0467 |           10 |       4 |       0.7695 |
| hybrid - ocular-only              | hybrid/frozen/lejepa_unguided   | ocular/frozen/noenc           | nrmse    |     -0.0004 |    0.0099 |           10 |       5 |       1      |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_unguided   | hybrid/frozen/random0         | r        |     -0.0003 |    0.0079 |           10 |       5 |       0.9219 |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_unguided   | hybrid/frozen/random0         | r2       |     -0.0046 |    0.0225 |           10 |       4 |       0.375  |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_unguided   | hybrid/frozen/random0         | nrmse    |      0.001  |    0.0045 |           10 |       7 |       0.3223 |
| hybrid - ocular-only              | hybrid/frozen/mae               | ocular/frozen/noenc           | r        |      0.0043 |    0.0239 |           10 |       5 |       0.625  |
| hybrid - ocular-only              | hybrid/frozen/mae               | ocular/frozen/noenc           | r2       |     -0.0059 |    0.0546 |           10 |       5 |       0.6953 |
| hybrid - ocular-only              | hybrid/frozen/mae               | ocular/frozen/noenc           | nrmse    |     -0      |    0.0115 |           10 |       5 |       1      |
| pretrained - random-init (frozen) | hybrid/frozen/mae               | hybrid/frozen/random0         | r        |     -0.0014 |    0.0155 |           10 |       4 |       0.6953 |
| pretrained - random-init (frozen) | hybrid/frozen/mae               | hybrid/frozen/random0         | r2       |     -0.009  |    0.0354 |           10 |       5 |       0.8457 |
| pretrained - random-init (frozen) | hybrid/frozen/mae               | hybrid/frozen/random0         | nrmse    |      0.0014 |    0.0073 |           10 |       5 |       0.9219 |
| hybrid - ocular-only              | hybrid/frozen/random0           | ocular/frozen/noenc           | r        |      0.0057 |    0.0172 |           10 |       5 |       0.375  |
| hybrid - ocular-only              | hybrid/frozen/random0           | ocular/frozen/noenc           | r2       |      0.0031 |    0.0297 |           10 |       5 |       0.6953 |
| hybrid - ocular-only              | hybrid/frozen/random0           | ocular/frozen/noenc           | nrmse    |     -0.0014 |    0.007  |           10 |       5 |       0.4922 |
| pretrained - scratch (fine-tune)  | neural/finetune/jepa_unguided   | neural/finetune/random0       | r        |      0.0024 |    0.0135 |           10 |       5 |       0.9219 |
| pretrained - scratch (fine-tune)  | neural/finetune/jepa_unguided   | neural/finetune/random0       | r2       |      0.0017 |    0.0339 |           10 |       5 |       1      |
| pretrained - scratch (fine-tune)  | neural/finetune/jepa_unguided   | neural/finetune/random0       | nrmse    |     -0.0011 |    0.0075 |           10 |       5 |       0.7695 |
| fine-tune - frozen                | neural/finetune/jepa_unguided   | neural/frozen/jepa_unguided   | r        |      0.0106 |    0.0198 |           10 |       7 |       0.1602 |
| fine-tune - frozen                | neural/finetune/jepa_unguided   | neural/frozen/jepa_unguided   | r2       |      0.0149 |    0.0438 |           10 |       8 |       0.2324 |
| fine-tune - frozen                | neural/finetune/jepa_unguided   | neural/frozen/jepa_unguided   | nrmse    |     -0.005  |    0.0106 |           10 |       2 |       0.1934 |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_guided   | neural/finetune/random0       | r        |     -0.0019 |    0.0097 |           10 |       2 |       0.1934 |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_guided   | neural/finetune/random0       | r2       |     -0.0045 |    0.0183 |           10 |       3 |       0.2754 |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_guided   | neural/finetune/random0       | nrmse    |      0.0012 |    0.0047 |           10 |       7 |       0.2324 |
| fine-tune - frozen                | neural/finetune/lejepa_guided   | neural/frozen/lejepa_guided   | r        |      0.0134 |    0.017  |           10 |       7 |       0.0273 |
| fine-tune - frozen                | neural/finetune/lejepa_guided   | neural/frozen/lejepa_guided   | r2       |      0.0173 |    0.0401 |           10 |       7 |       0.2754 |
| fine-tune - frozen                | neural/finetune/lejepa_guided   | neural/frozen/lejepa_guided   | nrmse    |     -0.005  |    0.009  |           10 |       3 |       0.084  |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_unguided | neural/finetune/random0       | r        |     -0.005  |    0.0103 |           10 |       4 |       0.1602 |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_unguided | neural/finetune/random0       | r2       |     -0.0141 |    0.0271 |           10 |       4 |       0.375  |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_unguided | neural/finetune/random0       | nrmse    |      0.0035 |    0.0063 |           10 |       7 |       0.375  |
| fine-tune - frozen                | neural/finetune/lejepa_unguided | neural/frozen/lejepa_unguided | r        |      0.0079 |    0.0186 |           10 |       7 |       0.2754 |
| fine-tune - frozen                | neural/finetune/lejepa_unguided | neural/frozen/lejepa_unguided | r2       |      0.0024 |    0.0355 |           10 |       7 |       0.6953 |
| fine-tune - frozen                | neural/finetune/lejepa_unguided | neural/frozen/lejepa_unguided | nrmse    |     -0.0014 |    0.008  |           10 |       3 |       0.5566 |
| pretrained - scratch (fine-tune)  | neural/finetune/mae             | neural/finetune/random0       | r        |     -0.0041 |    0.0063 |           10 |       2 |       0.0371 |
| pretrained - scratch (fine-tune)  | neural/finetune/mae             | neural/finetune/random0       | r2       |     -0.0162 |    0.0201 |           10 |       1 |       0.0195 |
| pretrained - scratch (fine-tune)  | neural/finetune/mae             | neural/finetune/random0       | nrmse    |      0.0036 |    0.0045 |           10 |       9 |       0.0273 |
| fine-tune - frozen                | neural/finetune/mae             | neural/frozen/mae             | r        |      0.0084 |    0.0183 |           10 |       6 |       0.2324 |
| fine-tune - frozen                | neural/finetune/mae             | neural/frozen/mae             | r2       |     -0.002  |    0.0359 |           10 |       4 |       0.7695 |
| fine-tune - frozen                | neural/finetune/mae             | neural/frozen/mae             | nrmse    |     -0.0005 |    0.0082 |           10 |       6 |       1      |
| fine-tune - frozen                | neural/finetune/random0         | neural/frozen/random0         | r        |      0.0122 |    0.0194 |           10 |       7 |       0.1055 |
| fine-tune - frozen                | neural/finetune/random0         | neural/frozen/random0         | r2       |      0.0121 |    0.0384 |           10 |       5 |       0.6953 |
| fine-tune - frozen                | neural/finetune/random0         | neural/frozen/random0         | nrmse    |     -0.0044 |    0.0081 |           10 |       4 |       0.1934 |
| pretrained - random-init (frozen) | neural/frozen/jepa_guided       | neural/frozen/random0         | r        |     -0.0052 |    0.0082 |           10 |       3 |       0.084  |
| pretrained - random-init (frozen) | neural/frozen/jepa_guided       | neural/frozen/random0         | r2       |     -0.0134 |    0.021  |           10 |       3 |       0.0645 |
| pretrained - random-init (frozen) | neural/frozen/jepa_guided       | neural/frozen/random0         | nrmse    |      0.0031 |    0.0045 |           10 |       7 |       0.084  |
| pretrained - random-init (frozen) | neural/frozen/jepa_unguided     | neural/frozen/random0         | r        |      0.004  |    0.0171 |           10 |       7 |       0.5566 |
| pretrained - random-init (frozen) | neural/frozen/jepa_unguided     | neural/frozen/random0         | r2       |     -0.0011 |    0.0322 |           10 |       5 |       1      |
| pretrained - random-init (frozen) | neural/frozen/jepa_unguided     | neural/frozen/random0         | nrmse    |     -0.0005 |    0.0069 |           10 |       5 |       0.9219 |
| pretrained - random-init (frozen) | neural/frozen/lejepa_guided     | neural/frozen/random0         | r        |     -0.003  |    0.0119 |           10 |       3 |       0.2754 |
| pretrained - random-init (frozen) | neural/frozen/lejepa_guided     | neural/frozen/random0         | r2       |     -0.0097 |    0.0265 |           10 |       2 |       0.1309 |
| pretrained - random-init (frozen) | neural/frozen/lejepa_guided     | neural/frozen/random0         | nrmse    |      0.0018 |    0.0053 |           10 |       8 |       0.1934 |
| pretrained - random-init (frozen) | neural/frozen/lejepa_unguided   | neural/frozen/random0         | r        |     -0.0007 |    0.0113 |           10 |       4 |       0.8457 |
| pretrained - random-init (frozen) | neural/frozen/lejepa_unguided   | neural/frozen/random0         | r2       |     -0.0045 |    0.0298 |           10 |       4 |       0.4922 |
| pretrained - random-init (frozen) | neural/frozen/lejepa_unguided   | neural/frozen/random0         | nrmse    |      0.0005 |    0.0058 |           10 |       6 |       0.5566 |
| pretrained - random-init (frozen) | neural/frozen/mae               | neural/frozen/random0         | r        |     -0.0003 |    0.0126 |           10 |       6 |       0.8457 |
| pretrained - random-init (frozen) | neural/frozen/mae               | neural/frozen/random0         | r2       |     -0.0021 |    0.0247 |           10 |       4 |       0.625  |
| pretrained - random-init (frozen) | neural/frozen/mae               | neural/frozen/random0         | nrmse    |     -0.0002 |    0.0057 |           10 |       4 |       0.9219 |

## Paired comparisons, trial-specific (SSL-27)

A trial-specific difference is a finding only if p < 0.05 on residual_r_matched (Wilcoxon, uncorrected); significant on residual_r_specific only reads 'not shown'. Matched values for older runs from output/proposed-model/rescoring/matched/20261126-083428.

| comparison                        | a                               |   mean_diff_matched |   mean_diff_rrs |   n_pos_matched |   n_pos_rrs |   p_wilcoxon_matched |   p_wilcoxon_rrs | reading_rrs   | reading_matched   | changed   |
|:----------------------------------|:--------------------------------|--------------------:|----------------:|----------------:|------------:|---------------------:|-----------------:|:--------------|:------------------|:----------|
| fine-tune - frozen                | hybrid/finetune/jepa_unguided   |              0.0258 |          0.024  |               8 |           7 |               0.0195 |           0.0371 | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | hybrid/finetune/lejepa_guided   |              0.0271 |          0.0315 |              10 |           8 |               0.002  |           0.0098 | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | hybrid/finetune/lejepa_unguided |              0.0238 |          0.0274 |               9 |           9 |               0.0098 |           0.0098 | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | hybrid/finetune/mae             |              0.0373 |          0.0453 |               9 |           9 |               0.0059 |           0.0137 | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | hybrid/finetune/random0         |              0.0237 |          0.02   |               7 |           7 |               0.0645 |           0.1934 | not shown     | not shown         | False     |
| fine-tune - frozen                | neural/finetune/jepa_unguided   |              0.0428 |          0.0497 |               9 |           9 |               0.0039 |           0.0059 | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | neural/finetune/lejepa_guided   |              0.0512 |          0.0616 |              10 |          10 |               0.002  |           0.002  | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | neural/finetune/lejepa_unguided |              0.0487 |          0.0626 |              10 |          10 |               0.002  |           0.002  | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | neural/finetune/mae             |              0.041  |          0.0343 |               9 |           9 |               0.0039 |           0.0371 | finding (+)   | finding (+)       | False     |
| fine-tune - frozen                | neural/finetune/random0         |              0.0625 |          0.0746 |              10 |          10 |               0.002  |           0.002  | finding (+)   | finding (+)       | False     |
| hybrid - ocular-only              | hybrid/finetune/jepa_unguided   |              0.064  |          0.0779 |               9 |           9 |               0.0039 |           0.0059 | finding (+)   | finding (+)       | False     |
| hybrid - ocular-only              | hybrid/finetune/lejepa_guided   |              0.0253 |          0.0354 |               6 |           9 |               0.1934 |           0.0371 | finding (+)   | not shown         | True      |
| hybrid - ocular-only              | hybrid/finetune/lejepa_unguided |              0.0283 |          0.0379 |               7 |           8 |               0.0488 |           0.0645 | not shown     | finding (+)       | True      |
| hybrid - ocular-only              | hybrid/finetune/mae             |              0.0512 |          0.0707 |               8 |           9 |               0.0137 |           0.0059 | finding (+)   | finding (+)       | False     |
| hybrid - ocular-only              | hybrid/finetune/random0         |              0.0347 |          0.0447 |               8 |           8 |               0.0098 |           0.0371 | finding (+)   | finding (+)       | False     |
| hybrid - ocular-only              | hybrid/frozen/jepa_guided       |              0.0035 |          0.0051 |               5 |           5 |               0.7695 |           0.625  | not shown     | not shown         | False     |
| hybrid - ocular-only              | hybrid/frozen/jepa_unguided     |              0.0382 |          0.0538 |               8 |           9 |               0.0098 |           0.0137 | finding (+)   | finding (+)       | False     |
| hybrid - ocular-only              | hybrid/frozen/lejepa_guided     |             -0.0018 |          0.0039 |               4 |           4 |               0.8457 |           1      | not shown     | not shown         | False     |
| hybrid - ocular-only              | hybrid/frozen/lejepa_unguided   |              0.0045 |          0.0106 |               5 |           5 |               0.8457 |           0.6953 | not shown     | not shown         | False     |
| hybrid - ocular-only              | hybrid/frozen/mae               |              0.0139 |          0.0254 |               5 |           6 |               0.375  |           0.1934 | not shown     | not shown         | False     |
| hybrid - ocular-only              | hybrid/frozen/random0           |              0.011  |          0.0247 |               7 |           7 |               0.375  |           0.1934 | not shown     | not shown         | False     |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_guided       |             -0.0075 |         -0.0195 |               3 |           3 |               0.1602 |           0.084  | not shown     | not shown         | False     |
| pretrained - random-init (frozen) | hybrid/frozen/jepa_unguided     |              0.0272 |          0.0292 |               9 |           9 |               0.0195 |           0.0137 | finding (+)   | finding (+)       | False     |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_guided     |             -0.0128 |         -0.0207 |               2 |           3 |               0.0488 |           0.0273 | finding (-)   | finding (-)       | False     |
| pretrained - random-init (frozen) | hybrid/frozen/lejepa_unguided   |             -0.0065 |         -0.0141 |               3 |           2 |               0.4316 |           0.1309 | not shown     | not shown         | False     |
| pretrained - random-init (frozen) | hybrid/frozen/mae               |              0.0029 |          0.0007 |               4 |           3 |               0.7695 |           0.625  | not shown     | not shown         | False     |
| pretrained - random-init (frozen) | neural/frozen/jepa_guided       |             -0.0239 |         -0.039  |               1 |           2 |               0.0645 |           0.0273 | finding (-)   | not shown         | True      |
| pretrained - random-init (frozen) | neural/frozen/jepa_unguided     |              0.0442 |          0.0498 |               9 |          10 |               0.0039 |           0.002  | finding (+)   | finding (+)       | False     |
| pretrained - random-init (frozen) | neural/frozen/lejepa_guided     |              0.0028 |         -0.0055 |               6 |           5 |               0.5566 |           0.6953 | not shown     | not shown         | False     |
| pretrained - random-init (frozen) | neural/frozen/lejepa_unguided   |              0.0036 |         -0.0071 |               5 |           4 |               0.9219 |           0.4922 | not shown     | not shown         | False     |
| pretrained - random-init (frozen) | neural/frozen/mae               |              0.0277 |          0.0402 |               8 |          10 |               0.0098 |           0.002  | finding (+)   | finding (+)       | False     |
| pretrained - scratch (fine-tune)  | hybrid/finetune/jepa_unguided   |              0.0292 |          0.0332 |               9 |           8 |               0.0039 |           0.0098 | finding (+)   | finding (+)       | False     |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_guided   |             -0.0094 |         -0.0093 |               3 |           4 |               0.1934 |           0.3223 | not shown     | not shown         | False     |
| pretrained - scratch (fine-tune)  | hybrid/finetune/lejepa_unguided |             -0.0065 |         -0.0067 |               4 |           4 |               0.2324 |           0.4922 | not shown     | not shown         | False     |
| pretrained - scratch (fine-tune)  | hybrid/finetune/mae             |              0.0165 |          0.026  |               8 |           9 |               0.0645 |           0.0273 | finding (+)   | not shown         | True      |
| pretrained - scratch (fine-tune)  | neural/finetune/jepa_unguided   |              0.0246 |          0.0248 |               7 |           7 |               0.0273 |           0.1055 | not shown     | finding (+)       | True      |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_guided   |             -0.0085 |         -0.0185 |               2 |           4 |               0.1934 |           0.1055 | not shown     | not shown         | False     |
| pretrained - scratch (fine-tune)  | neural/finetune/lejepa_unguided |             -0.0102 |         -0.0192 |               3 |           2 |               0.1602 |           0.0645 | not shown     | not shown         | False     |
| pretrained - scratch (fine-tune)  | neural/finetune/mae             |              0.0062 |         -0.0002 |               6 |           4 |               0.6953 |           1      | not shown     | not shown         | False     |

### Readings that change (5 of 39)

| comparison                        | a                               |   mean_diff_matched |   mean_diff_rrs |   n_pos_matched |   n_pos_rrs |   p_wilcoxon_matched |   p_wilcoxon_rrs | reading_rrs   | reading_matched   | changed   |
|:----------------------------------|:--------------------------------|--------------------:|----------------:|----------------:|------------:|---------------------:|-----------------:|:--------------|:------------------|:----------|
| hybrid - ocular-only              | hybrid/finetune/lejepa_guided   |              0.0253 |          0.0354 |               6 |           9 |               0.1934 |           0.0371 | finding (+)   | not shown         | True      |
| hybrid - ocular-only              | hybrid/finetune/lejepa_unguided |              0.0283 |          0.0379 |               7 |           8 |               0.0488 |           0.0645 | not shown     | finding (+)       | True      |
| pretrained - random-init (frozen) | neural/frozen/jepa_guided       |             -0.0239 |         -0.039  |               1 |           2 |               0.0645 |           0.0273 | finding (-)   | not shown         | True      |
| pretrained - scratch (fine-tune)  | hybrid/finetune/mae             |              0.0165 |          0.026  |               8 |           9 |               0.0645 |           0.0273 | finding (+)   | not shown         | True      |
| pretrained - scratch (fine-tune)  | neural/finetune/jepa_unguided   |              0.0246 |          0.0248 |               7 |           7 |               0.0273 |           0.1055 | not shown     | finding (+)       | True      |

## Versus the head floor on residual_r_matched

The head floor's matched score is 0, so this tests matched > 0. residual_r_specific is not compared with the floors (window effect, note below).

| comparison          | a                               | b          | metric             |   mean_diff |   sd_diff |   n_subjects |   n_pos |   p_wilcoxon |
|:--------------------|:--------------------------------|:-----------|:-------------------|------------:|----------:|-------------:|--------:|-------------:|
| config - head floor | hybrid/finetune/jepa_unguided   | head floor | residual_r_matched |      0.16   |    0.0761 |           10 |      10 |        0.002 |
| config - head floor | hybrid/finetune/lejepa_guided   | head floor | residual_r_matched |      0.1214 |    0.0624 |           10 |      10 |        0.002 |
| config - head floor | hybrid/finetune/lejepa_unguided | head floor | residual_r_matched |      0.1243 |    0.0578 |           10 |      10 |        0.002 |
| config - head floor | hybrid/finetune/mae             | head floor | residual_r_matched |      0.1472 |    0.0657 |           10 |      10 |        0.002 |
| config - head floor | hybrid/finetune/random0         | head floor | residual_r_matched |      0.1308 |    0.0547 |           10 |      10 |        0.002 |
| config - head floor | hybrid/frozen/jepa_guided       | head floor | residual_r_matched |      0.0996 |    0.0545 |           10 |      10 |        0.002 |
| config - head floor | hybrid/frozen/jepa_unguided     | head floor | residual_r_matched |      0.1342 |    0.0599 |           10 |      10 |        0.002 |
| config - head floor | hybrid/frozen/lejepa_guided     | head floor | residual_r_matched |      0.0942 |    0.0489 |           10 |      10 |        0.002 |
| config - head floor | hybrid/frozen/lejepa_unguided   | head floor | residual_r_matched |      0.1005 |    0.0458 |           10 |      10 |        0.002 |
| config - head floor | hybrid/frozen/mae               | head floor | residual_r_matched |      0.11   |    0.0557 |           10 |      10 |        0.002 |
| config - head floor | hybrid/frozen/random0           | head floor | residual_r_matched |      0.107  |    0.0502 |           10 |      10 |        0.002 |
| config - head floor | neural/finetune/jepa_unguided   | head floor | residual_r_matched |      0.1623 |    0.0759 |           10 |      10 |        0.002 |
| config - head floor | neural/finetune/lejepa_guided   | head floor | residual_r_matched |      0.1292 |    0.0595 |           10 |      10 |        0.002 |
| config - head floor | neural/finetune/lejepa_unguided | head floor | residual_r_matched |      0.1275 |    0.063  |           10 |      10 |        0.002 |
| config - head floor | neural/finetune/mae             | head floor | residual_r_matched |      0.1439 |    0.0741 |           10 |      10 |        0.002 |
| config - head floor | neural/finetune/random0         | head floor | residual_r_matched |      0.1377 |    0.0581 |           10 |      10 |        0.002 |
| config - head floor | neural/frozen/jepa_guided       | head floor | residual_r_matched |      0.0514 |    0.0427 |           10 |      10 |        0.002 |
| config - head floor | neural/frozen/jepa_unguided     | head floor | residual_r_matched |      0.1195 |    0.0527 |           10 |      10 |        0.002 |
| config - head floor | neural/frozen/lejepa_guided     | head floor | residual_r_matched |      0.078  |    0.039  |           10 |      10 |        0.002 |
| config - head floor | neural/frozen/lejepa_unguided   | head floor | residual_r_matched |      0.0788 |    0.0384 |           10 |      10 |        0.002 |
| config - head floor | neural/frozen/mae               | head floor | residual_r_matched |      0.1029 |    0.0502 |           10 |      10 |        0.002 |
| config - head floor | neural/frozen/random0           | head floor | residual_r_matched |      0.0752 |    0.0404 |           10 |      10 |        0.002 |
| config - head floor | ocular/frozen/noenc             | head floor | residual_r_matched |      0.096  |    0.0612 |           10 |      10 |        0.002 |

## Versus template_cue and the head floor (r)

| comparison            | a                               | b            | metric   |   mean_diff |   sd_diff |   n_subjects |   n_pos |   p_wilcoxon |
|:----------------------|:--------------------------------|:-------------|:---------|------------:|----------:|-------------:|--------:|-------------:|
| config - template_cue | hybrid/finetune/jepa_unguided   | template_cue | r        |      0.0173 |    0.0499 |           10 |       5 |       0.5566 |
| config - head floor   | hybrid/finetune/jepa_unguided   | head floor   | r        |      0.016  |    0.0462 |           10 |       6 |       0.5566 |
| config - template_cue | hybrid/finetune/lejepa_guided   | template_cue | r        |      0.0123 |    0.0516 |           10 |       3 |       0.8457 |
| config - head floor   | hybrid/finetune/lejepa_guided   | head floor   | r        |      0.0111 |    0.0484 |           10 |       3 |       0.9219 |
| config - template_cue | hybrid/finetune/lejepa_unguided | template_cue | r        |      0.0121 |    0.0493 |           10 |       4 |       0.7695 |
| config - head floor   | hybrid/finetune/lejepa_unguided | head floor   | r        |      0.0109 |    0.0456 |           10 |       4 |       0.7695 |
| config - template_cue | hybrid/finetune/mae             | template_cue | r        |      0.0163 |    0.0507 |           10 |       5 |       0.625  |
| config - head floor   | hybrid/finetune/mae             | head floor   | r        |      0.015  |    0.0472 |           10 |       5 |       0.5566 |
| config - template_cue | hybrid/finetune/random0         | template_cue | r        |      0.0144 |    0.0527 |           10 |       4 |       0.6953 |
| config - head floor   | hybrid/finetune/random0         | head floor   | r        |      0.0131 |    0.0492 |           10 |       4 |       0.6953 |
| config - template_cue | hybrid/frozen/jepa_guided       | template_cue | r        |      0.004  |    0.0432 |           10 |       4 |       0.6953 |
| config - head floor   | hybrid/frozen/jepa_guided       | head floor   | r        |      0.0028 |    0.0398 |           10 |       4 |       0.625  |
| config - template_cue | hybrid/frozen/jepa_unguided     | template_cue | r        |      0.0128 |    0.0418 |           10 |       6 |       0.5566 |
| config - head floor   | hybrid/frozen/jepa_unguided     | head floor   | r        |      0.0116 |    0.038  |           10 |       5 |       0.625  |
| config - template_cue | hybrid/frozen/lejepa_guided     | template_cue | r        |      0.0048 |    0.04   |           10 |       4 |       0.6953 |
| config - head floor   | hybrid/frozen/lejepa_guided     | head floor   | r        |      0.0035 |    0.0365 |           10 |       3 |       0.5566 |
| config - template_cue | hybrid/frozen/lejepa_unguided   | template_cue | r        |      0.0072 |    0.0432 |           10 |       3 |       0.7695 |
| config - head floor   | hybrid/frozen/lejepa_unguided   | head floor   | r        |      0.006  |    0.0399 |           10 |       4 |       0.8457 |
| config - template_cue | hybrid/frozen/mae               | template_cue | r        |      0.0061 |    0.0503 |           10 |       3 |       0.6953 |
| config - head floor   | hybrid/frozen/mae               | head floor   | r        |      0.0048 |    0.047  |           10 |       3 |       0.5566 |
| config - template_cue | hybrid/frozen/random0           | template_cue | r        |      0.0075 |    0.0404 |           10 |       4 |       1      |
| config - head floor   | hybrid/frozen/random0           | head floor   | r        |      0.0062 |    0.0372 |           10 |       4 |       0.9219 |
| config - template_cue | neural/finetune/jepa_unguided   | template_cue | r        |      0.02   |    0.0582 |           10 |       4 |       0.625  |
| config - head floor   | neural/finetune/jepa_unguided   | head floor   | r        |      0.0187 |    0.0545 |           10 |       4 |       0.6953 |
| config - template_cue | neural/finetune/lejepa_guided   | template_cue | r        |      0.0157 |    0.047  |           10 |       5 |       0.4922 |
| config - head floor   | neural/finetune/lejepa_guided   | head floor   | r        |      0.0145 |    0.043  |           10 |       5 |       0.5566 |
| config - template_cue | neural/finetune/lejepa_unguided | template_cue | r        |      0.0126 |    0.0534 |           10 |       5 |       0.9219 |
| config - head floor   | neural/finetune/lejepa_unguided | head floor   | r        |      0.0114 |    0.0501 |           10 |       5 |       0.9219 |
| config - template_cue | neural/finetune/mae             | template_cue | r        |      0.0134 |    0.0542 |           10 |       4 |       1      |
| config - head floor   | neural/finetune/mae             | head floor   | r        |      0.0122 |    0.0509 |           10 |       4 |       1      |
| config - template_cue | neural/finetune/random0         | template_cue | r        |      0.0176 |    0.0497 |           10 |       4 |       0.6953 |
| config - head floor   | neural/finetune/random0         | head floor   | r        |      0.0164 |    0.0462 |           10 |       4 |       0.6953 |
| config - template_cue | neural/frozen/jepa_guided       | template_cue | r        |      0.0002 |    0.0379 |           10 |       4 |       0.4922 |
| config - head floor   | neural/frozen/jepa_guided       | head floor   | r        |     -0.001  |    0.0345 |           10 |       3 |       0.4922 |
| config - template_cue | neural/frozen/jepa_unguided     | template_cue | r        |      0.0093 |    0.0477 |           10 |       5 |       0.9219 |
| config - head floor   | neural/frozen/jepa_unguided     | head floor   | r        |      0.0081 |    0.044  |           10 |       5 |       0.9219 |
| config - template_cue | neural/frozen/lejepa_guided     | template_cue | r        |      0.0024 |    0.0385 |           10 |       3 |       0.5566 |
| config - head floor   | neural/frozen/lejepa_guided     | head floor   | r        |      0.0011 |    0.0355 |           10 |       4 |       0.625  |
| config - template_cue | neural/frozen/lejepa_unguided   | template_cue | r        |      0.0047 |    0.0371 |           10 |       4 |       0.625  |
| config - head floor   | neural/frozen/lejepa_unguided   | head floor   | r        |      0.0034 |    0.0339 |           10 |       4 |       0.625  |
| config - template_cue | neural/frozen/mae               | template_cue | r        |      0.0051 |    0.0441 |           10 |       4 |       0.625  |
| config - head floor   | neural/frozen/mae               | head floor   | r        |      0.0038 |    0.0403 |           10 |       3 |       0.4922 |
| config - template_cue | neural/frozen/random0           | template_cue | r        |      0.0053 |    0.0356 |           10 |       5 |       0.7695 |
| config - head floor   | neural/frozen/random0           | head floor   | r        |      0.0041 |    0.0325 |           10 |       4 |       0.625  |
| config - template_cue | ocular/frozen/noenc             | template_cue | r        |      0.0018 |    0.0332 |           10 |       3 |       0.6953 |
| config - head floor   | ocular/frozen/noenc             | head floor   | r        |      0.0006 |    0.0301 |           10 |       5 |       0.625  |

## Note on residual_r_specific

Source: output/proposed-model/diagnostics/head-floor-rrs/20261126-010952. `residual_r` uses the whole scored window, the other-trial control only the bins two trials share. A predictor that is one function of time since LED-on therefore has rrs different from 0 on the scored windows, and 0 on one fixed window. Cause: the variable window end (tHandStop), see `fixed_end`. rrs above 0 alone is no evidence of trial-specific information.

| group              |   fixed |   fixed_end |   fixed_start |   scored |
|:-------------------|--------:|------------:|--------------:|---------:|
| constant_0         |      -0 |      0.0013 |        0.0096 |   0.0133 |
| cue_shift_+0.2s    |      -0 |      0.0003 |       -0.0275 |  -0.0288 |
| cue_shift_-0.2s    |       0 |      0.0039 |        0.0348 |   0.0413 |
| head_floor         |       0 |     -0.0022 |        0.0338 |   0.0333 |
| random_smooth (20) |      -0 |      0.0002 |       -0.0044 |  -0.0043 |
| unconditional_mean |      -0 |      0.0006 |        0.1721 |   0.1718 |

Experiment 1 configurations: window effect -0.002 to 0.029; matched-bin score (0 for every rigid predictor) 0.051 to 0.162.
