# Ocular coupling of the Experiment 1 predictions

Mean |Spearman rho| across lifts between the trial-specific prediction (per axis) and the VEOG / HEOG change from the pre-cue baseline (6 pairs), per subject, mean over seeds and subjects. `target`: the same for the true trajectory. Paired tests: per-subject mean |rho| vs the random-init frozen control of the same branches (Wilcoxon).

| config                        |   mean_abs_rho |   sd_over_subjects |   x~HEOG |   x~VEOG |   y~HEOG |   y~VEOG |   z~HEOG |   z~VEOG |
|:------------------------------|---------------:|-------------------:|---------:|---------:|---------:|---------:|---------:|---------:|
| ocular_frozen_noenc           |          0.272 |              0.129 |    0.226 |    0.214 |   -0.281 |   -0.27  |    0.305 |    0.243 |
| hybrid_frozen_random0         |          0.199 |              0.103 |    0.201 |    0.124 |   -0.206 |   -0.197 |    0.162 |    0.185 |
| hybrid_frozen_mae             |          0.186 |              0.105 |    0.151 |    0.106 |   -0.157 |   -0.147 |    0.185 |    0.137 |
| neural_frozen_random0         |          0.177 |              0.11  |    0.185 |    0.071 |   -0.194 |   -0.118 |    0.213 |    0.15  |
| neural_frozen_mae             |          0.164 |              0.097 |    0.119 |    0.077 |   -0.134 |   -0.119 |    0.195 |    0.133 |
| neural_frozen_jepa_unguided   |          0.158 |              0.091 |    0.103 |    0.099 |   -0.099 |   -0.146 |    0.118 |    0.101 |
| hybrid_frozen_jepa_unguided   |          0.157 |              0.081 |    0.11  |    0.118 |   -0.084 |   -0.146 |    0.125 |    0.102 |
| hybrid_finetune_random0       |          0.143 |              0.081 |    0.093 |    0.086 |   -0.072 |   -0.129 |    0.089 |    0.114 |
| hybrid_frozen_lejepa_guided   |          0.141 |              0.085 |    0.126 |    0.079 |   -0.114 |   -0.12  |    0.092 |    0.126 |
| hybrid_frozen_lejepa_unguided |          0.139 |              0.082 |    0.117 |    0.073 |   -0.121 |   -0.122 |    0.101 |    0.119 |
| neural_finetune_random0       |          0.139 |              0.073 |    0.105 |    0.091 |   -0.094 |   -0.113 |    0.086 |    0.112 |
| hybrid_frozen_jepa_guided     |          0.135 |              0.061 |    0.121 |    0.078 |   -0.088 |   -0.113 |    0.088 |    0.148 |
| neural_frozen_lejepa_unguided |          0.134 |              0.082 |    0.12  |    0.028 |   -0.12  |   -0.068 |    0.132 |    0.061 |
| hybrid_finetune_lejepa_guided |          0.133 |              0.076 |    0.116 |    0.066 |   -0.097 |   -0.106 |    0.044 |    0.108 |
| neural_finetune_lejepa_guided |          0.131 |              0.084 |    0.105 |    0.064 |   -0.088 |   -0.09  |    0.046 |    0.097 |
| neural_frozen_lejepa_guided   |          0.126 |              0.09  |    0.131 |    0.038 |   -0.127 |   -0.063 |    0.123 |    0.079 |
| neural_frozen_jepa_guided     |          0.092 |              0.046 |    0.068 |    0.036 |   -0.038 |   -0.038 |    0.054 |    0.048 |
| target                        |          0.081 |              0.033 |    0.033 |   -0.018 |    0.031 |    0.035 |   -0.051 |   -0.039 |

## Versus random-init (frozen)

| a                             | b                     |   mean_diff |   n_pos |   n_subjects |   p_wilcoxon |
|:------------------------------|:----------------------|------------:|--------:|-------------:|-------------:|
| hybrid_frozen_jepa_guided     | hybrid_frozen_random0 |     -0.0639 |       2 |           10 |       0.0273 |
| hybrid_frozen_jepa_unguided   | hybrid_frozen_random0 |     -0.0418 |       2 |           10 |       0.0488 |
| hybrid_frozen_lejepa_guided   | hybrid_frozen_random0 |     -0.058  |       1 |           10 |       0.0039 |
| hybrid_frozen_lejepa_unguided | hybrid_frozen_random0 |     -0.0604 |       1 |           10 |       0.0039 |
| hybrid_frozen_mae             | hybrid_frozen_random0 |     -0.0128 |       3 |           10 |       0.3223 |
| neural_frozen_jepa_guided     | neural_frozen_random0 |     -0.085  |       3 |           10 |       0.0645 |
| neural_frozen_jepa_unguided   | neural_frozen_random0 |     -0.0187 |       6 |           10 |       0.6953 |
| neural_frozen_lejepa_guided   | neural_frozen_random0 |     -0.0505 |       2 |           10 |       0.0195 |
| neural_frozen_lejepa_unguided | neural_frozen_random0 |     -0.0433 |       3 |           10 |       0.0371 |
| neural_frozen_mae             | neural_frozen_random0 |     -0.0125 |       3 |           10 |       0.3223 |
