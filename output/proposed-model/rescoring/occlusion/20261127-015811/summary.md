# Occlusion drops with residual_r_matched

27 runs, device cuda. Checks against the saved run (max |difference|): unoccluded y 0.00e+00, r 0.00e+00, rrs 0.00e+00.

Mean drop (full - occluded) over seeds and subjects:

|                                                                        |   drop_r |   drop_residual_r_specific |   drop_residual_r_matched |
|:-----------------------------------------------------------------------|---------:|---------------------------:|--------------------------:|
| ('exp1-occl__proposed_hybrid_finetune_lejepa_guided', 'contralateral') |   0.0184 |                     0.0523 |                    0.0275 |
| ('exp1-occl__proposed_hybrid_finetune_lejepa_guided', 'frontal')       |   0.0072 |                     0.0094 |                    0.0045 |
| ('exp1-occl__proposed_hybrid_finetune_lejepa_guided', 'ipsilateral')   |   0.0034 |                     0.0146 |                    0.0071 |
| ('exp1-occl__proposed_hybrid_finetune_lejepa_guided', 'occipital')     |   0.0115 |                     0.0514 |                    0.035  |
| ('exp1-occl__proposed_hybrid_finetune_lejepa_guided', 'prefrontal')    |   0.0013 |                     0.0077 |                    0.0065 |
| ('exp1-occl__proposed_hybrid_finetune_random0', 'contralateral')       |   0.0228 |                     0.0572 |                    0.0316 |
| ('exp1-occl__proposed_hybrid_finetune_random0', 'frontal')             |   0.007  |                     0.0197 |                    0.0159 |
| ('exp1-occl__proposed_hybrid_finetune_random0', 'ipsilateral')         |   0.0023 |                     0.0096 |                    0.0091 |
| ('exp1-occl__proposed_hybrid_finetune_random0', 'occipital')           |   0.0142 |                     0.0475 |                    0.0324 |
| ('exp1-occl__proposed_hybrid_finetune_random0', 'prefrontal')          |  -0.0005 |                     0.011  |                    0.0071 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_guided', 'contralateral')     |   0.0064 |                     0.0284 |                    0.0092 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_guided', 'frontal')           |   0.0025 |                     0.0132 |                    0.0139 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_guided', 'ipsilateral')       |  -0.0008 |                     0.01   |                    0.0031 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_guided', 'occipital')         |   0.0048 |                     0.0083 |                    0.0042 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_guided', 'prefrontal')        |  -0.0005 |                    -0.0037 |                   -0.0021 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_unguided', 'contralateral')   |   0.0047 |                     0.0264 |                    0.0098 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_unguided', 'frontal')         |   0.011  |                     0.0239 |                    0.031  |
| ('exp1-occl__proposed_hybrid_frozen_jepa_unguided', 'ipsilateral')     |   0.0021 |                     0.0132 |                    0.0053 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_unguided', 'occipital')       |   0.0049 |                     0.0123 |                    0.0103 |
| ('exp1-occl__proposed_hybrid_frozen_jepa_unguided', 'prefrontal')      |   0.0092 |                     0.0328 |                    0.0367 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_guided', 'contralateral')   |   0.0078 |                     0.0429 |                    0.0177 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_guided', 'frontal')         |   0.005  |                     0.0173 |                    0.0167 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_guided', 'ipsilateral')     |   0.0007 |                     0.0125 |                    0.0006 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_guided', 'occipital')       |   0.0091 |                     0.0117 |                    0.0114 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_guided', 'prefrontal')      |   0.0009 |                     0.0007 |                    0.0033 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_unguided', 'contralateral') |   0.0142 |                     0.0524 |                    0.0301 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_unguided', 'frontal')       |   0.0054 |                     0.0093 |                    0.0113 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_unguided', 'ipsilateral')   |   0.0014 |                     0.0255 |                    0.0105 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_unguided', 'occipital')     |   0.007  |                     0.019  |                    0.0189 |
| ('exp1-occl__proposed_hybrid_frozen_lejepa_unguided', 'prefrontal')    |  -0.0001 |                    -0.0004 |                    0.0006 |
| ('exp1-occl__proposed_hybrid_frozen_mae', 'contralateral')             |   0.0042 |                     0.0417 |                    0.016  |
| ('exp1-occl__proposed_hybrid_frozen_mae', 'frontal')                   |  -0.001  |                     0.0119 |                    0.0086 |
| ('exp1-occl__proposed_hybrid_frozen_mae', 'ipsilateral')               |   0.0009 |                     0.0122 |                    0.0063 |
| ('exp1-occl__proposed_hybrid_frozen_mae', 'occipital')                 |   0.0036 |                     0.0148 |                    0.0101 |
| ('exp1-occl__proposed_hybrid_frozen_mae', 'prefrontal')                |  -0.0006 |                     0.0103 |                    0.0052 |
| ('exp1-occl__proposed_hybrid_frozen_random0', 'contralateral')         |   0.0047 |                     0.034  |                    0.0125 |
| ('exp1-occl__proposed_hybrid_frozen_random0', 'frontal')               |   0.0027 |                     0.0091 |                    0.0117 |
| ('exp1-occl__proposed_hybrid_frozen_random0', 'ipsilateral')           |   0.0008 |                     0.0079 |                    0.0003 |
| ('exp1-occl__proposed_hybrid_frozen_random0', 'occipital')             |   0.002  |                     0.0083 |                    0.0033 |
| ('exp1-occl__proposed_hybrid_frozen_random0', 'prefrontal')            |   0.0004 |                     0.0015 |                    0.0023 |
| ('exp1-occl__proposed_neural_finetune_lejepa_guided', 'contralateral') |   0.0161 |                     0.0555 |                    0.0328 |
| ('exp1-occl__proposed_neural_finetune_lejepa_guided', 'frontal')       |   0.0087 |                     0.0034 |                    0.0001 |
| ('exp1-occl__proposed_neural_finetune_lejepa_guided', 'ipsilateral')   |   0.0042 |                     0.0171 |                    0.0101 |
| ('exp1-occl__proposed_neural_finetune_lejepa_guided', 'occipital')     |   0.0161 |                     0.0633 |                    0.0448 |
| ('exp1-occl__proposed_neural_finetune_lejepa_guided', 'prefrontal')    |   0.0023 |                     0.011  |                    0.0073 |
| ('exp1__proposed_hybrid_finetune_jepa_unguided', 'contralateral')      |   0.0096 |                     0.0388 |                    0.0244 |
| ('exp1__proposed_hybrid_finetune_jepa_unguided', 'frontal')            |   0.0103 |                     0.0464 |                    0.0347 |
| ('exp1__proposed_hybrid_finetune_jepa_unguided', 'ipsilateral')        |   0.0025 |                     0.0133 |                    0.0083 |
| ('exp1__proposed_hybrid_finetune_jepa_unguided', 'occipital')          |   0.0094 |                     0.0383 |                    0.0336 |
| ('exp1__proposed_hybrid_finetune_jepa_unguided', 'prefrontal')         |   0.0088 |                     0.0649 |                    0.048  |
| ('exp1__proposed_hybrid_finetune_lejepa_unguided', 'contralateral')    |   0.017  |                     0.0565 |                    0.0304 |
| ('exp1__proposed_hybrid_finetune_lejepa_unguided', 'frontal')          |   0.0056 |                     0.0088 |                    0.0045 |
| ('exp1__proposed_hybrid_finetune_lejepa_unguided', 'ipsilateral')      |   0.0035 |                     0.0166 |                    0.0103 |
| ('exp1__proposed_hybrid_finetune_lejepa_unguided', 'occipital')        |   0.0132 |                     0.0579 |                    0.0435 |
| ('exp1__proposed_hybrid_finetune_lejepa_unguided', 'prefrontal')       |   0.0016 |                     0.0036 |                    0.0034 |
| ('exp1__proposed_hybrid_finetune_mae', 'contralateral')                |   0.015  |                     0.0497 |                    0.028  |
| ('exp1__proposed_hybrid_finetune_mae', 'frontal')                      |   0.0066 |                     0.0152 |                    0.01   |
| ('exp1__proposed_hybrid_finetune_mae', 'ipsilateral')                  |   0.0036 |                     0.0094 |                    0.0068 |
| ('exp1__proposed_hybrid_finetune_mae', 'occipital')                    |   0.0146 |                     0.0581 |                    0.0423 |
| ('exp1__proposed_hybrid_finetune_mae', 'prefrontal')                   |   0.0023 |                     0.0181 |                    0.0142 |
| ('exp1__proposed_neural_finetune_jepa_unguided', 'contralateral')      |   0.0121 |                     0.0389 |                    0.0234 |
| ('exp1__proposed_neural_finetune_jepa_unguided', 'frontal')            |   0.0111 |                     0.0464 |                    0.0374 |
| ('exp1__proposed_neural_finetune_jepa_unguided', 'ipsilateral')        |   0.004  |                     0.017  |                    0.0119 |
| ('exp1__proposed_neural_finetune_jepa_unguided', 'occipital')          |   0.0125 |                     0.0412 |                    0.0331 |
| ('exp1__proposed_neural_finetune_jepa_unguided', 'prefrontal')         |   0.0114 |                     0.0659 |                    0.0492 |
| ('exp1__proposed_neural_finetune_lejepa_unguided', 'contralateral')    |   0.015  |                     0.0589 |                    0.0311 |
| ('exp1__proposed_neural_finetune_lejepa_unguided', 'frontal')          |   0.0053 |                     0.0123 |                    0.0082 |
| ('exp1__proposed_neural_finetune_lejepa_unguided', 'ipsilateral')      |   0.006  |                     0.0185 |                    0.0132 |
| ('exp1__proposed_neural_finetune_lejepa_unguided', 'occipital')        |   0.0139 |                     0.0649 |                    0.048  |
| ('exp1__proposed_neural_finetune_lejepa_unguided', 'prefrontal')       |   0.0018 |                     0.0068 |                    0.0056 |
| ('exp1__proposed_neural_finetune_mae', 'contralateral')                |   0.0184 |                     0.0527 |                    0.032  |
| ('exp1__proposed_neural_finetune_mae', 'frontal')                      |   0.0042 |                     0.0152 |                    0.0103 |
| ('exp1__proposed_neural_finetune_mae', 'ipsilateral')                  |   0.0038 |                     0.0071 |                    0.0071 |
| ('exp1__proposed_neural_finetune_mae', 'occipital')                    |   0.0128 |                     0.0581 |                    0.0406 |
| ('exp1__proposed_neural_finetune_mae', 'prefrontal')                   |   0.0042 |                     0.0221 |                    0.0159 |
