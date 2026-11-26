# residual_r_matched for saved decoder runs

78 runs, 780 run x subject rows. Reproduction against metrics.json: max |rrs difference| 0.00e+00, max |r difference| 0.00e+00.

Mean over runs (seeds) and subjects:

| name                                              |   residual_r_specific |   residual_r_matched |
|:--------------------------------------------------|----------------------:|---------------------:|
| exp1-occl__proposed_hybrid_finetune_lejepa_guided |                0.1514 |               0.1264 |
| exp1-occl__proposed_hybrid_finetune_random0       |                0.1546 |               0.1293 |
| exp1-occl__proposed_hybrid_frozen_jepa_guided     |                0.1117 |               0.1019 |
| exp1-occl__proposed_hybrid_frozen_jepa_unguided   |                0.1424 |               0.1249 |
| exp1-occl__proposed_hybrid_frozen_lejepa_guided   |                0.1119 |               0.0982 |
| exp1-occl__proposed_hybrid_frozen_lejepa_unguided |                0.1217 |               0.1096 |
| exp1-occl__proposed_hybrid_frozen_mae             |                0.1386 |               0.1145 |
| exp1-occl__proposed_hybrid_frozen_random0         |                0.1311 |               0.1065 |
| exp1-occl__proposed_neural_finetune_lejepa_guided |                0.1462 |               0.1264 |
| exp1__proposed_hybrid_finetune_jepa_unguided      |                0.1826 |               0.16   |
| exp1__proposed_hybrid_finetune_lejepa_guided      |                0.1401 |               0.1214 |
| exp1__proposed_hybrid_finetune_lejepa_unguided    |                0.1427 |               0.1243 |
| exp1__proposed_hybrid_finetune_mae                |                0.1754 |               0.1472 |
| exp1__proposed_hybrid_finetune_random0            |                0.1494 |               0.1308 |
| exp1__proposed_hybrid_frozen_jepa_guided          |                0.1099 |               0.0996 |
| exp1__proposed_hybrid_frozen_jepa_unguided        |                0.1586 |               0.1342 |
| exp1__proposed_hybrid_frozen_lejepa_guided        |                0.1087 |               0.0942 |
| exp1__proposed_hybrid_frozen_lejepa_unguided      |                0.1153 |               0.1005 |
| exp1__proposed_hybrid_frozen_mae                  |                0.1301 |               0.11   |
| exp1__proposed_hybrid_frozen_random0              |                0.1294 |               0.107  |
| exp1__proposed_neural_finetune_jepa_unguided      |                0.1893 |               0.1623 |
| exp1__proposed_neural_finetune_lejepa_guided      |                0.1459 |               0.1292 |
| exp1__proposed_neural_finetune_lejepa_unguided    |                0.1452 |               0.1275 |
| exp1__proposed_neural_finetune_mae                |                0.1642 |               0.1439 |
| exp1__proposed_neural_finetune_random0            |                0.1644 |               0.1377 |
| exp1__proposed_neural_frozen_jepa_guided          |                0.0508 |               0.0514 |
| exp1__proposed_neural_frozen_jepa_unguided        |                0.1396 |               0.1195 |
| exp1__proposed_neural_frozen_lejepa_guided        |                0.0843 |               0.078  |
| exp1__proposed_neural_frozen_lejepa_unguided      |                0.0827 |               0.0788 |
| exp1__proposed_neural_frozen_mae                  |                0.13   |               0.1029 |
| exp1__proposed_neural_frozen_random0              |                0.0898 |               0.0752 |
| exp1__proposed_ocular_frozen_noenc                |                0.1047 |               0.096  |
