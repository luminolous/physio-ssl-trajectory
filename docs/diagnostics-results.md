# Diagnostics results

Stage `diagnostics`, in progress. Sections are added as tasks finish. Protocol: [protocol.md](protocol.md). Baselines: [baseline-results.md](baseline-results.md).

## Cue-locked template

Run: `output/diagnostics/baselines/template_cue/20261104-234509`. Table across models: `output/diagnostics/summary/models.md`.

**The model.** The prediction is the mean training-subject trajectory as a function of time since LED-on. The experiment controls LED-on, so a real decoder could know it. This is cue timing, not oracle timing. No EEG is used.

**Scoring.** It is scored on the same protocol segment as every other model: [kinematic onset − 0.5 s, `tHandStop`]. The segment defines where we score. It is not a model input, and the prediction does not change if the test lift's onset changes. The segment is still derived from each lift's kinematics, for all models, so every score is measured over a window known to contain the movement.

**LED-on in the data.** LED-on is at 2.0 s in every lift of every subject, so time since LED-on is equivalent to time in the lift window.

### Result

| Model | Timing it uses | r (mean ± SD) | R² | NRMSE |
|---|---|---|---|---|
| Template | oracle onset | 0.864 ± 0.042 | 0.622 | 0.193 |
| **Cue-locked template** | **LED-on** | **0.803 ± 0.100** | **0.524** | **0.216** |
| CNN-LSTM | none | 0.499 ± 0.091 | 0.039 | 0.317 |
| EEGNet | none | 0.449 ± 0.075 | 0.021 | 0.323 |

- Removing oracle timing costs little: r falls by 0.061 on average.
- The cue-locked template still beats CNN-LSTM in 9 of 10 subjects on r and on R², and EEGNet in all 10 on r. P7 is the exception: CNN-LSTM r 0.676 vs 0.643.
- So the gap between EEG decoders and the template is mostly not explained by oracle timing. These decoders get no cue timing input, so the comparison is still not a test of what they could do with it.

### Reaction time and the gap to the oracle template

Reaction time is LED-on to kinematic onset. "Gap" is oracle template r minus cue-locked template r.

| Subject | Cue r | Oracle r | Gap | RT median (s) | RT IQR (s) |
|---|---|---|---|---|---|
| P1 | 0.771 | 0.812 | 0.041 | 0.306 | 0.100 |
| P2 | 0.611 | 0.818 | 0.207 | 0.603 | 0.274 |
| P4 | 0.798 | 0.848 | 0.050 | 0.302 | 0.064 |
| P5 | 0.894 | 0.913 | 0.019 | 0.483 | 0.191 |
| P7 | 0.643 | 0.798 | 0.155 | 0.694 | 0.310 |
| P8 | 0.866 | 0.890 | 0.024 | 0.357 | 0.092 |
| P9 | 0.840 | 0.882 | 0.041 | 0.356 | 0.135 |
| P10 | 0.887 | 0.911 | 0.024 | 0.382 | 0.130 |
| P11 | 0.857 | 0.883 | 0.026 | 0.385 | 0.152 |
| P12 | 0.861 | 0.885 | 0.024 | 0.443 | 0.209 |

- P2 and P7 have the slowest and most variable reaction times and the two largest gaps.
- The relationship rests mostly on those two subjects. Across the 10 subjects, the gap correlates with RT IQR at Pearson 0.73 but Spearman only 0.26.
- 8 lifts across the 10 test subjects have a negative reaction time, meaning the detected onset comes before LED-on. They are kept, as all lifts are (D8).

## Onset decoder + template

Run: `output/diagnostics/baselines/onset_template/20261104-235524`. Table across models: `output/diagnostics/summary/models.md`.

**The model.** LDA on lagged low-frequency EEG separates the 0.5 s before the kinematic onset from the 0.5 s after it, on training subjects. On a test lift the search starts at LED-on, and the decoded onset is the first sample whose probability exceeds a threshold chosen on inner validation by r. If nothing fires by the 95th percentile of training reaction times, that deadline is the onset. The prediction is a constant before the decoded onset and the oracle-aligned training template after it. The prediction at time t uses only EEG up to t, and tests check this. Scoring uses the same protocol segment, which is not a model input.

### Result

| Model | r (mean ± SD) | R² | NRMSE |
|---|---|---|---|
| Cue-locked template | 0.803 ± 0.100 | 0.524 | 0.216 |
| **Onset decoder + template** | **0.754 ± 0.094** | **0.406** | **0.241** |
| CNN-LSTM | 0.499 ± 0.091 | 0.039 | 0.317 |

- This onset decoder falls below the cue-locked template in 9 of 10 subjects (mean Δr −0.049 ± 0.055). P2 is the exception: 0.682 vs 0.611.
- It beats CNN-LSTM in 9 of 10 subjects on r and on R². P7 is the exception: 0.550 vs 0.676.
- Thresholds on validation: detection beat the deadline-only setting in every fold (mean r 0.769 vs 0.706), and the never-firing threshold was never chosen.

### Onset error

Decoded minus true onset, per test subject.

| Subject | r | Median (ms) | IQR (ms) | Within ±100 ms (%) | Reached deadline (%) | Fired at LED-on (%) |
|---|---|---|---|---|---|---|
| P1 | 0.719 | 196 | 204 | 8.5 | 17.0 | 5.8 |
| P2 | 0.682 | −88 | 330 | 34.4 | 51.0 | 2.4 |
| P4 | 0.712 | 322 | 224 | 5.4 | 25.2 | 2.4 |
| P5 | 0.845 | 198 | 338 | 20.7 | 53.7 | 8.5 |
| P7 | 0.550 | −320 | 592 | 27.9 | 7.8 | 24.5 |
| P8 | 0.729 | 500 | 595 | 9.5 | 50.7 | 9.9 |
| P9 | 0.822 | 2 | 342 | 40.8 | 6.5 | 18.7 |
| P10 | 0.836 | −137 | 345 | 32.3 | 0.3 | 27.2 |
| P11 | 0.828 | 135 | 274 | 24.5 | 20.1 | 6.5 |
| P12 | 0.812 | 89 | 486 | 18.7 | 20.4 | 14.6 |

- The onset error spread (mean IQR 373 ms) is about twice the reaction-time spread (mean IQR 166 ms). This fits the result: assuming a fixed time after LED-on is more accurate than this detector. Only 22% of lifts on average fall within ±100 ms.
- This describes one detector: cross-subject LDA with fixed shrinkage and a single first-crossing rule. It does not show that EEG holds no better onset information.

## History length

Question: does a longer causal EEG history help the deep baselines? EEGNet and CNN-LSTM, LOSO over 10 subjects, 3 seeds, fixed lr 1e-3 and weight decay 0.01. The 0.5 s rows are the data-baselines runs. Full table: `output/diagnostics/history/summary/`.

### Result

| Model | History | r | R² | NRMSE | Peak GPU (GB) | Epoch (s) | Subjects with higher r than 0.5 s |
|---|---|---|---|---|---|---|---|
| EEGNet | 0.5 s | 0.449 | 0.021 | 0.323 | 0.51 | 3.2 | n/a |
| EEGNet | 1 s | 0.520 | -0.004 | 0.323 | 0.66 | 6.1 | 10/10 |
| EEGNet | 2 s | 0.576 | 0.004 | 0.319 | 0.94 | 11.2 | 9/10 |
| EEGNet | 3 s | 0.603 | 0.125 | 0.301 | 1.22 | 16.4 | 9/10 |
| CNN-LSTM | 0.5 s | 0.499 | 0.039 | 0.317 | 0.53 | see note | n/a |
| CNN-LSTM | 1 s | 0.532 | 0.011 | 0.316 | 0.69 | 5.9 | 9/10 |
| CNN-LSTM | 2 s | 0.588 | 0.075 | 0.300 | 0.98 | 11.5 | 9/10 |
| CNN-LSTM | 3 s | 0.579 | 0.035 | 0.303 | 1.27 | 16.7 | 9/10 |

Note: the 0.5 s CNN-LSTM seeds ran at the same time, so their 10.7 s per epoch is not comparable. Epoch time grows about linearly with history.

- r rises with history in both models. EEGNet keeps rising to 3 s (2 s to 3 s: +0.027, 9/10 subjects up). CNN-LSTM stops at 2 s (2 s to 3 s: -0.009, 6/10 up).
- R² rises much less. It stays near zero up to 2 s for EEGNet, and only EEGNet at 3 s clearly improves on 0.5 s (+0.104, 7/10 up).
- CNN-LSTM gets less stable with history. The SD of r across subjects grows from 0.091 to 0.178, and seed means at 3 s span 0.545 to 0.609. P8 falls from r 0.420 to 0.193, and P12 reaches R² -0.88.
- No history length reaches the cue-locked template (r 0.803, R² 0.524, NRMSE 0.216).

Not yet known: how much of the gain is trial-specific. A longer window contains the LEDOn cue for more samples, so it may only give the model a better estimate of the time since the cue. The residual metric tests this.

### EEG before the lift window

The main cache holds only each lift's window, concatenated, and neighbouring windows overlap, so it cannot serve a history that starts before the lift. `scripts/build_pre_lift_cache.py` cuts those samples from the continuous recording, on the same 100 Hz grid and with the same causal chain. On P1, 0.5 s windows equal the main cache exactly, and 9300 pre-lift samples equal the previous lift at the same raw indices.

Lifts at the first scored sample (10 test subjects). Flagged lifts are reported and stay in the scoring:

| History | Filter transient | Before series start | History leaves the lift window |
|---|---|---|---|
| 0.5 s | 4.7% | 0% | 0% |
| 1 s | 4.8% | 0% | 0.1% |
| 2 s | 4.8% | 1.4% | 67.1% |
| 3 s | 5.1% | 3.0% | 99.1% |

Samples before the series start hold the earliest value on the lift's grid.

### Fixed lr

The fixed lr was chosen for this ablation only. The check on EEGNet at 3 s, seed 1, with the lr grid {1e-4, 3e-4, 1e-3} gave r 0.591 against 0.607 for fixed lr (-0.016, over the agreed 0.01 limit) and R² 0.090 against 0.097 (-0.008). The grid was worse, and the 5 folds that selected 1e-3 differ by a mean |r| of 0.022 from the initial random state alone. The user kept the fixed lr. It may still not be the best lr at 3 s: the grid selected a lower lr in 5 of 10 folds, and 9 of 30 EEGNet folds at 3 s reach their best validation at epoch 1.

## Residual metric

`residual_r`: per trial and axis, Pearson r between (prediction - T_cue) and (target - T_cue), aggregated as in D7. T_cue is the cue-locked template of the fold training subjects, which is known at test time. `residual_r_oracle` uses the template aligned on the kinematic onset of the test trial and is analysis only. Both are computed from saved predictions by `scripts/compute_residual_metrics.py` for 42 runs, and `scripts/summarize_runs.py --stage diagnostics` shows them. The template predictions reproduce exactly: both residuals are undefined (NaN) for the template they are measured against.

| Model | History | r | R² | residual_r (mean ± SD) | residual_r_oracle |
|---|---|---|---|---|---|
| Ridge | 0.5 s | 0.331 | -0.002 | 0.051 ± 0.152 | 0.042 |
| Kalman | 0.5 s | 0.466 | -3.084 | 0.181 ± 0.132 | 0.148 |
| EEGNet | 0.5 s | 0.449 | 0.021 | 0.086 ± 0.145 | 0.066 |
| EEGNet | 1 s | 0.520 | -0.004 | 0.116 ± 0.142 | 0.076 |
| EEGNet | 2 s | 0.576 | 0.004 | 0.148 ± 0.144 | 0.095 |
| EEGNet | 3 s | 0.603 | 0.125 | 0.149 ± 0.146 | 0.095 |
| CNN-LSTM | 0.5 s | 0.499 | 0.039 | 0.154 ± 0.138 | 0.120 |
| CNN-LSTM | 1 s | 0.532 | 0.011 | 0.182 ± 0.140 | 0.138 |
| CNN-LSTM | 2 s | 0.588 | 0.075 | 0.237 ± 0.179 | 0.175 |
| CNN-LSTM | 3 s | 0.579 | 0.035 | 0.237 ± 0.195 | 0.173 |
| Onset template | n/a | 0.754 | 0.406 | 0.021 ± 0.247 | -0.123 |
| Template (oracle timing) | n/a | 0.864 | 0.622 | 0.475 ± 0.093 | undefined |
| Cue template | n/a | 0.803 | 0.524 | undefined | -0.180 |

- EEG decoders leave most of the residual untracked. The best value is 0.237, and the subject SD is large.
- Longer history raises residual_r up to 2 s: EEGNet 0.5 s to 3 s +0.063 (8/10 subjects up), CNN-LSTM 0.5 s to 2 s +0.083 (8/10 up). From 2 s to 3 s nothing changes (EEGNet +0.001, CNN-LSTM 0.000).
- The oracle-aligned template reaches 0.475 because knowing the true onset explains much of the residual. The onset template gets 0.021, so its decoded onsets carry little of that timing.
- Kalman has a high residual_r but R² -3.08. Pearson r ignores amplitude, so this metric does not mean the prediction is accurate.

### Other-trial control

residual_r is not purely trial-specific. A model can correlate with the residual through what is shared across a subject's trials, such as a shape difference between that subject and the training subjects. `residual_r_other_trial` measures that part: each predicted residual is paired with the target residual of another trial of the same subject, on their shared bins of time since LED-on (at least 50). It uses 5 random trial cycles per subject, the same in every run (seed 20261106). `residual_r_specific` is residual_r minus this control.

| Model | History | residual_r | Other trial | Specific (mean ± SD) | Subjects above 0 |
|---|---|---|---|---|---|
| Ridge | 0.5 s | 0.051 | 0.005 | 0.045 ± 0.034 | 9/10 |
| Kalman | 0.5 s | 0.181 | 0.062 | 0.118 ± 0.046 | 10/10 |
| EEGNet | 0.5 s | 0.086 | 0.021 | 0.065 ± 0.044 | 9/10 |
| EEGNet | 1 s | 0.116 | 0.043 | 0.073 ± 0.044 | 9/10 |
| EEGNet | 2 s | 0.148 | 0.058 | 0.090 ± 0.048 | 10/10 |
| EEGNet | 3 s | 0.149 | 0.055 | 0.094 ± 0.058 | 10/10 |
| CNN-LSTM | 0.5 s | 0.154 | 0.061 | 0.093 ± 0.050 | 10/10 |
| CNN-LSTM | 1 s | 0.182 | 0.065 | 0.118 ± 0.058 | 10/10 |
| CNN-LSTM | 2 s | 0.237 | 0.102 | 0.134 ± 0.064 | 10/10 |
| CNN-LSTM | 3 s | 0.237 | 0.102 | 0.135 ± 0.073 | 10/10 |
| Onset template | n/a | 0.021 | -0.007 | 0.029 ± 0.037 | 8/10 |
| Template (oracle timing) | n/a | 0.475 | 0.281 | 0.195 ± 0.121 | 10/10 |

- Decoders hold a small but consistent trial-specific signal: 0.134 for CNN-LSTM at 2 s, above 0 in all 10 subjects.
- History helps this part only a little: EEGNet 0.5 s to 3 s +0.029 (7/10 subjects up), CNN-LSTM 0.5 s to 2 s +0.041 (8/10 up), against r gains of +0.154 and +0.089. These are different measures, so the sizes are not additive. From 2 s to 3 s the gain is +0.003 and 0.000.
- CNN-LSTM is above EEGNet at 0.5 s in all 10 subjects (+0.028), and CNN-LSTM at 2 s is above EEGNet at 3 s in 8/10 (+0.041), although EEGNet at 3 s has the higher r.

Not shown by this control: where the specific part comes from. It can be reaction-time timing, movement-related EEG, or artifacts, because all of them change from trial to trial. The specific value also compares different samples: residual_r uses the whole trial, and the control uses the bins two trials share.

## Within-subject vs LOSO

Diagnostic only (D6). Train S1-S5, validate S6, test S7-S9, per subject. EEGNet uses 3 s and CNN-LSTM 2 s, the best histories from the ablation, chosen on LOSO test scores. Deep models: 3 seeds, fixed lr 1e-3. Full tables: `output/diagnostics/within_subject/summary/`.

Confounds:
- S7-S9 are always weight series, while S1-S6 mix conditions.
- LOSO scores all nine series, so LOSO is also rescored on S7-S9 only. The table compares against that.
- A deep model trains on about 85k samples per epoch here, against 200k in LOSO, because the epoch size is capped at the training set size. Within-subject deep models therefore take fewer gradient steps per epoch.
- EEG normalization remains transductive (D9) and includes the test series, without labels.

| Model | r LOSO (all series) | r LOSO (S7-S9) | r within | Difference | Subjects up | R² LOSO (S7-S9) | R² within | NRMSE LOSO (S7-S9) | NRMSE within |
|---|---|---|---|---|---|---|---|---|---|
| Template (oracle timing) | 0.864 | 0.876 | 0.892 | +0.015 | 6/10 | 0.607 | 0.700 | 0.199 | 0.172 |
| Cue template | 0.803 | 0.840 | 0.864 | +0.025 | 6/10 | 0.537 | 0.648 | 0.219 | 0.189 |
| Ridge, 0.5 s | 0.331 | 0.276 | 0.435 | +0.159 | 9/10 | -0.068 | 0.024 | 0.344 | 0.317 |
| EEGNet, 3 s | 0.603 | 0.618 | 0.788 | +0.171 | 9/10 | 0.101 | 0.446 | 0.308 | 0.238 |
| CNN-LSTM, 2 s | 0.588 | 0.581 | 0.687 | +0.106 | 9/10 | -0.001 | 0.317 | 0.314 | 0.253 |

r difference per subject (within minus LOSO on S7-S9):

| Subject | Ridge | EEGNet | CNN-LSTM | Cue template |
|---|---|---|---|---|
| P1 | +0.371 | +0.261 | +0.197 | +0.163 |
| P2 | +0.233 | +0.196 | +0.023 | -0.003 |
| P4 | +0.313 | +0.172 | +0.122 | +0.136 |
| P5 | +0.109 | +0.188 | +0.058 | +0.015 |
| P7 | -0.179 | -0.100 | -0.372 | -0.060 |
| P8 | +0.199 | +0.466 | +0.538 | +0.065 |
| P9 | +0.023 | +0.020 | +0.019 | +0.001 |
| P10 | +0.201 | +0.094 | +0.070 | -0.024 |
| P11 | +0.090 | +0.161 | +0.142 | -0.062 |
| P12 | +0.231 | +0.247 | +0.263 | +0.015 |

- EEG decoders gain much more than the templates from training on the same subject. EEGNet: r +0.171 and R² +0.345, against +0.025 and +0.112 for the cue template. So part of the LOSO loss is subject transfer.
- The within-subject split does not beat the fair floor. EEGNet is below the cue template by 0.076 (above it in 1/10 subjects), and CNN-LSTM is lower still. A subject's own cue template reaches r 0.864.
- P7 gets worse in every model, although it is among the best LOSO subjects. P8 gains the most with deep models and P9 hardly changes.

Residual metrics. Each split subtracts its own template, so these do not share a reference. Within-subject residuals are measured from the subject's own mean shape.

| Model | residual_r LOSO | residual_r within | Specific LOSO | Specific within | Specific up |
|---|---|---|---|---|---|
| Ridge, 0.5 s | 0.051 | 0.190 | 0.045 | 0.040 | 4/10 |
| EEGNet, 3 s | 0.149 | 0.339 | 0.094 | 0.181 | 8/10 |
| CNN-LSTM, 2 s | 0.237 | 0.383 | 0.134 | 0.212 | 6/10 |

- The model order changes again: EEGNet has the higher r in all 10 subjects (0.788 against 0.687), and CNN-LSTM has the higher residual_r and specific part.
- The specific part about doubles for EEGNet and rises by 0.078 for CNN-LSTM, so subject-specific training helps track single trials, not only the mean shape. The test-series confound still applies.
- Ridge gains in r and residual_r, but its specific part does not change: the gain is shared across trials.

## Prefrontal-only control

Direct check for ocular contamination: only Fp1 and Fp2, same models and protocol, 0.5 s history (the history of the existing artifact-control tables). EEGNet has 3 seeds. Under CAR, the average is still computed over all 32 channels before the two are selected. Full tables: `output/diagnostics/artifact-controls/summary/{ridge,eegnet}_controls.md`. The other rows are the data-baselines runs and reproduce their earlier values. The ratio is control r / all-channel r under the same reference.

| Input | Ridge, CAR | Ridge, orig. ref | EEGNet, CAR | EEGNet, orig. ref |
|---|---|---|---|---|
| All 32 channels | 0.331 | 0.324 | 0.449 | 0.460 |
| Sensorimotor (11 ch) | 0.299 (0.90) | 0.284 (0.88) | 0.402 (0.90) | 0.401 (0.87) |
| Frontal (7 ch) | 0.184 (0.56) | 0.229 (0.71) | 0.324 (0.72) | 0.320 (0.70) |
| Prefrontal (Fp1, Fp2) | 0.149 (0.45) | 0.105 (0.32) | 0.261 (0.58) | 0.207 (0.45) |
| Temporal (4 ch) | 0.143 (0.43) | 0.225 (0.70) | 0.238 (0.53) | 0.291 (0.63) |

- Prefrontal channels stay below all-channel EEG: above it in at most 1 of 10 subjects, in every column.
- They are not near zero. With EEGNet, two channels reach 0.58 (CAR) and 0.45 (original reference) of all-channel r. So an ocular share cannot be ruled out, and this control cannot bound it from above.
- The CAR values are not pure Fp1 and Fp2, because the average mixes in the other channels. The original-reference values are lower, unlike frontal and temporal, which stay level or rise under that reference.
- r can come from any signal locked to the cue or movement, so a prefrontal r does not by itself show eye activity, and it does not separate eye activity from a frontal brain response.

Not covered: this control is at 0.5 s only, so it does not test whether the gain at 2 to 3 s comes from eye activity. The models were not rerun on prefrontal channels at those lengths, because the pre-lift cache supports all channels only.
