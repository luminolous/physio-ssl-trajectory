# Diagnostics results

Stage `diagnostics`. Protocol: [protocol.md](protocol.md). Baselines: [baseline-results.md](baseline-results.md).

## Answers

All values use per-trial Pearson r (D7), LOSO over 10 test subjects unless stated, with deep models averaged over 3 seeds. The sections below hold the detail, and [Run directories](#run-directories) lists the source of each row.

**The main finding:** on average, no EEG decoder beats the fair floor. The best LOSO result is EEGNet at 3 s with r 0.603, against r 0.803 for a template that uses no EEG. P7 is the only subject above the floor, with both EEGNet at 3 s and CNN-LSTM at 2 s. Decoders do recover a small trial-specific signal, and training on the same subject increases it.

### What is a fair, non-oracle floor?

| Floor | Uses | r | R² | NRMSE |
|---|---|---|---|---|
| Cue-locked template | LED-on only, no EEG | 0.803 | 0.524 | 0.216 |
| Onset decoder + template | EEG up to t, causally | 0.754 | 0.406 | 0.241 |
| Oracle template (upper reference) | true kinematic onset | 0.864 | 0.622 | 0.193 |

The cue-locked template is the simplest fair floor, and it beats CNN-LSTM at 0.5 s in 9 of 10 subjects. Removing oracle timing costs only 0.061 in r. Onset decoding does not help: it falls below the cue-locked template in 9 of 10 subjects.

All scores use a segment derived from each lift's kinematics, for every model. So every score is measured over a window known to contain the movement.

### Is the 500 ms history the bottleneck?

Partly for r, but it is not the main limit.

| Model | History | r | R² | residual_r | Specific |
|---|---|---|---|---|---|
| EEGNet | 0.5 s | 0.449 | 0.021 | 0.086 | 0.065 |
| EEGNet | 3 s | 0.603 | 0.125 | 0.149 | 0.094 |
| CNN-LSTM | 0.5 s | 0.499 | 0.039 | 0.154 | 0.093 |
| CNN-LSTM | 2 s | 0.588 | 0.075 | 0.237 | 0.134 |

- r rises with history, but no length reaches the floor. CNN-LSTM stops improving at 2 s.
- The trial-specific part gains far less (+0.029 and +0.041) and stops at 2 s for both models.
- **Almost all of the r gain from a longer history is available from Fp1 and Fp2 alone: +0.152 against +0.154 on all 32 channels (EEGNet, CAR, 0.5 s to 3 s). A rise in r from history must not be claimed as motor information.** It fits cue-locked timing or eye activity, and these are not separated.

### How much trial-specific information do decoders recover?

A small amount, and it is consistent across subjects. The best specific part is 0.134 (CNN-LSTM at 2 s), above 0 in all 10 subjects. It is residual_r 0.237 minus 0.102 from pairing predictions with other trials. Its source is not separated: it can be reaction-time timing, movement-related EEG, or artifacts. Prefrontal channels reach about half of the all-channel value at 3 s (0.048 against 0.094).

### How much of the gap is cross-subject transfer?

A large part, on this diagnostic. On S7-S9, the gap to the cue-locked template falls when training on the same subject:

| Model | Gap, LOSO | Gap, within subject |
|---|---|---|
| EEGNet, 3 s | 0.222 | 0.076 |
| CNN-LSTM, 2 s | 0.259 | 0.177 |

The specific part also rises, from 0.094 to 0.181 for EEGNet and from 0.134 to 0.212 for CNN-LSTM, though each split uses its own template. EEGNet still stays below the subject's own cue-locked template in 9 of 10 subjects. S7-S9 are always weight series, so this is a diagnostic only.

### Are there bugs or under-tuning?

No bug was found, and tuning did not help.

- **Kalman:** R² -3.08 is model misspecification, not a bug. Its observation noise is strongly autocorrelated while the filter assumes white noise, and predictions swing about twice as wide as the target. See `output/diagnostics/fixes/kalman/explanation.md`.
- **Tuning (lr 1e-4, 3e-4, 1e-3):** r moved from 0.449 to 0.456 for EEGNet, 0.454 to 0.451 for ShallowConvNet, and 0.499 to 0.478 for CNN-LSTM. The selection gain is no larger than the noise from initial weights. See `output/diagnostics/tuning/summary/summary.md`.
- **Not excluded:** lr 1e-3 is the upper edge of the grid, and EEGNet chose it in all 10 folds at 0.5 s, so higher values are untested. At 3 s the grid picked lower lr in 5 of 10 folds without a test-score gain. Architecture was not searched.
- **Data access:** the main cache cannot serve a history that starts before a lift. The ablation uses pre-lift EEG from the continuous recording instead. Earlier 0.5 s runs are unaffected: their windows stay inside the lift.

### Protocol decisions from these results

Decided on 2026-11-06, recorded in [protocol.md](protocol.md): history 2 s at 100 Hz (D3); `residual_r` and `residual_r_specific` as secondary metrics, with `template_cue` as the main fair floor (D7); fixed budget for supervised deep models (D11); claim rules (D12). Pre-lift EEG exists for broadband CAR only. The prefrontal control was run at 0.5 and 3 s, not at the chosen 2 s.

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

### Prefrontal at 3 s

Does the gain from a longer history need the whole scalp? EEGNet on Fp1 and Fp2, CAR, 3 s, 3 seeds, against all channels on the same seeds. Table: `output/diagnostics/history/summary/prefrontal.csv`.

| Input | r at 0.5 s | r at 3 s | Gain | R² at 3 s | residual_r at 3 s | Specific at 3 s |
|---|---|---|---|---|---|---|
| Prefrontal (Fp1, Fp2) | 0.261 | 0.412 | +0.152 | 0.003 | 0.076 | 0.048 |
| All 32 channels | 0.449 | 0.603 | +0.154 | 0.125 | 0.149 | 0.094 |
| Ratio, prefrontal / all | 0.58 | 0.68 | | | 0.51 | 0.51 |

- **Almost all of the r gain from a longer history is available from Fp1 and Fp2 alone: +0.152 against +0.154 on all 32 channels. A rise in r from history must not be claimed as motor information.** The prefrontal gain is larger in 5 of 10 subjects and positive in all 10.
- This fits a signal locked to the cue, which gives the model the time since the cue. It also fits eye activity. The control does not separate them.
- All channels still add something: prefrontal is below all-channel r in all 10 subjects at 3 s, and above it on the specific part in 1 of 10. The specific part at 3 s is about half of the all-channel value.
- So raw r overstates what a longer history adds. Use it together with residual_r_specific when judging a history length.

Limits: under CAR, Fp1 and Fp2 contain the average of all 32 channels, so this is not a pure prefrontal signal. The original reference was not run at 3 s because the pre-lift cache holds CAR only. There is no 0.5 s specific value for prefrontal, because those runs did not save predictions.

## Pre-design checks

Added on 2026-11-06 to give the SSL architecture an evidence base. Protocol as above (LOSO over 10 subjects, 2 s history, CAR, causal filters, fixed budget D11). The main readout is `residual_r_specific`, then r and R². For the screening, alignment, and channel-importance runs, inner-validation scores (the fold's validation subjects, never its test subject) are the selection readout and LOSO test scores are descriptive; inner-validation folds share subjects, so their p values are descriptive too. Formal selection stays in Exp 1 on P3 and P6. Outputs are under `output/diagnostics/pre-design/`.

These checks inform the design. They do not decide it; that is discussed separately.

| Check | Main result | Informs |
|---|---|---|
| Neurophysiology | The 0.1-2 Hz band is dominated by frontal, eye-like activity; no pre-onset MRCP; ERD bilateral; only beta is lateralized (right hand) | Target-view priors: band, lateralization |
| Residual formulation | Predicting the residual over the cue template reaches the template, not beyond it | Residual head or direct prediction |
| Band and channel screening | Broadband, broadband without Fp1/Fp2, and MRCP tie; envelopes carry no trial-specific signal; contra > ipsi is small | Target-view priors: band, channel set, lateralization |
| Channel importance | Cz and the contralateral group matter, but occipital and prefrontal channels matter as much in deep models | Target-view priors: channel set, lateralization |
| Lag profile | With true timing, trial-specific information peaks near zero lag, not with EEG leading | Target-view priors: band and timing |
| Euclidean alignment | No gain; test scores slightly worse | Alignment |
| Unlabeled data inventory | 11.7 h of EEG in total, 8.5 h of it in lift windows | Pretraining data |

### Neurophysiology sanity check

MRCP cache variant (0.1-2 Hz), event-related potentials and ERD maps, 10 LOSO subjects. Run: `output/diagnostics/pre-design/neurophysiology/20261106-132757/`.

- Eye-like activity dominates 0.1-2 Hz. Fp1 and Fp2 fall to about -275 and -210 uV near 0.25 s after onset, against about -30 uV at Cz and C3. The fall starts before LED-on. There is no EOG channel, so the cause is likely but not proven.
- No self-paced MRCP before onset: Cz +2.4 uV in the 0.5 s before onset (5/10 negative, p 0.77).
- The C3-C4 MRCP difference (-17.5 uV, 9/10) is not motor evidence. It is part of a whole-scalp left-right gradient, largest at F7-F8 (-67 uV), and follows F7-F8 across subjects (Spearman 0.77-0.89).
- ERD is present and bilateral: 0 to 1 s after onset, mu -25 % (left) and -27 % (right), beta -36 % and -32 %, all 10/10.
- Only beta supports right-hand movement: before onset, the left-minus-right beta cluster is -6.5 points (10/10, p 0.002). Mu is not lateralized.
- The causal filter delays 0.5, 1, and 2 Hz components by 224, 172, and 135 ms.

Informs target-view priors. A gain from the low band cannot be read as motor without an ocular control. Beta is the cleanest lateralized signal, but it is small.

### Residual formulation

The model predicts the target minus the cue-locked template T_cue (fitted on the fold's training subjects), and T_cue is added back. CNN-LSTM and EEGNet, 2 s, 3 seeds. Runs: `output/diagnostics/pre-design/residual/`; comparison: `output/diagnostics/pre-design/residual/summary/`.

| Model | Form | r | R² | Specific |
|---|---|---|---|---|
| CNN-LSTM | Direct | 0.588 | 0.075 | 0.134 |
| CNN-LSTM | Residual | 0.813 | 0.510 | 0.192 |
| EEGNet | Direct | 0.576 | 0.004 | 0.090 |
| EEGNet | Residual | 0.812 | 0.497 | 0.161 |
| `template_cue` | No EEG | 0.803 | 0.524 | |

- Against `template_cue`, the residual models gain nothing reliable: CNN-LSTM r +0.010 (5/10, p 0.23), R² -0.014; EEGNet r +0.010 (6/10), R² -0.028 (2/10, p 0.064).
- Nearly all of the gain over direct prediction comes from T_cue. The rise in specific does not prove more information: direct predictions carry a large shape error that the residual form removes.
- Early stopping picks a very early epoch (median 1 for CNN-LSTM, 2 for EEGNet), so the residual is learned early under the fixed budget.
- P7 is the exception: direct r 0.796, residual 0.683.

Informs the choice between a residual head and direct prediction.

### Band and channel screening

CNN-LSTM, 2 s. 14 configurations at seed 1; 11 of them at seeds 2-3 (the envelope configurations stayed at seed 1). New cache variants: mu and beta band-power envelopes (8-12 and 13-30 Hz, rectified, causal 5 Hz low-pass). Comparison: `output/diagnostics/pre-design/screening/summary/20261107-160546/`.

| Band × channels | Val specific | Val r | Test specific | Test r | Test R² |
|---|---|---|---|---|---|
| Broadband × all | 0.161 | 0.644 | 0.134 | 0.588 | 0.075 |
| Broadband × no Fp1/Fp2 | 0.155 | 0.635 | 0.127 | 0.570 | 0.037 |
| MRCP × all | 0.153 | 0.621 | 0.129 | 0.554 | 0.104 |
| MRCP + envelopes × all | 0.142 | 0.676 | 0.135 | 0.646 | 0.202 |
| Broadband × sensorimotor | 0.081 | 0.574 | 0.073 | 0.526 | -0.013 |
| MRCP × contralateral | 0.077 | 0.477 | 0.072 | 0.441 | 0.051 |
| Broadband × prefrontal | 0.071 | 0.414 | 0.062 | 0.362 | -0.028 |
| Broadband × contralateral | 0.070 | 0.537 | 0.064 | 0.489 | -0.025 |
| MRCP × ipsilateral | 0.065 | 0.464 | 0.061 | 0.419 | 0.021 |
| MRCP × prefrontal | 0.065 | 0.414 | 0.059 | 0.357 | -0.006 |
| Broadband × ipsilateral | 0.065 | 0.509 | 0.057 | 0.456 | -0.058 |
| Envelopes × all (seed 1) | 0.022 | 0.500 | 0.018 | 0.510 | 0.044 |

- Broadband, broadband without Fp1/Fp2, and MRCP do not differ on specific (differences below 0.01, no consistent direction).
- Adding envelopes to MRCP raises test r (+0.058, 8/10, p 0.020) and R² (+0.126) but lowers specific on inner validation (-0.019 against broadband, 1/10 folds up). The gain is not trial-specific.
- Envelopes alone carry almost no trial-specific signal.
- Contralateral minus ipsilateral, specific: MRCP +0.012 on inner validation (9/10, p 0.006) and +0.011 on test (9/10, p 0.014); broadband +0.006 (8/10, p 0.16). Small, and Cz is in both sets.
- Fp1 and Fp2 alone match the contralateral set on specific with broadband (0.071 vs 0.070). Dropping them from all channels costs 0.005 (not significant).
- No configuration beats `template_cue` on test r or R².
- Channel and parameter counts differ across configurations, so channel-set contrasts are confounded with capacity.

Informs target-view priors (band, channel set, lateralization). Candidates for Exp 1: broadband × all, broadband without Fp1/Fp2, MRCP × all.

### Channel importance

Occlusion on test data: each channel, and each group (prefrontal, contralateral, ipsilateral, occipital), set to its normalizer mean, and the drop in r and specific measured. CNN-LSTM broadband, CNN-LSTM MRCP, and EEGNet broadband (2 s), plus lagged ridge (0.1-4 Hz, 500 ms) with Haufe activation patterns. Retrained at seed 1 because the earlier runs saved no weights. Summary and topographies: `output/diagnostics/pre-design/channel-importance/summary/20261107-202517/`.

| Group occluded | CNN-LSTM broadband | CNN-LSTM MRCP | EEGNet | Ridge |
|---|---|---|---|---|
| Contralateral | 0.050 | 0.045 | 0.029 | 0.026 |
| Ipsilateral | 0.028 | 0.013 | 0.005 | 0.007 |
| Occipital | 0.065 | 0.047 | 0.041 | 0.006 |
| Prefrontal | 0.044 | 0.022 | 0.010 | 0.003 |

Values are drops in test `residual_r_specific`.

- Cz is the top or second single channel in every model. Next come F7, Fp1/Fp2, and O1/Oz in the deep models. C3 alone barely matters.
- The contralateral group beats its mirror in 10/10 subjects in every model (specific +0.020 to +0.031, p 0.002). Both groups hold Cz, so the difference comes from FC1, FC5, CP1, CP5, and C3.
- In the deep models the occipital group matters as much as, or more than, the contralateral group. Ridge relies least on occipital and prefrontal channels.
- Ridge Haufe patterns peak at the frontal midline (Fz) with temporal minima; no focal C3 pattern.
- So importance does not concentrate over contralateral sensorimotor cortex. A motor contribution is present, but ocular, head-movement, or visual (LED, gaze) activity may carry much of the rest; these data cannot separate them.

Limits: one seed; CAR spreads each channel into the others; occlusion moves inputs off the training distribution; groups have 2 to 6 channels.

Informs target-view priors (channel set, lateralization).

### Lag profile

Analysis only, outside the causal protocol; these are not decoders. Ridge on 0.1-2 Hz EEG with a single lag from -500 ms (EEG after the kinematic sample) to +1000 ms (EEG before it), every 20 ms, LOSO, for all channels, sensorimotor, and prefrontal. Both the causal MRCP cache and a zero-phase (forward-backward) variant. Run: `output/diagnostics/pre-design/lag-profile/20261107-203436/`.

- Zero phase (true timing), all channels: specific peaks at -20 ms (0.058), with a broad plateau from about -200 to +100 ms; per-subject peaks range from -500 to +620 ms (median -70 ms). EEG after versus before the sample: +0.009 (6/10, p 0.23). Sensorimotor has the same shape (peak -40 ms, 0.046).
- r is highest with EEG after the kinematic sample (-320 ms, 0.503) and falls as the EEG moves earlier (0.287 at +1000 ms).
- Causal filter: the same curves, shifted later by about 100 ms. EEG before then beats EEG after, but that is the filter delay, not a neural lead.
- Prefrontal: specific stays at most 0.016 and peaks with EEG about 340-400 ms before the sample. That fits eye movements preceding the hand, but is small and not proven.
- So there is no clear neural lead. Trial-specific information peaks near zero lag and is at least as large with EEG after the movement, which points to feedback, movement-related artifacts, or concurrent potentials rather than to motor planning. Slow EEG and a slow position target limit timing resolution to a few hundred ms.

Informs target-view priors (band and timing).

### Euclidean alignment

Each subject's EEG is whitened by the mean of its per-lift spatial covariances, without labels (transductive for the test subject, like D9), before normalization. After CAR the covariance has rank 31 (30 for P2), so the inverse square root is taken on the non-null subspace. CNN-LSTM broadband, all channels, 2 s, 3 seeds, against the same configuration without alignment. Comparison: `output/diagnostics/pre-design/euclidean-alignment/summary/20261108-092151/`.

| Split | Metric | Aligned | Not aligned | Difference |
|---|---|---|---|---|
| Inner validation | r | 0.649 | 0.644 | +0.005 (6/10, p 0.43) |
| Inner validation | Specific | 0.162 | 0.161 | +0.001 (3/10, p 0.77) |
| Test | r | 0.561 | 0.588 | -0.026 (2/10, p 0.084) |
| Test | R² | -0.061 | 0.075 | -0.136 (2/10, p 0.037) |
| Test | NRMSE | 0.317 | 0.300 | worse in 9/10 (p 0.020) |
| Test | Specific | 0.134 | 0.134 | -0.001 |

Alignment adds no trial-specific information and slightly worsens the scale and shape of test predictions (P12's R² falls to -1.56). Informs the alignment decision.

### Unlabeled data inventory

Hours from the raw HS files; 2 s windows on the 100 Hz grid, inside contiguous segments only. Run: `output/diagnostics/pre-design/data-inventory/20261108-093500/`.

| Source | Hours (12 subjects) |
|---|---|
| Lift windows (merged) | 8.45 |
| Between lifts (HS S1-S9) | 2.42 |
| ST recording | 0.89 |
| Total | 11.74 |

- Per subject: 0.87 to 1.12 h. Between-lift segments are short (median 5.6 s, max 13.3 s).
- 2 s windows over all subjects: 363k at 0.1 s stride, 146k at 0.25 s, 74k at 0.5 s, 38k at 1 s, 19.5k at 2 s.
- Per LOSO fold: about 10.7 h from the other 11 subjects (including P3 and P6), and 0.9 to 1.1 h from the test subject (usable without labels for transductive pretraining).
- Unlabeled EEG beyond the lift windows adds about 3.3 h, all from the same task sessions. The windows overlap heavily at small strides.
- The first seconds of each recording, where causal filters settle, are included.

Informs the pretraining-data decision.

### Open points for the design discussion

- No ocular-cleaned variant has been run. Several checks (neurophysiology, channel importance, lag profile) point to eye-like or other non-motor activity, and none can bound its share.
- Channel importance used one seed; the screening envelope configurations used one seed.

## Run directories

Each row is one configuration, with several run IDs being seeds. r is the mean over subjects, seeds averaged first. Source tables: `output/diagnostics/summary/models.md`, `output/diagnostics/history/summary/`, `output/diagnostics/within_subject/summary/`, `output/diagnostics/artifact-controls/summary/`. Residual metrics: `output/diagnostics/residual/20261106-081026/residual.csv`.

| Folder | Run IDs | r |
|---|---|---|
| `output/data-baselines/baselines/template/` | `20261029-184451` | 0.864 |
| `output/diagnostics/baselines/template_cue/` | `20261104-234509` | 0.803 |
| `output/diagnostics/baselines/onset_template/` | `20261104-235524` | 0.754 |
| `output/data-baselines/baselines/ridge_lag/` | `20261030-034531` | 0.331 |
| `output/data-baselines/baselines/kalman/` | `20261030-035949` | 0.466 |
| `output/data-baselines/baselines/eegnet/` | `20261030-042902_seed1`, `20261030-043700_seed2`, `20261030-044818_seed3` | 0.449 |
| `output/data-baselines/baselines/shallowconvnet/` | `20261030-052039_seed1`, `20261030-052040_seed2`, `20261030-052044_seed3` | 0.454 |
| `output/data-baselines/baselines/cnn_lstm/` | `20261030-054018_seed1`, `20261030-054023_seed2`, `20261030-054029_seed3` | 0.499 |
| `output/diagnostics/tuning/eegnet/` | `20261104-154725_seed1`, `20261104-163014_seed2`, `20261104-163643_seed3` | 0.456 |
| `output/diagnostics/tuning/shallowconvnet/` | `20261104-164437_seed1`, `20261104-170247_seed2`, `20261104-171058_seed3` | 0.451 |
| `output/diagnostics/tuning/cnn_lstm/` | `20261104-171617_seed1`, `20261104-174108_seed2`, `20261104-174855_seed3` | 0.478 |
| `output/diagnostics/history/eegnet/history_100/` | `20261105-010536_seed1`, `20261105-134134_seed2`, `20261105-144044_seed3` | 0.520 |
| `output/diagnostics/history/eegnet/history_200/` | `20261105-011656_seed1`, `20261105-135359_seed2`, `20261105-145450_seed3` | 0.576 |
| `output/diagnostics/history/eegnet/history_300/` | `20261105-013341_seed1`, `20261105-141526_seed2`, `20261105-151425_seed3` | 0.603 |
| `output/diagnostics/history/eegnet_lr_grid/history_300/` | `20261105-093231_seed1` | 0.591 |
| `output/diagnostics/history/cnn_lstm/history_100/` | `20261105-205741_seed1`, `20261105-222121_seed2`, `20261105-233338_seed3` | 0.532 |
| `output/diagnostics/history/cnn_lstm/history_200/` | `20261105-210722_seed1`, `20261105-223130_seed2`, `20261105-234321_seed3` | 0.588 |
| `output/diagnostics/history/cnn_lstm/history_300/` | `20261105-213451_seed1`, `20261105-225723_seed2`, `20261106-000701_seed3` | 0.579 |
| `output/diagnostics/within_subject/template/` | `20261106-053055` | 0.892 |
| `output/diagnostics/within_subject/template_cue/` | `20261106-053104` | 0.864 |
| `output/diagnostics/within_subject/ridge_lag/` | `20261106-053115` | 0.435 |
| `output/diagnostics/within_subject/eegnet/history_300/` | `20261106-053214_seed1`, `20261106-055144_seed2`, `20261106-060914_seed3` | 0.788 |
| `output/diagnostics/within_subject/cnn_lstm/history_200/` | `20261106-063243_seed1`, `20261106-064509_seed2`, `20261106-065642_seed3` | 0.687 |
| `output/diagnostics/artifact-controls/eegnet_prefrontal_car/` | `20261106-072319_seed1`, `20261106-072956_seed2`, `20261106-073453_seed3` | 0.261 |
| `output/diagnostics/artifact-controls/eegnet_prefrontal_car/history_300/` | `20261106-075555_seed1`, `20261106-080048_seed2`, `20261106-080514_seed3` | 0.412 |
| `output/diagnostics/artifact-controls/eegnet_prefrontal_orig/` | `20261106-073947_seed1`, `20261106-074335_seed2`, `20261106-074634_seed3` | 0.207 |
| `output/diagnostics/artifact-controls/ridge_prefrontal_car/` | `20261106-071752` | 0.149 |
| `output/diagnostics/artifact-controls/ridge_prefrontal_orig/` | `20261106-072013` | 0.105 |

### Pre-design runs

r is the mean over test subjects, averaged over seeds. Analysis runs without a decoder (neurophysiology, lag profile, data inventory, cache builds) are listed in their sections.

| Folder | Run IDs | r |
|---|---|---|
| `output/diagnostics/pre-design/channel-importance/cnn_lstm_broadband/` | `20261107-162507_seed1` | 0.601 |
| `output/diagnostics/pre-design/channel-importance/cnn_lstm_mrcp/` | `20261107-173404_seed1` | 0.550 |
| `output/diagnostics/pre-design/channel-importance/eegnet_broadband/` | `20261107-184217_seed1` | 0.568 |
| `output/diagnostics/pre-design/channel-importance/ridge_lag/` | `20261107-195309_seed1` | 0.331 |
| `output/diagnostics/pre-design/euclidean-alignment/cnn_lstm_ea/` | `20261108-072521_seed1`, `20261108-072528_seed2`, `20261108-072542_seed3` | 0.561 |
| `output/diagnostics/pre-design/residual/cnn_lstm/` | `20261106-155151_seed1`, `20261106-163003_seed2`, `20261106-170433_seed3` | 0.813 |
| `output/diagnostics/pre-design/residual/eegnet/` | `20261106-160658_seed1`, `20261106-164552_seed2`, `20261106-172022_seed3` | 0.812 |
| `output/diagnostics/pre-design/screening/broadband__all/` | `20261106-180430_seed1`, `20261107-035149_seed2`, `20261107-105612_seed3` | 0.588 |
| `output/diagnostics/pre-design/screening/broadband__contralateral/` | `20261106-183341_seed1`, `20261107-064739_seed2`, `20261107-134251_seed3` | 0.489 |
| `output/diagnostics/pre-design/screening/broadband__ipsilateral/` | `20261106-184019_seed1`, `20261107-065636_seed2`, `20261107-135448_seed3` | 0.456 |
| `output/diagnostics/pre-design/screening/broadband__no_prefrontal/` | `20261106-184624_seed1`, `20261107-062236_seed2`, `20261107-131901_seed3` | 0.570 |
| `output/diagnostics/pre-design/screening/broadband__prefrontal/` | `20261106-191020_seed1`, `20261107-074411_seed2`, `20261107-144457_seed3` | 0.362 |
| `output/diagnostics/pre-design/screening/broadband__sensorimotor/` | `20261106-191546_seed1`, `20261107-073345_seed2`, `20261107-142924_seed3` | 0.526 |
| `output/diagnostics/pre-design/screening/env__all/` | `20261106-192718_seed1` | 0.510 |
| `output/diagnostics/pre-design/screening/env__contralateral/` | `20261106-195940_seed1` | 0.427 |
| `output/diagnostics/pre-design/screening/env__ipsilateral/` | `20261106-200752_seed1` | 0.414 |
| `output/diagnostics/pre-design/screening/mrcp__all/` | `20261106-213001_seed1`, `20261107-042453_seed2`, `20261107-112411_seed3` | 0.554 |
| `output/diagnostics/pre-design/screening/mrcp__contralateral/` | `20261106-220207_seed1`, `20261107-070759_seed2`, `20261107-140108_seed3` | 0.441 |
| `output/diagnostics/pre-design/screening/mrcp__ipsilateral/` | `20261106-221119_seed1`, `20261107-072458_seed2`, `20261107-141704_seed3` | 0.419 |
| `output/diagnostics/pre-design/screening/mrcp__prefrontal/` | `20261106-221906_seed1`, `20261107-075025_seed2`, `20261107-145745_seed3` | 0.357 |
| `output/diagnostics/pre-design/screening/mrcp_env__all/` | `20261106-201552_seed1`, `20261107-050320_seed2`, `20261107-120229_seed3` | 0.646 |
