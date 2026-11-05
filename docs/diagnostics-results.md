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

Limit: residual_r is not purely trial-specific. A model can correlate with the residual through what is shared across a subject's trials, such as a shape difference between that subject and the training subjects.
