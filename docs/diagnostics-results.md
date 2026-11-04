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
