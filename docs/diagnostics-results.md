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
