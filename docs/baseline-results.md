# Baseline results

**Older protocol.** These runs use the 500 ms history and the floors and metrics before 2026-11-06. They stay valid under that protocol. Current protocol: history 2 s, cue-locked template as the main fair floor. See [protocol.md](protocol.md#runs-under-the-older-protocol) and [diagnostics-results.md](diagnostics-results.md).

Stage `data-baselines`, runs of 2026-10-29/30. Protocol: [protocol.md](protocol.md). Data facts: [dataset.md](dataset.md).

All numbers come from `output/data-baselines/summary/models.md`, written by `scripts/summarize_runs.py`. That file lists the run directory behind every row. Artifact-control tables: `output/data-baselines/artifact-controls/summary/{ridge,eegnet}_controls.md` (`scripts/compare_controls.py`).

## Setup in short

- Target: wrist 3D position, re-centered at kinematic onset, 100 Hz. Scored segment per lift: [onset − 0.5 s, `tHandStop`].
- EEG: causal band-pass (0.1–4 Hz for ridge and Kalman, 0.1–40 Hz for deep models), bad-channel interpolation, CAR, z-scored per subject. Causal history of 500 ms.
- Split: leave-one-subject-out over the 10 non-screening subjects (P1, P2, P4, P5, P7–P12). P3 and P6 are screening subjects and never a test fold.
- Metric: Pearson r per trial and axis, averaged over trials, then axes, then subjects. Reported as mean ± SD across the 10 test subjects. Pooled r over concatenated trials is inflated and given only as a secondary column.
- Deep models: 3 seeds each. Per-subject values are averaged over seeds; the SD of the aggregate across seeds is ≤ 0.008 for every model.

## Main table

| Model | Uses | r (mean ± SD) | R² | NRMSE | Params | Δr vs stereotypy |
|---|---|---|---|---|---|---|
| Template (mean trajectory vs time since onset) | oracle onset timing | 0.864 ± 0.042 | 0.622 | 0.193 | – | +0.025 |
| Stereotypy floor (another trial of the same subject) | oracle onset timing | 0.839 ± 0.059 | 0.488 | 0.211 | – | 0 |
| CNN-LSTM | EEG | **0.499 ± 0.091** | 0.039 | 0.317 | 10,843 | −0.340 |
| Kalman (EEG observations) | EEG + oracle onset timing | 0.466 ± 0.077 | −3.084 | 0.623 | – | −0.373 |
| ShallowConvNet | EEG | 0.454 ± 0.062 | 0.002 | 0.327 | 51,963 | −0.386 |
| EEGNet | EEG | 0.449 ± 0.075 | 0.021 | 0.323 | 1,555 | −0.390 |
| Ridge, 25 lags (0–480 ms) | EEG | 0.331 ± 0.068 | −0.002 | 0.330 | 2,403 | −0.508 |
| Kalman, prior only (no EEG) | oracle onset timing | 0.203 ± 0.093 | −1.626 | 0.530 | – | −0.636 |

"Oracle onset timing" means the model is told when the test lift's movement starts (the template and stereotypy floor are aligned on it; the Kalman filter starts at onset − 0.5 s). The EEG-only decoders get no timing information.

## Does EEG beat the template and stereotypy floor?

No. Every EEG decoder is below both floors in all 10 test subjects.

- The best EEG decoder (CNN-LSTM, r 0.499) is 0.365 below the template and 0.340 below the stereotypy floor.
- The floors are strong because reach-and-lift trajectories are highly stereotyped: once onset is known, the shape is almost fixed.
- The comparison is not equal: the floors know the onset time, the EEG decoders do not. Even so, the result says that the EEG decoders do not yet recover more than "when does the movement happen and roughly which shape".
- EEG decoders reach R² ≈ 0: their predictions follow the trajectory's shape (r ≈ 0.3–0.5) but not its per-trial amplitude and offset. Kalman has a strongly negative R² (NRMSE 0.623): its errors are larger than the trial's own variance.
- The Kalman filter shows that EEG adds information on top of a timing prior: r 0.203 without EEG, 0.466 with EEG (+0.263).

## How much comes from EMG, frontal, and temporal channels?

Each control uses the same model and protocol with only the input changed. The ratio is control r / all-channel r under the same reference.

| Input | Ridge, CAR | Ridge, orig. ref | EEGNet, CAR | EEGNet, orig. ref |
|---|---|---|---|---|
| All 32 channels | 0.331 | 0.324 | 0.449 | 0.460 |
| Sensorimotor (11 ch) | 0.299 (0.90) | 0.284 (0.88) | 0.402 (0.90) | 0.401 (0.87) |
| Frontal (7 ch) | 0.184 (0.56) | 0.229 (0.71) | 0.324 (0.72) | 0.320 (0.70) |
| Temporal (4 ch) | 0.143 (0.43) | 0.225 (0.70) | 0.238 (0.53) | 0.291 (0.63) |
| EMG envelope (5 ch, no EEG) | 0.870, R² 0.553 | | 0.914, R² 0.705 | |

- **Sensorimotor channels carry most of the decodable EEG information** (~90% of all-channel r with both models).
- **Frontal and temporal channels are weaker.** Under CAR they never beat all-channel EEG in more than 1 of 10 subjects. There is no sign that eye or jaw/neck artifacts drive the EEG decoders.
  - EEGNet extracts more from frontal channels than ridge (0.72 vs 0.56 of all-channel). Fz/F3/F4 lie near the motor areas, so this is not proof of ocular contamination. A direct check (e.g. Fp1/Fp2 only) is still open.
- **The original reference blurs the channel sets.** Frontal and temporal rise to ~0.70 under it. This fits an online reference near FCz, which mixes central activity into every channel. CAR separates the sets better and is used for the main runs.
- **EMG predicts the trajectory almost completely** (r 0.87–0.91), above the oracle-timed template with EEGNet. Muscle activity is a near-complete description of the movement; EEG decoders are far below it. If EMG leaked strongly into the EEG decoders, they would be expected to score much higher than they do.

## Which baseline is strongest?

- **Overall:** the oracle-timed template (r 0.864, R² 0.622). It is a floor, not a decoder.
- **Among EEG decoders:** CNN-LSTM (r 0.499, R² 0.039, NRMSE 0.317), best on every aggregate metric. It is the best EEG decoder in 4 of 10 subjects (Kalman in 3, ShallowConvNet in 2, EEGNet in 1).
- EEGNet and ShallowConvNet are tied (0.449 vs 0.454). ShallowConvNet had its best validation epoch at epoch 1 in 17 of 30 fold-seed runs, so it overfits fast. All three deep models beat ridge in every subject.
- Kalman matches the deep models on r but has a much worse R², and it uses oracle timing.

## Per-subject r

| Subject | Template | Stereotypy | Ridge | Kalman | EEGNet | ShallowConvNet | CNN-LSTM | EMG (EEGNet) |
|---|---|---|---|---|---|---|---|---|
| P1 | 0.812 | 0.895 | 0.357 | 0.580 | 0.501 | 0.462 | 0.571 | 0.947 |
| P2 | 0.818 | 0.784 | 0.191 | 0.331 | 0.343 | 0.359 | 0.475 | 0.877 |
| P4 | 0.848 | 0.880 | 0.344 | 0.514 | 0.449 | 0.428 | 0.542 | 0.946 |
| P5 | 0.913 | 0.883 | 0.321 | 0.561 | 0.433 | 0.457 | 0.462 | 0.893 |
| P7 | 0.798 | 0.703 | 0.455 | 0.488 | 0.625 | 0.572 | 0.676 | 0.893 |
| P8 | 0.890 | 0.877 | 0.270 | 0.431 | 0.409 | 0.509 | 0.420 | 0.939 |
| P9 | 0.882 | 0.855 | 0.365 | 0.468 | 0.446 | 0.403 | 0.436 | 0.881 |
| P10 | 0.911 | 0.816 | 0.353 | 0.486 | 0.441 | 0.504 | 0.581 | 0.939 |
| P11 | 0.883 | 0.832 | 0.348 | 0.404 | 0.389 | 0.406 | 0.377 | 0.907 |
| P12 | 0.885 | 0.866 | 0.308 | 0.397 | 0.454 | 0.436 | 0.451 | 0.915 |

Source: `output/data-baselines/summary/per_subject_r.csv` (deep models averaged over 3 seeds).

## Implications for the SSL stage

- **Beat the right numbers.** The supervised targets to beat are CNN-LSTM (r 0.499) and EEGNet (r 0.449, 1,555 params) under the same LOSO protocol and the same metric.
- **Report against the floors.** Any SSL result should also be shown next to the template and stereotypy floors. A decoder that only finds movement timing will look decent on r and still sit far below them.
- **R² is the weak spot.** No EEG decoder gets the per-trial amplitude right. Report R² and NRMSE next to r, and check whether a representation improves amplitude, not only correlation.
- **Physiology guidance has a basis.** Sensorimotor channels hold ~90% of the decodable information; frontal and temporal channels add little. This supports priors or augmentations built around the sensorimotor montage.
- **EMG is an upper reference and a possible teacher, not an input.** EMG reaches r ~0.9. It can serve as a pretext target or distillation signal during pretraining, but it must not be fed to the EEG decoder at test time. Any EEG model that approaches EMG-level performance needs the artifact controls rerun.
- **Cross-subject transfer is the hard part.** Subject SD (0.06–0.09) is ten times the seed SD. The within-subject diagnostic split (train S1–S5, val S6, test S7–S9) has not been run yet; it would show how much of the gap is subject transfer.

## Open items

- Within-subject diagnostic split: not run.
- Prefrontal-only (Fp1/Fp2) control as a direct ocular-artifact check: not run.
- Velocity target (D1 secondary): not run.
