# Experimental protocol

Locked on 2026-10-29. Changes are recorded with a date and reason at the bottom. Runs made under an older protocol are marked as such. Data facts behind these choices: [dataset.md](dataset.md).

## Target (D1)

- **Primary:** wrist (position sensor 4), 3D position (x, y, z) in the lab frame.
  - Re-centered per lift: the wrist position at movement onset is subtracted.
- **Secondary:** wrist 3D velocity, from the smoothed position.

## Trial segment and onset (D2)

- **Movement onset** is derived from kinematics, not taken from the labeled `tHandStart`.
  - Smooth the wrist position with a zero-phase low-pass: 10 Hz, order 2.
  - Find the peak wrist speed between LED-on and first digit touch.
  - Onset is the last sample before that peak where speed drops below 5% of the peak.
- **Why:** `tHandStart` is wrong in some lifts. The hand had already reached the object ~1 s earlier.
  - Full cache: 113 of 3528 lifts (3.2%). P7 38, P11 23, P2 18, P10 10, P12 10, P6 8, P5 4, P4 1, P9 1, P1/P3/P8 0.
  - In the other lifts, the kinematic onset agrees with `tHandStart` to a median of 12–44 ms.
  - Validation plots: `output/data-baselines/data-inspection/onset-validation/`.
- Lifts with |onset − `tHandStart`| > 0.3 s are flagged `onset_mismatch` and kept.
- **Segment** scored per lift: [onset − 0.5 s, `tHandStop`].
- Per-sample phase labels are stored: pre, reach, load, hold, replace, release.

## Sampling and decoder window (D3)

- EEG and targets at **100 Hz**: anti-alias filter, then decimate 500 Hz by 5.
- **Causal EEG history of 500 ms** (50 samples) before each target sample.
- Ridge uses lags every 20 ms (26 lags). Deep models get the full 50-sample window.

## Filtering (D4)

- **EEG:** causal Butterworth band-pass (order 2, `sosfilt`).
  - Applied to the continuous HS recording of each series. WS windows are then cut using the verified HS↔WS sample mapping.
  - Bands: **0.1–4 Hz** for linear models, **0.1–40 Hz** for deep models.
- **Start-up transient:**
  - The filter is initialized with `sosfilt_zi`, scaled by the first sample.
  - It still needs ~13.4–13.9 s to settle within 1% of steady state (measured on P1 S1 and P2 S2).
  - Lifts whose first used EEG sample falls inside the measured settle time are flagged `filter_transient`. This is roughly lifts 1–3 of each series, ~4.7% of lifts.
  - The cache builder measures the settle time per series.
  - Flagged lifts are kept. Excluding them is a sensitivity analysis.
- **Causal delay:** causal filtering delays the EEG.
  - 0.1–4 Hz: ~74–150 ms at 0.5–2 Hz.
  - 0.1–40 Hz: ~11–28 ms at 1–2 Hz.
  - The 500 ms history covers this. Read decoder lag profiles as EEG lag plus filter delay.
  - The EEG anti-alias filter before decimation is causal, order 4, at 40 Hz (~10 ms delay).
- **Kinematics:** smoothed zero-phase (`sosfiltfilt`). Targets are not decoder inputs, so this does not leak into features.
- **EMG** (artifact control only): rectified, causal low-pass, downsampled to 100 Hz.

## Referencing and channels (D5)

- Bad channels are detected per subject, without labels, on the continuous recordings.
  - Rule: high-passed MAD below 0.1× or above 5× the median channel in most series. Example: P2 Fz.
  - They are interpolated with spherical splines and logged in the cache manifest.
- Then **common average reference** over the 32 channels. The original online reference is undocumented, likely near FCz.
- Channel sets:

  | Set | Channels |
  |---|---|
  | frontal | Fp1, Fp2, F7, F3, Fz, F4, F8 |
  | temporal | T7, T8, TP9, TP10 |
  | sensorimotor | FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6 |
  | all | all 32 |

- Main baselines use all channels.
- Artifact controls run under both CAR and the original reference.

## Splits (D6)

- **Screening subjects: P3 and P6.**
  - Drawn once on 2026-10-29 with seed 20261029, before any model results (`configs/splits/default.yaml`).
  - Used only for Exp 1 (candidate screening). Never a test fold in Exp 2.
- **Main evaluation:** leave-one-subject-out over P1, P2, P4, P5, P7, P8, P9, P10, P11, P12.
- **Inner validation:** 2 of the 9 training subjects in each fold, drawn deterministically from the fold seed. All tuning, early stopping, and model selection use the inner validation only.
- **Secondary diagnostic:** within subject, train on S1–S6, test on S7–S9.
  - Confound: S7–S9 are always weight series, while S1–S6 mix weight, friction, and mixed series. The condition distribution therefore differs between train and test.
  - Report this split as a diagnostic only.

## Metrics (D7)

- **Primary:** Pearson r computed per trial and per axis over the scored segment. Then average over trials, then over axes, to get one value per subject. Report mean ± SD across subjects, and per axis.
- **Secondary:** R² and NRMSE (RMSE divided by the per-axis range of the test trial), aggregated the same way.
- Pooled r (over concatenated trials) only in an appendix, labeled "pooled".
- Every decoder is reported next to the **template** and **stereotypy** floors.
  - Trajectories are highly stereotyped: the mean trajectory explains 0.80–0.97 of per-axis variance.
  - The template uses oracle onset timing.

## Data included (D8)

- All lifts of series S1–S9 for every subject: 294 per subject.
- Weight, surface, previous weight/surface, and series type are kept as metadata for per-condition breakdowns. No lift is excluded by condition.
- Not used in this stage: `HS_*_ST` (EEG only, purpose unknown) and between-lift EEG from HS. They may serve as unlabeled data in the self-supervised stage.

## Normalization (D9)

- **EEG:** per-subject, per-channel z-score after filtering and re-referencing.
  - Statistics come from all of that subject's task EEG, with no labels used.
  - This is **transductive** for the test subject and is stated with every result.
  - A strict variant (statistics from training subjects only) is run as a sensitivity check.
- **Targets:** per-axis z-score with mean and SD from training subjects only. Predictions are mapped back to the original units before scoring.

## Artifacts (D10)

- No artifact removal beyond bad-channel interpolation.
- Contamination by eye movements, muscle, and EMG is estimated with artifact controls: decoders that use only frontal, temporal, sensorimotor, or EMG inputs, run under both references.
- An ICA-cleaned variant may be added later as a separate condition.

## Change log

(none)
