# Experimental protocol

Locked on 2026-10-29, updated on 2026-11-06 after the diagnostics stage. Changes are recorded with a date and reason at the bottom. Runs made under an older protocol are [marked as such](#runs-under-the-older-protocol). Data facts behind these choices: [dataset.md](dataset.md).

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
- **Causal EEG history of 2 s** (200 samples) before each target sample, at 100 Hz, for all later models.
  - Config: `configs/data/windows_long_history.yaml`.
  - Evidence: `residual_r_specific` does not rise from 2 s to 3 s, and CNN-LSTM stops improving at 2 s. 2 s is not best on every metric: EEGNet has a higher R² at 3 s (0.125 against 0.004). See [diagnostics-results.md](diagnostics-results.md).
- **A rise in r from a longer history must not be claimed as motor information.** Almost all of it is available from Fp1 and Fp2 alone: EEGNet under CAR, 0.5 s to 3 s, gains r +0.152 on those two channels against +0.154 on all 32. It fits cue-locked timing or eye activity, and the control does not separate them. This was measured at 3 s only, not at 2 s, and not with CNN-LSTM.
- **Pre-lift EEG.** The main cache holds only each lift's window, and neighbouring windows overlap, so it cannot serve a history that starts before the lift. At 2 s, 67.1% of lifts need samples from before their window.
  - They are cut from the continuous recording with the same causal chain and grid (`scripts/build_pre_lift_cache.py`).
  - **The pre-lift cache exists only for broadband CAR.** Other variants (low-frequency, original reference) are built when needed.
  - Lifts whose history reaches before the series start (1.4% at 2 s) hold the earliest value on the lift's grid. They are kept.
  - EEG normalization statistics stay computed from the lift samples only.
- Ridge at 500 ms uses lags every 20 ms inside the window: 0, 20, …, 480 ms (25 lags, 800 features with 32 channels). There is no ridge run at 2 s, which would need the low-frequency pre-lift cache. Deep models get the full window.

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
  - The 2 s history covers this, as did the earlier 500 ms. Read decoder lag profiles as EEG lag plus filter delay.
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
  | prefrontal (control, subset of frontal) | Fp1, Fp2 |
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
- **Secondary diagnostic:** within subject, train on S1–S5, validate on S6, test on S7–S9.
  - Confound: S7–S9 are always weight series, while S1–S6 mix weight, friction, and mixed series. The condition distribution therefore differs between train and test.
  - Report this split as a diagnostic only.

## Metrics (D7)

- **Primary:** Pearson r computed per trial and per axis over the scored segment. Then average over trials, then over axes, to get one value per subject. Report mean ± SD across subjects, and per axis.
- **Secondary:** R² and NRMSE (RMSE divided by the per-axis range of the test trial), aggregated the same way.
- **Secondary, residual metrics**, aggregated the same way:
  - `residual_r`: r between (prediction − T_cue) and (target − T_cue), with T_cue the cue-locked template from the training subjects.
  - `residual_r_specific`: `residual_r` minus `residual_r_other_trial`, a control that pairs each predicted residual with another trial of the same subject. `residual_r` alone also picks up what is shared across a subject's trials.
  - `residual_r_oracle` uses the oracle-aligned template and is **analysis only**.
- Pooled r (over concatenated trials) only in an appendix, labeled "pooled".
- Every decoder is reported next to the official floors:
  - **Main fair floor: cue-locked template** (`template_cue`). It uses LED-on only and no EEG. LOSO r 0.803.
  - **Secondary fair floor: onset decoder + template** (`onset_template`). LOSO r 0.754.
  - **Upper reference: oracle template** (`template`). It uses the test trial's kinematic onset, so it is not a fair floor. LOSO r 0.864.
  - The stereotypy floor stays reported. Trajectories are highly stereotyped: the mean trajectory explains 0.80–0.97 of per-axis variance.

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
- Contamination by eye movements, muscle, and EMG is estimated with artifact controls: decoders that use only frontal, prefrontal, temporal, sensorimotor, or EMG inputs, run under both references.
- A long-history prefrontal control exists for CAR only, because of the pre-lift cache (D3).
- An ICA-cleaned variant may be added later as a separate condition.

## Deep-model budget (D11)

- **Supervised deep models: fixed budget.** lr 1e-3, weight decay 0.01, at most 30 epochs, patience 5, 3 seeds, no lr grid.
- Evidence: a per-fold lr grid gave no gain (EEGNet r 0.449 to 0.456, CNN-LSTM 0.499 to 0.478), and the selection gain is no larger than the noise from initial weights.
- Known limit: lr 1e-3 is the upper edge of the tested grid. At 3 s the grid selected a lower lr in 5 of 10 folds, without a test-score gain.
- The budget for SSL fine-tuning and the linear probe is open. It is decided when the SSL stage is activated (`.agents/ssl-pretraining/design.md`, SSL-8 and SSL-9).

## Claims (D12)

- Claims are relative: SSL against supervised from scratch, on the same LOSO protocol. They are reported with r, R², NRMSE, `residual_r_specific`, and the official floors.
- A rise in r without a rise in `residual_r_specific` does not count as trial-specific information.
- A rise in r from a longer history must not be claimed as motor information (D3).
- A claimed result comes with a prefrontal control.
- Decoding onset from EEG gave no benefit over the cue: `onset_template` r 0.754 against `template_cue` r 0.803. So no claim may rest on EEG-derived timing.
- Beating the cue-locked template is not required. On average no decoder approaches it, and giving a model the cue time as an input could raise r with no EEG information.

## Runs under the older protocol

These runs used the 500 ms history and the floors and metrics before 2026-11-06. They stay valid under that protocol and remain references. They were not changed or rerun.

| Runs | Note |
|---|---|
| `output/data-baselines/baselines/{ridge_lag,kalman,eegnet,shallowconvnet,cnn_lstm}/` | 500 ms |
| `output/data-baselines/artifact-controls/` | 500 ms |
| `output/diagnostics/tuning/` | 500 ms, lr grid |
| `output/diagnostics/baselines/onset_template/` | lagged EEG within 500 ms; stays the secondary floor |
| `output/diagnostics/within_subject/ridge_lag/` | 500 ms |
| `output/diagnostics/artifact-controls/{ridge,eegnet}_prefrontal_{car,orig}/` (direct runs) | 500 ms |

`template`, `template_cue`, and `stereotypy` use no EEG, so the history does not apply to them. Runs at other lengths in `output/diagnostics/history/` and the 3 s runs elsewhere are diagnostic. Only `history_200` matches the current D3.

## Change log

- 2026-11-06, D3: EEG history changed from 500 ms to 2 s at 100 Hz, with pre-lift EEG for broadband CAR. Reason: history ablation. Decided by the user.
- 2026-11-06, D7: added `residual_r` and `residual_r_specific` as secondary metrics and `residual_r_oracle` as analysis only. `template_cue` becomes the main fair floor, `onset_template` the secondary one, and the oracle template an upper reference. Reason: the oracle template is not a fair floor, and r alone hides how little is trial-specific. Decided by the user.
- 2026-11-06, D5, D10: added the prefrontal channel set as an artifact control.
- 2026-11-06, D11 (new): fixed budget for supervised deep models. Reason: lr tuning gave no gain. Decided by the user. The SSL budget stays open.
- 2026-11-06, D12 (new): claim rules for the SSL stage. Decided by the user.

- 2026-10-29, D6: added S6 as the inner-validation series of the within-subject diagnostic (train S1–S5, val S6, test S7–S9). No runs existed under the old wording.
- 2026-10-29, D3: ridge lags corrected to 25 (0…480 ms) to fit the 50-sample history. It had been stated as 26. No runs existed before.
