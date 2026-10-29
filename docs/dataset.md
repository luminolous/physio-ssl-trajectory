# WAY-EEG-GAL dataset

Reference: Luciw, Jarocka & Edin (2014), "Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction", *Scientific Data* 1:140047.

This page describes the files **as they are on disk**, verified with `scripts/inspect_mat.py`. The raw files live in `dataset/raw/P1` … `dataset/raw/P12` (9.9 GB, git-ignored, read-only).

## Task

The participant reaches for an object, grasps it with thumb and index finger, lifts it, holds it, puts it back, and releases it. Between lifts, the object's weight and/or the contact surface changes unpredictably.

## Files per subject

| File | Content | Use |
|---|---|---|
| `WS_P<n>_S<k>.mat` (k = 1..9) | windowed: one struct per lift with EEG, EMG, kinematics, events, and conditions | per-lift data |
| `HS_P<n>_S<k>.mat` (k = 1..9) | holistic: continuous signals for a whole series | continuous data |
| `HS_P<n>_ST.mat` | continuous EEG only (234–301 s), no kinematics, EMG, or events | **purpose unknown**, unlabeled |
| `P<n>_AllLifts.mat` | table with one row per lift (43 columns): events, conditions, force and grip measures | trial metadata |

All files are classic MATLAB v5. Load them with `scipy.io.loadmat(path, simplify_cells=True)`. Signal arrays are stored **samples × channels**.

## How WS, HS, and ST relate

Verified on P1 and P2 (18 series, 600 windows) with `scripts/compare_hs_ws.py`.

**WS vs HS**

- Each WS window is an exact copy of a slice of the HS series. The difference is 0 for EEG, EMG, and kinematics.
- A window starts at HS sample `round((StartTime − 2) · 500)`. The offset was 0 for every window.
- `StartTime` (= WS `trial_start_time`) is the **LED-on time** in series time, not the window start. The window begins 2 s earlier. Using `round(StartTime · 500)` directly is off by exactly 1000 samples.
- How much of HS the windows cover:
  - weight series: 99.6–99.9%;
  - friction and mixed series: only 54.5–67.2%, because the gaps between lifts last 6–8 s (likely surface changes).

  The EEG in these gaps is extra unlabeled data.
- HS has the `env` (surface, weight) and `misc` (button, LEDs, magnet, temperature) channels. WS adds 9 derived grip/load-force columns that HS lacks.

**ST**

- EEG only, same 32 channels as the task series: 234.3 s (P1), 301.4 s (P2).
- Its amplitude is lower than in S1. C3 std is 80 vs 161 (P1) and 139 vs 308 (P2). Frontal delta power is also lower, so there are fewer eye/movement artifacts.
- This is consistent with a non-task baseline recording, but the files do not document what ST is. **Its purpose is still unknown.**

**Planned use** (proposal, to be confirmed when the protocol is locked):

- Supervised decoding reads **WS**. Each window starts 2 s before the LED cue, and hand start is at ≥ 1.75 s, so a 500 ms EEG history before movement onset fits inside the window.
- **HS** is used only for continuous data, between-lift EEG, and the `env`/`misc` channels.
- Do not concatenate WS windows into a continuous stream: consecutive windows in weight series overlap by up to ~2.8 s.
- **ST** has no labels. Whether it is used, for example as unlabeled data for self-supervised pretraining, is open.

## Signals

| Modality | Channels | Rate | Notes |
|---|---|---|---|
| EEG | 32 (Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, TP9, CP5, CP1, CP2, CP6, TP10, P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10) | 500 Hz | units not stored, likely µV |
| EMG | 5 (anterior deltoid, brachioradialis, flexor digitorum, common extensor digitorum, first dorsal interosseus) | 4000 Hz | units not stored |
| Kinematics | 4 position sensors (x, y, z), 4 × 3 angles, 2 force/torque plates | 500 Hz | same samples as EEG |

Position sensors:

| Sensor | Body part | Evidence |
|---|---|---|
| 1 | object | does not move during the reach, rises ~6 when lifted |
| 2, 3 | index finger and thumb (which is which: not yet confirmed) | their distance equals the `GripAparture_Max` column |
| 4 | wrist | moves during the reach, stays at a fixed distance from both digits |

Position units are most likely cm. The coordinate frame is a fixed lab frame, with +z pointing up.

## Events and conditions

- Event times in AllLifts and WS (`tHandStart`, `tFirstDigitTouch`, `tLiftOff`, `tReplace`, `tHandStop`, …) are **seconds from the start of the lift's window**.
- Each window starts 2 s before the LED cue (`LEDOn = 2`).
- `StartTime` (= WS `trial_start_time`) is the LED-on time in series time. The window starts at HS sample `round((StartTime − 2) · 500)`.
- Weight: 1 = 165 g, 2 = 330 g, 4 = 660 g. Surface: 1 = silk, 2 = suede, 3 = sandpaper.
- Series types (`BlockType`): 1 = weight series, 2 = friction series, 3 = mixed. S7–S9 are weight series for every subject.

## Size

- 12 subjects × 294 lifts = **3,528 lifts** in the local files.
- Per subject: seven series of 34 lifts and two (mixed) series of 28.
- The paper title says 3,936. This difference is not yet explained.

## Caveats

- `tPeakVelHandReach` / `tPeakVelHandRetract` contain values far outside the window length. Do not treat them as times.
- WS windows of consecutive lifts can overlap by up to ~2.8 s (weight series).
