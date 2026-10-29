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
| `HS_P<n>_ST.mat` | continuous EEG only (~234 s), no kinematics or events | under investigation |
| `P<n>_AllLifts.mat` | table with one row per lift (43 columns): events, conditions, force and grip measures | trial metadata |

All files are classic MATLAB v5. Load them with `scipy.io.loadmat(path, simplify_cells=True)`. Signal arrays are stored **samples × channels**.

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
- Each window starts 2 s before the LED cue (`LEDOn = 2`). `StartTime` is the window start in series time.
- Weight: 1 = 165 g, 2 = 330 g, 4 = 660 g. Surface: 1 = silk, 2 = suede, 3 = sandpaper.
- Series types (`BlockType`): 1 = weight series, 2 = friction series, 3 = mixed. S7–S9 are weight series for every subject.

## Size

- 12 subjects × 294 lifts = **3,528 lifts** in the local files.
- Per subject: seven series of 34 lifts and two (mixed) series of 28.
- The paper title says 3,936. This difference is not yet explained.

## Caveats

- `tPeakVelHandReach` / `tPeakVelHandRetract` contain values far outside the window length. Do not treat them as times.
- WS windows of consecutive lifts can overlap by up to ~2.8 s (weight series).
