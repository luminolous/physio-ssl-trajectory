# Why the Kalman decoder has R² −3.08

**Conclusion: no bug was found. The negative R² comes from model misspecification.** `src/eegtraj/baselines/kalman.py` is unchanged, and the run `output/data-baselines/baselines/kalman/20261030-035949` stays valid.

Evidence comes from `20261104-131858_refit/` (folds P1 and P11, full data). Figures are in its `figures/` folder.

## Where the R² comes from

Per trial and axis, R² = 2ra − a² − b², with a = SD(pred) / SD(true) and b = mean(pred − true) / SD(true). Mean over the 10 test subjects (`r2_decomposition_summary.csv`):

| Model | r | a | +2ra | −a² | −b² | R² |
|---|---|---|---|---|---|---|
| Kalman | 0.466 | 1.96 | +1.80 | −4.43 | −0.45 | −3.08 |
| Ridge | 0.331 | 0.32 | +0.23 | −0.12 | −0.11 | −0.00 |
| EEGNet (seed 1) | 0.455 | 0.45 | +0.43 | −0.22 | −0.18 | 0.03 |
| CNN-LSTM (seed 1) | 0.505 | 0.58 | +0.61 | −0.37 | −0.21 | 0.04 |

Predictions swing about twice as wide as the target, while r stays moderate. The other decoders shrink their output (a < 1), which protects R². See `r2_decomposition.png`, `traces_P1.png`, `traces_P11.png`, and `rmse_by_time.png`.

## Bug checks

| Check | Result |
|---|---|
| Refit reproduces the old run | P1 R² −2.465 and P11 −6.334, equal to the old per-subject values |
| Independent filter vs decoder | max difference 0 on both folds |
| State initialization | x0 is about (−2.1, 2.2, −1.9) in normalized units, which is the onset position; RMSE at the segment start is ~0.5 cm |
| De-normalization | predictions start at the target and share its cm scale |
| Filter divergence | none: claimed position variance levels off (P1: 0.07–0.11 at sample 100, 0.10–0.13 at the end); gain norm settles at ~0.05–0.06 |
| Stability of A | all eigenvalue magnitudes are below 1 (P1 max 0.99992, P11 max 0.99942) |
| Units of A/Q | position process variance matches velocity process variance / fs² (P1 x: 7.5e-7 vs 0.0075 / 100²), after removing the 1e-6 diagonal term |

## Causes

1. **Observation noise is strongly autocorrelated, but the filter assumes it is white.**
   - Autocorrelation of the observation noise on test subjects: 0.993–0.995 at 10 ms, 0.64–0.79 at 100 ms, 0.37–0.51 at 500 ms.
   - Each sample's noise size is about right: normalized innovation squared per dimension is 0.99 (P1) and 1.47 (P11).
   - The filter therefore counts the same slow noise as new evidence at every sample and becomes overconfident.
   - Real position error variance is 29–42× the claimed variance on P1 and 42–75× on P11. On scored training subjects it is 22–51×, so this is not only a transfer effect.
2. **The dynamics prior alone already overshoots.** With no EEG, a² is 1.60 (P1) and 2.24 (P11), against offset terms of 0.08 and 0.19. This matches the prior-only R² of −1.63 being negative. The shape of the prior trajectory was not plotted.

## Counterfactual: inflate R

Diagnostic only, scored on the test subjects, so these values are not results and must not be reported as decoder scores. k = 50, from `counterfactuals.csv` and `figures/counterfactuals.png`:

| R multiplier | P1 r | P1 R² | P11 r | P11 R² |
|---|---|---|---|---|
| 1 (as run) | 0.580 | −2.47 | 0.404 | −6.33 |
| 10 | 0.638 | −0.90 | 0.485 | −1.00 |
| 30 | 0.624 | −0.36 | 0.419 | −0.62 |
| 100 | 0.549 | −0.29 | 0.279 | −0.97 |
| prior only | 0.437 | −0.59 | 0.161 | −1.98 |

- Less trust in the EEG improves both r and R², which supports cause 1.
- R² stays negative at every multiplier, which fits cause 2.
- Changing k does not fix it: at multiplier 1, k = 10 gives R² −3.04 (P1) and −4.74 (P11).

## Limits

- Refit diagnostics cover 2 of 10 folds. The decomposition table covers all 10 test subjects.
- Autocorrelation is measured, but its causal role is inferred from the counterfactual. No model with colored noise was fitted.
- k was selected by r on inner validation, so R² played no part in the choice.
