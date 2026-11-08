"""Oracle-warp ceiling: how far above the cue-locked template could a parametric head get?

ANALYSIS ONLY. Every parameter here is fitted by least squares to the test trial's **own** trajectory (oracle),
so no variant is a decoder. The result is an upper bound on what a head with these few free parameters could
reach if an encoder predicted them perfectly.

Base shape: the fold's cue-locked template (mean training trajectory vs time since LED-on, the ``template_cue``
baseline). Time is handled in seconds since LED-on, on the template's bin grid, so identity parameters reproduce
``template_cue`` exactly. Variants (``VARIANTS``):

- ``tau``: time shift. Template time g(t) = t - tau.
- ``tau_scale``: shift + time scale about the template's median onset m: g(t) = m + (t - m - tau) / s.
- ``tau_scale_amp``: plus a per-axis amplitude A (prediction A * T(g(t))).
- ``tau_scale_amp_offset``: plus a per-axis offset c (prediction A * T(g(t)) + c).
- ``piecewise_landmark``: piecewise-linear, continuous time warp that maps the trial's own phase boundaries
  (kinematic onset, first digit touch, lift-off, replace, hand stop) onto the template's median boundaries;
  slope 1 outside them. One shift + scale per phase, set by the oracle landmarks; plus A and c.
- ``piecewise_fit``: the same warp with the trial breakpoints refined by least squares (bounded around the
  landmarks, kept ordered, so the warp stays continuous); plus A and c.
- ``min_jerk``: no template. Four minimum-jerk submovements (Flash & Hogan 1985), superposed:
  reach, lift, lower, return. x(t) = p0 + sum_k d_k phi((t - t_k) / D_k), phi(u) = 10u^3 - 15u^4 + 6u^5 on
  [0, 1], 0 before and 1 after. Start t_k and duration D_k are fitted per submovement; p0 and the
  displacements d_k (per axis) are solved by linear least squares, so endpoints are continuous by construction.

A per-axis amplitude and offset do not change the per-trial, per-axis Pearson r. Variants 3 and 4 can raise
R2 and NRMSE only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

# Trial events that bound the phases, in time order (columns of the cached trials table).
PHASE_EVENTS = ("onset_s", "ev_tFirstDigitTouch", "ev_tLiftOff", "ev_tReplace", "ev_tHandStop")
SUBMOVEMENTS = ("reach", "lift", "lower", "return")
_SNAP = 1e-6  # template-bin positions this close to an integer are taken as that integer


@dataclass(frozen=True)
class Bounds:
    """Plausible ranges of the fitted parameters (seconds, unitless, cm)."""

    tau_s: tuple[float, float] = (-1.0, 2.0)
    scale: tuple[float, float] = (0.5, 2.0)
    amp: tuple[float, float] = (0.3, 3.0)
    offset_cm: tuple[float, float] = (-10.0, 10.0)
    breakpoint_shift_s: float = 0.3  # piecewise_fit: each breakpoint within +- this of its landmark
    mj_start_shift_s: float = 0.5  # min_jerk: each start within +- this of its initial value
    mj_duration_s: tuple[float, float] = (0.1, 3.0)
    mj_init_lift_lower_s: float = 0.5  # min_jerk: initial duration of the lift and lower submovements


@dataclass(frozen=True)
class FitSettings:
    tau_grid_step_s: float = 0.01
    scale_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0)
    diff_step: float = 1e-3
    max_nfev: int = 200
    bounds: Bounds = field(default_factory=Bounds)


@dataclass
class Template:
    """Cue-locked template as a continuous function of time since LED-on.

    values: float (3, n_bins), cm; bin k is at time (bin0 + k) / fs_hz after LED-on.
    landmarks_s: float (5,), median time since LED-on of ``PHASE_EVENTS`` over the training lifts.
    Linear interpolation between bins; edge values are held outside the range, as in ``template_cue``.
    """

    values: np.ndarray
    bin0: int
    fs_hz: float
    landmarks_s: np.ndarray

    def __call__(self, t_s: np.ndarray) -> np.ndarray:
        """Template at times ``t_s`` (any shape, s since LED-on) -> shape (*t_s.shape, 3)."""
        k = np.asarray(t_s, np.float64) * self.fs_hz - self.bin0
        kr = np.round(k)
        k = np.where(np.abs(k - kr) < _SNAP, kr, k)
        k = np.clip(k, 0, self.values.shape[1] - 1)
        i = np.minimum(np.floor(k).astype(np.int64), self.values.shape[1] - 2)
        w = (k - i)[..., None]
        v = self.values.T.astype(np.float64)
        return v[i] * (1 - w) + v[i + 1] * w

    @property
    def onset_s(self) -> float:
        return float(self.landmarks_s[0])


@dataclass
class TrialData:
    """One test trial over the scored segment.

    t_s: (n,) seconds since LED-on, on the template bin grid (round(t * fs) / fs, as ``template_cue``).
    y: (n, 3) re-centered wrist position, cm.
    landmarks_s: (5,) the trial's own ``PHASE_EVENTS`` in seconds since LED-on (oracle).
    """

    t_s: np.ndarray
    y: np.ndarray
    landmarks_s: np.ndarray


@dataclass
class FitResult:
    variant: str
    pred: np.ndarray  # (n, 3) cm
    params: dict[str, float]
    cost: float  # 0.5 * sum of squared errors, cm^2
    at_bound: list[str]
    success: bool


# ---------------------------------------------------------------- warps and the minimum-jerk basis

def shift_scale_warp(t_s: np.ndarray, tau_s: float, scale: float, pivot_s: float) -> np.ndarray:
    """Template time for trial time ``t_s``: pivot + (t - pivot - tau) / scale. Identity at tau=0, scale=1."""
    return pivot_s + (t_s - pivot_s - tau_s) / scale


def piecewise_warp(t_s: np.ndarray, trial_bp_s: np.ndarray, template_bp_s: np.ndarray) -> np.ndarray:
    """Continuous piecewise-linear map of trial breakpoints onto template breakpoints; slope 1 outside them."""
    g = np.interp(t_s, trial_bp_s, template_bp_s)
    g = np.where(t_s < trial_bp_s[0], t_s - trial_bp_s[0] + template_bp_s[0], g)
    return np.where(t_s > trial_bp_s[-1], t_s - trial_bp_s[-1] + template_bp_s[-1], g)


def min_jerk_profile(u: np.ndarray) -> np.ndarray:
    """Normalized minimum-jerk position 10u^3 - 15u^4 + 6u^5, with u clipped to [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def min_jerk_basis(t_s: np.ndarray, starts_s: np.ndarray, durations_s: np.ndarray) -> np.ndarray:
    """Design matrix (n, 1 + n_sub): a constant column, then one min-jerk step per submovement."""
    cols = [np.ones_like(t_s)] + [min_jerk_profile((t_s - a) / d) for a, d in zip(starts_s, durations_s)]
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------- per-axis amplitude and offset

def solve_amp_offset(T: np.ndarray, y: np.ndarray, mode: str, bounds: Bounds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded least squares of y ~ A * T + c, independently per axis.

    T, y: (n, 3). mode: ``none`` (A=1, c=0), ``amp`` (c=0), or ``amp_offset``.
    Returns (pred (n, 3), A (3,), c (3,)). The box QP is solved exactly: the optimum is the interior solution or
    the clipped 1-D optimum on one of the box edges.
    """
    n_axes = T.shape[1]
    if mode == "none":
        return T.copy(), np.ones(n_axes), np.zeros(n_axes)
    alo, ahi = bounds.amp
    clo, chi = bounds.offset_cm
    A, c = np.ones(n_axes), np.zeros(n_axes)
    for a in range(n_axes):
        t, v = T[:, a], y[:, a]
        tt = max(float(t @ t), 1e-12)
        if mode == "amp":
            A[a] = np.clip(float(t @ v) / tt, alo, ahi)
            continue
        if mode != "amp_offset":
            raise ValueError(f"unknown mode {mode!r}")
        cands = []
        tc, vc = t - t.mean(), v - v.mean()
        var = float(tc @ tc)
        if var > 1e-12:
            a0 = float(tc @ vc) / var
            c0 = v.mean() - a0 * t.mean()
            if alo <= a0 <= ahi and clo <= c0 <= chi:
                cands.append((a0, c0))
        for af in (alo, ahi):
            cands.append((af, float(np.clip(np.mean(v - af * t), clo, chi))))
        for cf in (clo, chi):
            cands.append((float(np.clip(float(t @ (v - cf)) / tt, alo, ahi)), cf))
        sse = [float(np.sum((v - ca * t - cc) ** 2)) for ca, cc in cands]
        A[a], c[a] = cands[int(np.argmin(sse))]
    return T * A + c, A, c


# ---------------------------------------------------------------- fitting

def _at_bound(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, names: list[str], tol: float = 1e-6) -> list[str]:
    return [n for n, v, a, b in zip(names, x, lo, hi) if v <= a + tol or v >= b - tol]


def _lin_flags(A: np.ndarray, c: np.ndarray, mode: str, bounds: Bounds) -> list[str]:
    out = []
    for name, vals, (lo, hi), used in (("amp", A, bounds.amp, mode != "none"), ("offset", c, bounds.offset_cm, mode == "amp_offset")):
        if used:
            out += [f"{name}_{ax}" for ax, v in zip("xyz", vals) if v <= lo + 1e-6 or v >= hi - 1e-6]
    return out


def _lin_params(A: np.ndarray, c: np.ndarray, mode: str) -> dict[str, float]:
    out = {}
    if mode != "none":
        out |= {f"amp_{ax}": float(v) for ax, v in zip("xyz", A)}
    if mode == "amp_offset":
        out |= {f"offset_{ax}": float(v) for ax, v in zip("xyz", c)}
    return out


def _sse(pred: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * float(np.sum((pred - y) ** 2))


def fit_shift_scale(trial: TrialData, template: Template, use_scale: bool, mode: str, settings: FitSettings,
                    x0: np.ndarray | None = None, grid: bool = True) -> FitResult:
    """Variants ``tau`` (use_scale False) and ``tau_scale*``. Grid search (A=1, c=0), then bounded least squares
    from the grid optimum and from ``x0``. With ``grid=False`` only ``x0`` is used."""
    b = settings.bounds
    pivot = template.onset_s

    def predict(x):
        g = shift_scale_warp(trial.t_s, x[0], x[1] if use_scale else 1.0, pivot)
        return solve_amp_offset(template(g), trial.y, mode, b)

    n_x = 2 if use_scale else 1
    starts = []
    if grid or x0 is None:
        taus = np.arange(b.tau_s[0], b.tau_s[1] + 1e-9, settings.tau_grid_step_s)
        scales = np.array([s for s in settings.scale_grid if b.scale[0] <= s <= b.scale[1]]) if use_scale else np.array([1.0])
        sse = np.empty((taus.size, scales.size))  # one scale at a time to bound memory
        for j, s in enumerate(scales):
            g = shift_scale_warp(trial.t_s[None, :], taus[:, None], s, pivot)
            sse[:, j] = np.sum((template(g) - trial.y) ** 2, axis=(-1, -2))
        i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
        starts.append(np.array([taus[i], scales[j]][:n_x]))
    if x0 is not None:
        starts.append(np.asarray(x0, np.float64)[:n_x])
    lo = np.array([b.tau_s[0], b.scale[0]][:n_x])
    hi = np.array([b.tau_s[1], b.scale[1]][:n_x])
    best = None
    for s0 in starts:
        s0 = np.clip(s0, lo, hi)
        res = least_squares(lambda x: (predict(x)[0] - trial.y).ravel(), s0, bounds=(lo, hi),
                            diff_step=settings.diff_step, max_nfev=settings.max_nfev)
        if best is None or res.cost < best.cost:
            best = res
    pred, A, c = predict(best.x)
    names = ["tau_s", "scale"][: len(best.x)]
    params = {n: float(v) for n, v in zip(names, best.x)} | _lin_params(A, c, mode)
    variant = {("none", False): "tau", ("none", True): "tau_scale", ("amp", True): "tau_scale_amp",
               ("amp_offset", True): "tau_scale_amp_offset"}.get((mode, use_scale), f"shift_{mode}")
    return FitResult(variant, pred, params, _sse(pred, trial.y), _at_bound(best.x, lo, hi, names)
                     + _lin_flags(A, c, mode, b), bool(best.success))


def _ordered(bp: np.ndarray, min_gap_s: float) -> np.ndarray:
    """Make breakpoints strictly increasing (each at least ``min_gap_s`` after the previous one)."""
    out = bp.copy()
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1] + min_gap_s)
    return out


def fit_piecewise(trial: TrialData, template: Template, refine: bool, settings: FitSettings) -> FitResult:
    """Variants ``piecewise_landmark`` (refine False) and ``piecewise_fit``; both with per-axis A and c."""
    b = settings.bounds
    gap = 1.0 / template.fs_hz
    tmpl_bp = template.landmarks_s

    def predict(bp):
        g = piecewise_warp(trial.t_s, _ordered(bp, gap), tmpl_bp)
        return solve_amp_offset(template(g), trial.y, "amp_offset", b)

    x = _ordered(trial.landmarks_s.astype(np.float64), gap)
    flags: list[str] = []
    success = True
    names = [f"bp_{e}" for e in ("onset", "touch", "liftoff", "replace", "handstop")]
    if refine:
        lo, hi = x - b.breakpoint_shift_s, x + b.breakpoint_shift_s
        res = least_squares(lambda v: (predict(v)[0] - trial.y).ravel(), x, bounds=(lo, hi),
                            diff_step=settings.diff_step, max_nfev=settings.max_nfev)
        x, success = _ordered(res.x, gap), bool(res.success)
        flags = _at_bound(res.x, lo, hi, names)
    pred, A, c = predict(x)
    seg_scale = np.diff(tmpl_bp) / np.diff(x)  # template seconds per trial second, per phase
    params = {n: float(v) for n, v in zip(names, x)}
    params |= {f"bp_shift_{n[3:]}": float(v - l) for n, v, l in zip(names, x, trial.landmarks_s)}
    params |= {f"phase_rate_{k}": float(v) for k, v in enumerate(seg_scale)}
    params |= _lin_params(A, c, "amp_offset")
    return FitResult("piecewise_fit" if refine else "piecewise_landmark", pred, params, _sse(pred, trial.y),
                     flags + _lin_flags(A, c, "amp_offset", b), success)


def min_jerk_init(landmarks_s: np.ndarray, bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    """Initial (starts, durations) of reach, lift, lower, return from the trial's phase boundaries."""
    onset, touch, liftoff, replace, stop = landmarks_s
    d = bounds.mj_init_lift_lower_s
    starts = np.array([onset, liftoff, replace - d, replace])
    durations = np.array([touch - onset, d, d, stop - replace])
    return starts, np.clip(durations, *bounds.mj_duration_s)


def fit_min_jerk(trial: TrialData, settings: FitSettings) -> FitResult:
    """Variant ``min_jerk``: 8 nonlinear timing parameters, 5 linear coefficients per axis."""
    b = settings.bounds
    s0, d0 = min_jerk_init(trial.landmarks_s, b)
    x0 = np.concatenate([s0, d0])
    lo = np.concatenate([s0 - b.mj_start_shift_s, np.full(4, b.mj_duration_s[0])])
    hi = np.concatenate([s0 + b.mj_start_shift_s, np.full(4, b.mj_duration_s[1])])

    def predict(x):
        X = min_jerk_basis(trial.t_s, x[:4], x[4:])
        coef, *_ = np.linalg.lstsq(X, trial.y, rcond=None)  # (5, 3)
        return X @ coef, coef

    res = least_squares(lambda x: (predict(x)[0] - trial.y).ravel(), np.clip(x0, lo, hi), bounds=(lo, hi),
                        diff_step=settings.diff_step, max_nfev=settings.max_nfev)
    pred, coef = predict(res.x)
    names = [f"{k}_start_s" for k in SUBMOVEMENTS] + [f"{k}_duration_s" for k in SUBMOVEMENTS]
    params = {n: float(v) for n, v in zip(names, res.x)}
    params |= {f"{k}_start_shift_s": float(v - s) for k, v, s in zip(SUBMOVEMENTS, res.x[:4], s0)}
    params |= {f"p0_{ax}": float(coef[0, a]) for a, ax in enumerate("xyz")}
    params |= {f"{k}_disp_{ax}": float(coef[1 + i, a]) for i, k in enumerate(SUBMOVEMENTS) for a, ax in enumerate("xyz")}
    return FitResult("min_jerk", pred, params, _sse(pred, trial.y), _at_bound(res.x, lo, hi, names), bool(res.success))


VARIANTS = ("tau", "tau_scale", "tau_scale_amp", "tau_scale_amp_offset", "piecewise_landmark", "piecewise_fit",
            "min_jerk")

# Free parameters per trial (all axes). Landmark breakpoints are oracle event times, not fitted.
N_PARAMS = {"template_cue": 0, "tau": 1, "tau_scale": 2, "tau_scale_amp": 5, "tau_scale_amp_offset": 8,
            "piecewise_landmark": 6, "piecewise_fit": 11, "min_jerk": 23}


def fit_all(trial: TrialData, template: Template, settings: FitSettings) -> dict[str, FitResult]:
    """Every variant for one trial. The amplitude variants start from the previous variant's timing (no grid)."""
    out = {"tau": fit_shift_scale(trial, template, False, "none", settings)}
    out["tau_scale"] = fit_shift_scale(trial, template, True, "none", settings, x0=np.array([out["tau"].params["tau_s"], 1.0]))
    prev = np.array([out["tau_scale"].params["tau_s"], out["tau_scale"].params["scale"]])
    out["tau_scale_amp"] = fit_shift_scale(trial, template, True, "amp", settings, x0=prev, grid=False)
    prev = np.array([out["tau_scale_amp"].params["tau_s"], out["tau_scale_amp"].params["scale"]])
    out["tau_scale_amp_offset"] = fit_shift_scale(trial, template, True, "amp_offset", settings, x0=prev, grid=False)
    out["piecewise_landmark"] = fit_piecewise(trial, template, False, settings)
    out["piecewise_fit"] = fit_piecewise(trial, template, True, settings)
    out["min_jerk"] = fit_min_jerk(trial, settings)
    return out
