"""Movement timelapse: one lift as an animation, input EEG next to the 3D wrist trajectory and saved predictions.

Times are seconds since LED-on. Lift sample j (0-based, in the lift window) sits at (j + 1) / fs_hz after the window
start, the same convention as ``eegtraj.baselines.template.time_since_event``. Predictions come from saved
``predictions.npz`` files only; they exist on the protocol scoring segment [onset - 0.5 s, tHandStop] and are NaN
elsewhere. No model is trained or run here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eegtraj.data.cache import load_subject
from eegtraj.data.windows import WRIST_INDEX, segment_samples

OBJECT_INDEX = 0  # sensor 1 in kin_pos (4, 3, T): the object (docs/dataset.md)
# Events drawn on the timeline, in time order: label -> trials column (LED-on is time 0 by definition).
EVENTS = {"LED on": "ev_LEDOn", "onset": "onset_s", "first touch": "ev_tFirstDigitTouch", "lift-off": "ev_tLiftOff",
          "replace": "ev_tReplace", "release": "ev_tBothReleased", "hand stop": "ev_tHandStop"}
# Movement phases of the ground truth, in order. Boundaries: onset, first touch, hold start, hold end, release, hand
# stop. Hold start and end come from the object height (``hold_interval``); the dataset has no hold event.
PHASES = ("reach", "lift", "hold", "replace", "return")
PHASE_COLORS = {"reach": "#0d0887", "lift": "#7e03a8", "hold": "#cc4778", "replace": "#f0803a", "return": "#c9a800"}
OUTSIDE_COLOR = "0.7"  # ground truth before onset and after hand stop
HOLD_FRACTION = 0.9  # hold = object at or above this fraction of its peak rise above the lift-off height


@dataclass
class LiftBundle:
    """Everything one animation needs for one lift.

    t_s: (n,) seconds since LED-on for every sample of the lift window.
    eeg: label -> (n,) EEG trace (cached units, likely uV). topo: (32, n) EEG for the topomap, or None.
    wrist, obj: (n, 3) cm, re-centered on the wrist position at kinematic onset (as the decoder target).
    preds: label -> (n, 3) cm, NaN outside the scored segment.
    events: label -> seconds since LED-on.
    phases: phase -> (start, end) seconds since LED-on (``movement_phases``).
    """

    subject: str
    series: int
    lift: int
    weight_g: int
    surface: str
    t_s: np.ndarray
    eeg: dict[str, np.ndarray]
    wrist: np.ndarray
    obj: np.ndarray
    preds: dict[str, np.ndarray]
    events: dict[str, float]
    eeg_names: list[str] = field(default_factory=list)
    topo: np.ndarray | None = None
    phases: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def rt_s(self) -> float:
        return self.events["onset"]


# ---------------------------------------------------------------- timing

def frame_times(t_start_s: float, t_end_s: float, fps: float, speed: float) -> np.ndarray:
    """Cursor time (s since LED-on) of every frame: t_start + k * speed / fps, up to t_end inclusive."""
    n = int(np.floor((t_end_s - t_start_s) * fps / speed + 1e-9)) + 1
    return t_start_s + np.arange(n) * speed / fps


def sample_at(t_s: np.ndarray, time_s: float) -> int:
    """Index of the last lift sample at or before ``time_s`` (the cursor draws data up to and including it)."""
    return int(np.searchsorted(t_s, time_s + 1e-9, side="right") - 1)


def event_frames(events: dict[str, float], t_start_s: float, fps: float, speed: float) -> dict[str, int]:
    """Frame whose cursor time is closest to each event."""
    return {k: int(round((v - t_start_s) * fps / speed)) for k, v in events.items()}


def hold_interval(t_s: np.ndarray, obj_z: np.ndarray, t_lift_off: float, t_replace: float,
                  fraction: float = HOLD_FRACTION) -> tuple[float, float]:
    """(start, end) s of the hold: first and last sample between lift-off and replace at which the object is at or
    above ``fraction`` of its peak rise above the lift-off height."""
    m = (t_s >= t_lift_off) & (t_s <= t_replace)
    if not m.any():
        raise ValueError("no sample between lift-off and replace")
    rise = obj_z[m] - float(np.interp(t_lift_off, t_s, obj_z))
    up = np.flatnonzero(rise >= fraction * rise.max())
    return float(t_s[m][up[0]]), float(t_s[m][up[-1]])


def movement_phases(events: dict[str, float], hold: tuple[float, float]) -> dict[str, tuple[float, float]]:
    """Phase -> (start, end) s. reach: onset to first touch; lift: first touch to hold start (grasp, load, lift-off,
    rise); hold; replace: hold end to release (lower, set down, release); return: release to hand stop."""
    edges = [events["onset"], events["first touch"], hold[0], hold[1], events["release"], events["hand stop"]]
    if np.any(np.diff(edges) < 0):
        raise ValueError(f"phase boundaries out of order: {edges}")
    return {p: (edges[k], edges[k + 1]) for k, p in enumerate(PHASES)}


def phase_at(phases: dict[str, tuple[float, float]], events: dict[str, float], time_s: float) -> str:
    """Phase name at ``time_s``: a movement phase, 'reaction' between LED-on and onset, else 'rest'."""
    for name, (a, b) in phases.items():
        if a <= time_s < b:
            return name
    if events["LED on"] <= time_s < events["onset"]:
        return "reaction"
    return "rest"


def phase_samples(t_s: np.ndarray, phases: dict[str, tuple[float, float]], i0: int, i: int) -> dict[str, slice]:
    """Sample slice of every ground-truth piece drawn from ``i0`` up to sample ``i``: 'before' (to onset), each phase,
    'after' (from hand stop). Neighbouring pieces share their boundary sample, so the line stays continuous. Pieces
    with fewer than two samples are left out."""
    names = ["before", *phases, "after"]
    edges = [sample_at(t_s, a) for a, _ in phases.values()] + [sample_at(t_s, list(phases.values())[-1][1])]
    lo = [i0] + edges
    hi = edges + [len(t_s) - 1]
    out = {}
    for name, a, b in zip(names, lo, hi):
        a, b = max(a, i0), min(b, i)
        if b > a:
            out[name] = slice(a, b + 1)
    return out


# ---------------------------------------------------------------- data

def lift_times(trial: pd.Series, fs_hz: float) -> np.ndarray:
    """(n_samples,) seconds since LED-on of the lift window samples."""
    return (np.arange(int(trial.n_samples)) + 1) / fs_hz - float(trial.ev_LEDOn)


def lift_predictions(path: Path, subject: str, trials: pd.DataFrame, row: int, fs_hz: float,
                     wrist_check: np.ndarray | None = None) -> np.ndarray:
    """Saved predictions of one lift on its whole window: (n_samples, 3), NaN outside the scored segment.

    The saved samples of a trial are its segment samples in order (``segment_samples``), whatever the run's
    history or pre-lift layout. With ``wrist_check`` the saved targets must equal the re-centered wrist there.
    """
    with np.load(Path(path) / "predictions.npz") as z:
        rows, y_pred, y_true = z[f"{subject}__trial_row"], z[f"{subject}__y_pred"], z[f"{subject}__y_true"]
    idx, seg_rows = segment_samples(trials, fs_hz, 1)
    m = rows == row
    j = idx[seg_rows == row] - int(trials.offset_start.iloc[row])  # sample indices inside the lift window
    if m.sum() != j.size:
        raise ValueError(f"{path}: {subject} row {row} has {m.sum()} saved samples, segment has {j.size}")
    out = np.full((int(trials.n_samples.iloc[row]), 3), np.nan, dtype=np.float64)
    out[j] = y_pred[m]
    if wrist_check is not None and not np.allclose(y_true[m], wrist_check[j], atol=1e-3):
        raise ValueError(f"{path}: saved targets of {subject} row {row} differ from the cached wrist")
    return out


def load_lift(cache_dir: str | Path, subject: str, row: int, channels: list[str], cleaned_channels: list[str],
              prediction_runs: dict[str, str | Path], topomap_band: str | None = "broadband") -> LiftBundle:
    """Load one lift (trial ``row`` of ``subject``) with its EEG traces, kinematics, events, and predictions.

    channels: uncleaned broadband CAR traces. cleaned_channels: ICA-cleaned traces, labeled '<ch> (ICA)'.
    """
    arrays = ["kin_pos", "eeg_broadband_car"] + (["eeg_broadband_ica_car"] if cleaned_channels else [])
    data, trials, manifest = load_subject(cache_dir, subject, arrays=arrays)
    fs, names = float(manifest["fs_hz"]), list(manifest["eeg_names"])
    tr = trials.iloc[row]
    sl = slice(int(tr.offset_start), int(tr.offset_end))
    kin = np.asarray(data["kin_pos"][:, :, sl], dtype=np.float64)  # (4, 3, n)
    t_lift = (np.arange(int(tr.n_samples)) + 1) / fs
    ref = np.array([np.interp(tr.onset_s, t_lift, kin[WRIST_INDEX, a]) for a in range(3)])  # as recentered_wrist
    wrist = (kin[WRIST_INDEX] - ref[:, None]).T
    obj = (kin[OBJECT_INDEX] - ref[:, None]).T
    eeg = {ch: np.asarray(data["eeg_broadband_car"][names.index(ch), sl], dtype=np.float64) for ch in channels}
    eeg |= {f"{ch} (ICA)": np.asarray(data["eeg_broadband_ica_car"][names.index(ch), sl], dtype=np.float64)
            for ch in cleaned_channels}
    topo = np.asarray(data[f"eeg_{topomap_band}_car"][:, sl], dtype=np.float64) if topomap_band else None
    preds = {label: lift_predictions(Path(p), subject, trials, row, fs, wrist) for label, p in prediction_runs.items()}
    events = {label: float(tr[col]) - float(tr.ev_LEDOn) for label, col in EVENTS.items()}
    t_s = lift_times(tr, fs)
    phases = movement_phases(events, hold_interval(t_s, obj[:, 2], events["lift-off"], events["replace"]))
    return LiftBundle(subject, int(tr.series), int(tr.lift), int(tr.weight_g), str(tr.surface), t_s, eeg, wrist, obj,
                      preds, events, names, topo, phases)


def select_lifts(trials: dict[str, pd.DataFrame], rules: dict[str, dict], seed: int,
                 exclude_flags: tuple[str, ...] = ("onset_mismatch", "filter_transient_broadband")) -> pd.DataFrame:
    """One lift per rule, each from a different subject, drawn with ``seed``.

    trials: subject -> trials table. A rule may set ``subject``, ``weight_g``, and an RT percentile range
    ``rt_percentile: [lo, hi]`` (percentiles over all given lifts). Rules with a fixed subject are drawn first.
    Returns one row per rule: rule, subject, row, series, lift, weight_g, surface, rt_s.
    """
    rng = np.random.default_rng(seed)
    all_rows = []
    for s, t in trials.items():
        keep = ~t[list(exclude_flags)].astype(bool).any(axis=1)
        all_rows.append(t.loc[keep].assign(subject=s, row=np.flatnonzero(keep), rt_s=(t.onset_s - t.ev_LEDOn)[keep]))
    pool = pd.concat(all_rows, ignore_index=True)
    used: set[str] = set()
    out = []
    for name, rule in sorted(rules.items(), key=lambda kv: "subject" not in kv[1]):
        c = pool
        if "subject" in rule:
            c = c[c.subject == rule["subject"]]
        else:
            c = c[~c.subject.isin(used)]
        if "weight_g" in rule:
            c = c[c.weight_g == rule["weight_g"]]
        if "rt_percentile" in rule:
            lo, hi = np.percentile(pool.rt_s, rule["rt_percentile"])
            c = c[(c.rt_s >= lo) & (c.rt_s <= hi)]
        if c.empty:
            raise ValueError(f"no lift matches rule {name!r}")
        pick = c.iloc[int(rng.integers(len(c)))]
        used.add(pick.subject)
        out.append({"rule": name, "subject": pick.subject, "row": int(pick.row), "series": int(pick.series),
                    "lift": int(pick.lift), "weight_g": int(pick.weight_g), "surface": pick.surface,
                    "rt_s": float(pick.rt_s)})
    order = {k: i for i, k in enumerate(rules)}
    return pd.DataFrame(out).sort_values("rule", key=lambda r: r.map(order)).reset_index(drop=True)


# ---------------------------------------------------------------- animation

GROUND_TRUTH_LW = 3.0
STYLE = {"template_cue": dict(color="0.55", lw=1.5, ls="--")}
PALETTE = ("tab:green", "tab:cyan", "tab:brown", "tab:olive", "tab:blue")  # kept apart from PHASE_COLORS
EMPTY = np.empty(0)  # mplot3d needs arrays, not lists, for empty lines


def axis_limits(bundle: LiftBundle, i0: int, i1: int, margin_cm: float = 1.0) -> list[tuple[float, float]]:
    """Fixed per-axis limits (cm) covering every drawn trajectory between samples i0 and i1, equal spans."""
    arrs = [bundle.wrist[i0:i1 + 1], bundle.obj[i0:i1 + 1]] + [p[i0:i1 + 1] for p in bundle.preds.values()]
    stack = np.concatenate(arrs)
    lo, hi = np.nanmin(stack, axis=0) - margin_cm, np.nanmax(stack, axis=0) + margin_cm
    half = (hi - lo).max() / 2
    mid = (hi + lo) / 2
    return [(m - half, m + half) for m in mid]


def animate_lift(bundle: LiftBundle, path: str | Path, fps: float = 25, speed: float = 1.0, pre_s: float = 1.0,
                 post_s: float = 0.5, topomap: bool = True, view: tuple[float, float] | None = None,
                 figsize: tuple[float, float] = (12.8, 7.2), dpi: int = 100) -> dict:
    """Render the timelapse of one lift to an MP4 (needs ffmpeg, e.g. from imageio-ffmpeg).

    The window runs from LED-on - ``pre_s`` to hand stop + ``post_s`` (clipped to the lift window). ``speed`` 1 is
    real time. ``view`` is (elevation, azimuth) in degrees; by default the azimuth puts the reach (wrist at first
    touch) across the screen, left to right. Returns timing facts for verification: frame times, event frames, and
    the sample index per frame.
    """
    import imageio_ffmpeg
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    t = bundle.t_s
    t_start = max(-pre_s, float(t[0]))
    t_end = min(bundle.events["hand stop"] + post_s, float(t[-1]))
    times = frame_times(t_start, t_end, fps, speed)
    samples = np.array([sample_at(t, x) for x in times])
    i0, i1 = sample_at(t, t_start), samples[-1]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.15, left=0.08, right=0.98, bottom=0.14, top=0.88)
    ax_eeg = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_topo = fig.add_axes((0.85, 0.66, 0.13, 0.2)) if topomap and bundle.topo is not None else None

    # Input panel: stacked traces over the whole window, a moving cursor, and fixed event markers.
    seg = slice(i0, i1 + 1)
    labels = list(bundle.eeg)
    # One amplitude scale for every trace (uncleaned and cleaned stay comparable); large blinks may leave their row.
    spread = float(np.median([np.percentile(np.abs(v[seg] - np.median(v[seg])), 98) for v in bundle.eeg.values()]))
    offset = 3.0 * spread
    for k, lab in enumerate(labels):
        v = bundle.eeg[lab][seg]
        ax_eeg.plot(t[seg], np.clip(v - np.median(v), -1.5 * offset, 1.5 * offset) - k * offset, lw=0.7,
                    color="tab:red" if "ICA" in lab else "0.15")
    ax_eeg.set_yticks([-k * offset for k in range(len(labels))], labels)
    ax_eeg.set_xlim(t_start, t_end)
    ax_eeg.set_ylim(-(len(labels) - 0.4) * offset, 0.8 * offset)
    for name, (a, b) in bundle.phases.items():  # phase bands, same colours as the ground truth
        ax_eeg.axvspan(a, b, color=PHASE_COLORS[name], alpha=0.08, lw=0)
    last_top = -np.inf
    y_bottom = -(len(labels) - 0.4) * offset
    for k, (lab, te) in enumerate(bundle.events.items()):
        if t_start <= te <= t_end:
            ax_eeg.axvline(te, color="tab:blue", lw=0.8, ls=":")
            top = te - last_top >= 0.35  # a label close to the previous one goes to the bottom edge
            ax_eeg.text(te, 0.8 * offset if top else y_bottom, lab, rotation=90, va="top" if top else "bottom",
                        ha="right", fontsize=7, color="tab:blue")
            last_top = te if top else last_top
    ax_eeg.set_xlabel("time since LED-on (s)")
    ax_eeg.set_title(f"Input: EEG 0.1-40 Hz, causal, CAR; trace spacing {offset:.0f} (cached units, likely uV); "
                     "red = ICA-cleaned", fontsize=9)
    cursor = ax_eeg.axvline(t_start, color="tab:red", lw=1.2)

    # Output panel: trajectories drawn up to the cursor.
    lims = axis_limits(bundle, i0, i1)
    ax3d.set(xlim=lims[0], ylim=lims[1], zlim=lims[2], xlabel="x (cm)", ylabel="y (cm)", zlabel="z (cm)")
    if view is None:
        touch = bundle.wrist[sample_at(t, bundle.events["first touch"])]
        view = (20.0, float(np.degrees(np.arctan2(touch[1], touch[0]))) - 90.0)
    ax3d.view_init(*view)
    ax3d.computed_zorder = False  # fixed draw order: ground truth above the predictions
    # Ground truth: one line per phase (shows the direction of movement), grey before onset and after hand stop.
    from matplotlib.lines import Line2D
    gt_colors = {"before": OUTSIDE_COLOR} | PHASE_COLORS | {"after": OUTSIDE_COLOR}
    gt_lines = {name: ax3d.plot(EMPTY, EMPTY, EMPTY, color=c, lw=GROUND_TRUTH_LW, solid_capstyle="round", zorder=4)[0]
                for name, c in gt_colors.items()}
    (gt_head,) = ax3d.plot(EMPTY, EMPTY, EMPTY, "o", color="k", ms=5, zorder=5)
    i_stop = sample_at(t, bundle.phases["return"][1])
    w0 = bundle.wrist[sample_at(t, bundle.phases["reach"][0])]
    w1 = bundle.wrist[i_stop]
    marker = dict(ms=10, mfc="white", mec="k", mew=2.0, ls="", zorder=6)
    ax3d.plot(w0[0:1], w0[1:2], w0[2:3], marker="o", **marker)  # start: wrist at onset, shown throughout
    (end_pt,) = ax3d.plot(EMPTY, EMPTY, EMPTY, marker="s", **marker)  # end: wrist at hand stop, once reached
    preds = bundle.preds
    lines, heads = {}, {}
    for k, (lab, arr) in enumerate(preds.items()):
        st = STYLE.get(lab, dict(color=PALETTE[(k - 1) % len(PALETTE)], lw=1.4))
        (lines[lab],) = ax3d.plot(EMPTY, EMPTY, EMPTY, label=lab, **st)
        (heads[lab],) = ax3d.plot(EMPTY, EMPTY, EMPTY, "o", color=st["color"], ms=4)
    (obj_pt,) = ax3d.plot(EMPTY, EMPTY, EMPTY, "s", color="k", alpha=0.45, ms=7, label="object (sensor 1)")
    pred_legend = ax3d.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 1.02), title="predictions",
                              title_fontsize=7)
    gt_handles = [Line2D([], [], color=PHASE_COLORS[p], lw=GROUND_TRUTH_LW, label=p) for p in PHASES]
    gt_handles += [Line2D([], [], color=OUTSIDE_COLOR, lw=GROUND_TRUTH_LW, label="before onset, after hand stop"),
                   Line2D([], [], marker="o", **(marker | dict(ms=7)), label="start (onset)"),
                   Line2D([], [], marker="s", **(marker | dict(ms=7)), label="end (hand stop)")]
    pos = ax3d.get_position()  # strip under the 3D panel, clear of its tick labels
    fig.legend(handles=gt_handles, loc="lower center", bbox_to_anchor=((pos.x0 + pos.x1) / 2, 0.0), ncol=4,
               fontsize=7, title="ground truth (wrist), by phase", title_fontsize=7, frameon=False)
    ax3d.add_artist(pred_legend)

    topo_vlim = None
    info = None
    if ax_topo is not None:
        import mne
        info = mne.create_info(bundle.eeg_names, 100.0, "eeg")
        info.set_montage("standard_1005")
        lim = float(np.percentile(np.abs(bundle.topo[:, seg]), 98))
        topo_vlim = (-lim, lim)

    title = fig.suptitle("", fontsize=11)

    def draw(f: int):
        i = int(samples[f])
        cursor.set_xdata([times[f], times[f]])
        pieces = phase_samples(t, bundle.phases, i0, i)
        for name, ln in gt_lines.items():
            a = bundle.wrist[pieces[name]] if name in pieces else np.empty((0, 3))
            ln.set_data_3d(a[:, 0], a[:, 1], a[:, 2])
        gt_head.set_data_3d(bundle.wrist[i, 0:1], bundle.wrist[i, 1:2], bundle.wrist[i, 2:3])
        end = w1 if i >= i_stop else np.full(3, np.nan)
        end_pt.set_data_3d(end[0:1], end[1:2], end[2:3])
        for lab, arr in preds.items():
            a = arr[i0:i + 1]
            lines[lab].set_data_3d(a[:, 0], a[:, 1], a[:, 2])
            last = arr[i] if np.all(np.isfinite(arr[i])) else np.full(3, np.nan)
            heads[lab].set_data_3d(last[0:1], last[1:2], last[2:3])
        obj_pt.set_data_3d(bundle.obj[i, 0:1], bundle.obj[i, 1:2], bundle.obj[i, 2:3])
        if ax_topo is not None:
            import mne
            ax_topo.clear()
            mne.viz.plot_topomap(bundle.topo[:, i], info, axes=ax_topo, show=False, contours=0, sensors=False, res=32,
                                 vlim=topo_vlim, cmap="RdBu_r")
            ax_topo.set_title("scalp EEG at the cursor (uncleaned)", fontsize=8)
        title.set_text(f"{bundle.subject} S{bundle.series} lift {bundle.lift} | {bundle.weight_g} g, {bundle.surface} | "
                       f"t = {times[f]:5.2f} s after LED-on | phase: {phase_at(bundle.phases, bundle.events, times[f])}")
        return []

    anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    return {"path": path.as_posix(), "n_frames": len(times), "fps": fps, "speed": speed, "t_start_s": t_start,
            "t_end_s": float(times[-1]), "frame_times_s": times, "frame_samples": samples,
            "event_frames": event_frames({k: v for k, v in bundle.events.items() if t_start <= v <= t_end}, t_start,
                                         fps, speed)}
