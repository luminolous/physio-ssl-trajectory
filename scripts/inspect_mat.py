"""Dump the structure of WAY-EEG-GAL .mat files for one subject.

Opens one WS file, one HS file, the HS ST file, and the AllLifts file, and writes
a tree of variables (shape, dtype, value range, short values) to a text report.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\inspect_mat.py --subject P1 --series S1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io

RAW_DIR = Path("dataset/raw")
OUT_DIR = Path("output/data-baselines/data-inspection")
MAX_LIST_ITEMS = 2  # list/cell elements expanded in full; the rest are only summarized
MAX_STR_ITEMS = 80  # string-array items printed in full (e.g. channel names)
MAX_DEPTH = 8


def describe_array(arr: np.ndarray) -> str:
    desc = f"ndarray shape={arr.shape} dtype={arr.dtype}"
    if arr.size == 0:
        return desc + " (empty)"
    if arr.dtype.kind in "biuf":
        finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
        n_nan = arr.size - finite.size
        if finite.size:
            desc += f" min={finite.min():.6g} max={finite.max():.6g}"
        if n_nan:
            desc += f" n_nonfinite={n_nan}"
        if arr.size <= 12:
            desc += f" values={np.array2string(arr.ravel(), precision=6, max_line_width=200)}"
    elif arr.dtype.kind in "US":
        items = [str(x) for x in arr.ravel()[:MAX_STR_ITEMS]]
        desc += f" values={items}"
    return desc


def signature(obj) -> str:
    """Short structural signature used to check whether list elements share a layout."""
    if isinstance(obj, dict):
        return "dict(" + ",".join(obj.keys()) + ")"
    if isinstance(obj, np.ndarray):
        return f"ndarray{obj.shape}{obj.dtype}"
    return type(obj).__name__


def walk(obj, name: str, depth: int, lines: list[str]) -> None:
    pad = "  " * depth
    if depth > MAX_DEPTH:
        lines.append(f"{pad}{name}: ... (max depth)")
        return
    if isinstance(obj, dict):
        lines.append(f"{pad}{name}: struct with {len(obj)} fields")
        for key, val in obj.items():
            walk(val, key, depth + 1, lines)
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == object):
        items = list(obj.ravel()) if isinstance(obj, np.ndarray) else list(obj)
        if items and all(isinstance(x, str) for x in items):
            lines.append(f"{pad}{name}: cell of {len(items)} str")
            for i, s in enumerate(items[:MAX_STR_ITEMS]):
                lines.append(f"{pad}  [{i}] {s}")
            return
        sigs = {signature(x) for x in items}
        lines.append(f"{pad}{name}: list/cell len={len(items)}, distinct element layouts={len(sigs)}")
        for i, item in enumerate(items[:MAX_LIST_ITEMS]):
            walk(item, f"[{i}]", depth + 1, lines)
        if len(items) > MAX_LIST_ITEMS:
            lines.append(f"{pad}  ... {len(items) - MAX_LIST_ITEMS} more elements")
            if len(sigs) > 1:
                lines.append(f"{pad}  layouts: {sorted(sigs)[:5]}")
            # For lists of numeric arrays, summarize the shape spread (e.g. window lengths).
            shapes = [x.shape for x in items if isinstance(x, np.ndarray)]
            if shapes and len(set(shapes)) > 1:
                lens = np.array([s[0] if s else 0 for s in shapes])
                lines.append(f"{pad}  first-dim lengths: min={lens.min()} max={lens.max()}")
    elif isinstance(obj, np.ndarray):
        lines.append(f"{pad}{name}: {describe_array(obj)}")
    elif isinstance(obj, (str, bytes)):
        lines.append(f"{pad}{name}: str {obj!r}")
    else:
        lines.append(f"{pad}{name}: {type(obj).__name__} {obj!r}")


def mat_version(path: Path) -> str:
    with open(path, "rb") as f:
        header = f.read(128)
    text = header[:116].decode("latin-1").strip("\x00 ")
    return text


def inspect_file(path: Path) -> list[str]:
    lines = [f"=== {path} ({path.stat().st_size / 1e6:.1f} MB)", f"header: {mat_version(path)}"]
    data = scipy.io.loadmat(path, simplify_cells=True)
    for key, val in data.items():
        if key.startswith("__"):
            continue
        walk(val, key, 0, lines)
    lines.append("")
    return lines


def load_alllifts(subject: str) -> tuple[np.ndarray, list[str]]:
    p = scipy.io.loadmat(RAW_DIR / subject / f"{subject}_AllLifts.mat", simplify_cells=True)["P"]
    return p["AllLifts"], list(p["ColNames"])


def summarize_alllifts(subject: str) -> list[str]:
    table, cols = load_alllifts(subject)
    lines = [f"=== AllLifts column summary ({subject}, {table.shape[0]} rows)"]
    for j, name in enumerate(cols):
        col = table[:, j]
        finite = col[np.isfinite(col)]
        lines.append(
            f"  {j:2d} {name:<20s} min={finite.min():9.4g} max={finite.max():9.4g} "
            f"n_nan={int(np.isnan(col).sum()):3d} n_unique={len(np.unique(finite)):4d}"
        )
    lines.append("")
    return lines


def summarize_trial_counts() -> list[str]:
    """Lifts per subject and series (Run) from every AllLifts file (small files only)."""
    lines = ["=== Lifts per subject x series (from AllLifts)"]
    for subj_dir in sorted(RAW_DIR.glob("P*"), key=lambda p: int(p.name[1:])):
        table, cols = load_alllifts(subj_dir.name)
        runs, counts = np.unique(table[:, cols.index("Run")], return_counts=True)
        blocks = {int(r): sorted({int(b) for b in table[table[:, cols.index("Run")] == r, cols.index("BlockType")]})
                  for r in runs}
        per_run = " ".join(f"S{int(r)}={c}" for r, c in zip(runs, counts))
        lines.append(f"  {subj_dir.name:<4s} total={table.shape[0]:4d} {per_run}  BlockType per series={blocks}")
    lines.append("")
    return lines


def summarize_ws(subject: str, series: str) -> list[str]:
    """Compare WS window metadata with AllLifts, sensor motion, and condition codes."""
    ws = scipy.io.loadmat(RAW_DIR / subject / f"WS_{subject}_{series}.mat", simplify_cells=True)["ws"]
    table, cols = load_alllifts(subject)
    rows = table[table[:, cols.index("Run")] == int(series[1:])]
    c = cols.index
    lines = [f"=== WS {subject} {series}: {len(ws['win'])} windows, AllLifts rows for this series: {len(rows)}"]

    # Timing: WS fields vs AllLifts, plus effective sampling rates.
    d_start = [w["trial_start_time"] - r[c("StartTime")] for w, r in zip(ws["win"], rows)]
    d_ledoff = [w["LEDoff"] - r[c("LEDOff")] for w, r in zip(ws["win"], rows)]
    durs = np.array([w["trial_end_time"] - w["trial_start_time"] for w in ws["win"]])
    last_t = np.array([w["eeg_t"][-1] for w in ws["win"]])
    fs_eeg = [1 / np.median(np.diff(w["eeg_t"])) for w in ws["win"]]
    fs_emg = [1 / np.median(np.diff(w["emg_t"])) for w in ws["win"]]
    lines += [
        f"  max |trial_start_time - StartTime| = {np.max(np.abs(d_start)):.2e} s",
        f"  max |LEDoff(ws) - LEDOff(AllLifts)| = {np.max(np.abs(d_ledoff)):.2e} s",
        f"  window duration (end-start) s: min={durs.min():.3f} max={durs.max():.3f}; "
        f"max |duration - eeg_t[-1] + 0.002| = {np.max(np.abs(durs - last_t + 0.002)):.2e}",
        f"  EEG fs from eeg_t: {np.min(fs_eeg):.2f}-{np.max(fs_eeg):.2f} Hz; EMG fs from emg_t: "
        f"{np.min(fs_emg):.2f}-{np.max(fs_emg):.2f} Hz",
        f"  window overlap with next window (s): "
        f"{[round(ws['win'][i]['trial_end_time'] - ws['win'][i + 1]['trial_start_time'], 3) for i in range(3)]}",
        "  event order (tHandStart < tFirstDigitTouch < tLiftOff < tReplace < tHandStop) holds for "
        f"{int(np.sum(np.all(np.diff(rows[:, [c('tHandStart'), c('tFirstDigitTouch'), c('tLiftOff'), c('tReplace'), c('tHandStop')]], axis=1) > 0, axis=1)))}"
        f"/{len(rows)} rows",
    ]

    # Kinematic sensors: displacement during reach vs during lift identifies object vs hand sensors.
    names = list(ws["names"]["kin"])
    nan_cols = sorted({names[j].split(" -")[0] for w in ws["win"] for j in range(w["kin"].shape[1])
                       if np.isnan(w["kin"][:, j]).any()})
    lines.append(f"  kin columns containing NaN in any window: {nan_cols}")
    reach_disp = {s: [] for s in range(1, 5)}
    lift_disp = {s: [] for s in range(1, 5)}
    pos = {s: [names.index(f"P{a}{s} - position {a} sensor {s}") for a in "xyz"] for s in range(1, 5)}
    for w, r in zip(ws["win"], rows):
        t = w["eeg_t"]
        for s in range(1, 5):
            p = w["kin"][:, pos[s]]
            reach = p[(t >= r[c("tHandStart")]) & (t <= r[c("tFirstDigitTouch")])]
            hold = p[(t >= r[c("tLiftOff")]) & (t <= r[c("tReplace")])]
            reach_disp[s].append(np.linalg.norm(reach[-1] - reach[0]))
            lift_disp[s].append(np.ptp(hold[:, 2]))
    for s in range(1, 5):
        lines.append(f"  sensor {s}: median 3D displacement during reach = {np.median(reach_disp[s]):6.2f}, "
                     f"median z range lift->replace = {np.median(lift_disp[s]):5.2f} (kin units)")
    aperture = [np.nanmax(np.linalg.norm(w["kin"][:, pos[2]] - w["kin"][:, pos[3]], axis=1)) for w in ws["win"]]
    lines.append(f"  max |sensor2 - sensor3| distance per window vs AllLifts GripAparture_Max: "
                 f"max abs diff = {np.max(np.abs(np.array(aperture) - rows[:, c('GripAparture_Max')])):.3f}")

    # Condition codes.
    weights = sorted({(int(w["weight"]), str(w["weight_id"])) for w in ws["win"]})
    surfs = sorted({(int(w["surf"]), str(w["surf_id"])) for w in ws["win"]})
    lines += [f"  weight codes: {weights}", f"  surface codes: {surfs}", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default="P1")
    parser.add_argument("--series", default="S1")
    parser.add_argument("--summary-series", nargs="*", default=None,
                        help="series for the WS summary section (default: all S1..S9 of the subject)")
    args = parser.parse_args()

    subj_dir = RAW_DIR / args.subject
    files = [
        subj_dir / f"WS_{args.subject}_{args.series}.mat",
        subj_dir / f"HS_{args.subject}_{args.series}.mat",
        subj_dir / f"HS_{args.subject}_ST.mat",
        subj_dir / f"{args.subject}_AllLifts.mat",
    ]
    lines: list[str] = []
    for path in files:
        lines.extend(inspect_file(path))
    lines.extend(summarize_alllifts(args.subject))
    lines.extend(summarize_trial_counts())
    for series in args.summary_series or [f"S{i}" for i in range(1, 10)]:
        lines.extend(summarize_ws(args.subject, series))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"inspect_mat_{args.subject}_{args.series}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
