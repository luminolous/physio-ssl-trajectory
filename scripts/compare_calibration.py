r"""Compare the calibration-within-subject runs: every condition vs the subject cue template, per k.

Reads the newest non-smoke results.csv of each calibration run (uncleaned, cleaned, prefrontal). Per subject the
draws are averaged first; comparisons are paired across the 10 subjects (Wilcoxon). Writes comparison.csv,
per_subject.csv, calibration_by_k.png, and summary.md to output/diagnostics/final-checks/calibration/summary/<run-id>/.

Usage (from repo root):
    .venv\Scripts\python scripts\compare_calibration.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eegtraj.diagnostics.calibration import METRICS, paired, per_subject

ROOT = Path("output/diagnostics/final-checks/calibration")
DATA = ("uncleaned", "cleaned", "prefrontal")
K_ORDER = ["10", "30", "all"]


def newest(name: str) -> Path | None:
    runs = sorted((ROOT / name).glob("2*/results.csv"))
    return runs[-1] if runs else None


def main() -> None:
    out = ROOT / "summary" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    tables, sources = [], {}
    for d in DATA:
        f = newest(d)
        if f is None:
            continue
        sources[d] = f.parent.as_posix()
        tables.append(per_subject(pd.read_csv(f, dtype={"k": str}), d))
    t = pd.concat(tables, ignore_index=True)
    # The subject template uses no EEG, so it is the same floor for every data variant.
    floor = ("uncleaned", "template")
    rows = []
    for k in K_ORDER:
        for d, c in [(d, c) for d in DATA for c in ("no_calibration", "finetune", "scratch")]:
            rows += paired(t, (d, c), floor, k)
        rows += paired(t, ("uncleaned", "finetune"), ("uncleaned", "scratch"), k)
        rows += paired(t, ("cleaned", "finetune"), ("cleaned", "scratch"), k)
        for c in ("finetune", "scratch", "no_calibration"):
            rows += paired(t, ("cleaned", c), ("uncleaned", c), k)
        rows += paired(t, ("prefrontal", "finetune"), ("uncleaned", "finetune"), k)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "comparison.csv", index=False)
    t.to_csv(out / "per_subject.csv", index=False)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))
    means = t.groupby(["data", "condition", "k"])[list(METRICS)].mean().reset_index()
    for ax, m in zip(axes, METRICS):
        for (d, c), g in means.groupby(["data", "condition"]):
            if c == "template" and d != "uncleaned":
                continue
            g = g.set_index("k").reindex([k for k in K_ORDER if k in set(g.k)])
            style = "--" if d == "cleaned" else (":" if d == "prefrontal" else "-")
            label = "subject template" if c == "template" else f"{d} {c}"
            ax.plot(g.index, g[m], style, marker="o", label=label, color="k" if c == "template" else None)
        ax.set_title(m)
        ax.set_xlabel("calibration lifts k")
    axes[0].legend(fontsize=7)
    fig.suptitle("Calibration within subject, test S7-S9, mean over 10 subjects (seed 1)")
    fig.tight_layout()
    fig.savefig(out / "calibration_by_k.png", dpi=150)
    plt.close(fig)

    vs_floor = comp[comp.b == "uncleaned/template"]
    lines = ["# Calibration within subject\n", "Sources: " + ", ".join(f"{d} `{p}`" for d, p in sources.items()) + "\n",
             "## Means per k (over subjects; draws averaged first)\n",
             means.round(3).to_markdown(index=False), "",
             "## Each condition vs the subject cue template\n", vs_floor.round(3).to_markdown(index=False), "",
             "## Other comparisons\n", comp[comp.b != "uncleaned/template"].round(3).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(means.round(3).to_string(index=False))
    print(comp.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
