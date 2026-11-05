"""Collect every finished baseline run of a stage into one summary table.

--stage data-baselines (default) reads metrics.json of every non-smoke run under
output/data-baselines/baselines/ and output/data-baselines/artifact-controls/.
--stage diagnostics reads output/data-baselines/baselines/ and output/diagnostics/{baselines,tuning,history}/,
and adds the residual metrics from the newest output/diagnostics/residual/<run-id>/residual.csv.
Runs of one configuration are grouped by
(group, name, tag without the seed part): several runs in a group are seeds, so per-subject
values are averaged over seeds first and the SD of the aggregate r across seeds is reported as
``r_seed_sd``. Paired per-subject differences are reported against the template and the
stereotypy floor.

Writes to output/<stage>/summary/:
    runs.csv         one row per run (aggregate metrics as logged)
    models.csv/.md   one row per configuration (seed-averaged)
    per_subject_r.csv  per-subject r (seed-averaged), one column per configuration
    per_subject_residual_r.csv, per_subject_residual_r_specific.csv  same for those metrics (diagnostics only)

Usage (from repo root):
    .venv\\Scripts\\python scripts\\summarize_runs.py [--stage diagnostics]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path("output")
# stage -> (source root, group) pairs
SOURCES = {
    "data-baselines": ((OUTPUT / "data-baselines", "baselines"), (OUTPUT / "data-baselines", "artifact-controls")),
    "diagnostics": ((OUTPUT / "data-baselines", "baselines"), (OUTPUT / "diagnostics", "baselines"),
                    (OUTPUT / "diagnostics", "tuning"), (OUTPUT / "diagnostics", "history")),
}
RESIDUAL_ROOT = OUTPUT / "diagnostics" / "residual"
RESIDUAL_METRICS = ("residual_r", "residual_r_oracle", "residual_r_other_trial", "residual_r_specific")
METRICS = ("r", "r2", "rmse", "nrmse")
FLOORS = ("template", "stereotypy", "template_cue")
RUN_ID = re.compile(r"^\d{8}-\d{6}(?:_(?P<tag>.+))?$")
SEED_TAG = re.compile(r"^seed\d+$")


def config_key(group: str, name: str, run_dir: Path) -> str:
    tag = RUN_ID.match(run_dir.name).group("tag") or ""
    variant = "" if SEED_TAG.match(tag) else tag
    return f"{name} ({variant})" if variant else name


def collect(stage: str) -> dict[tuple[str, str], list[tuple[Path, dict]]]:
    configs: dict[tuple[str, str], list[tuple[Path, dict]]] = {}
    for root, group in SOURCES[stage]:
        for f in sorted((root / group).glob("**/2*/metrics.json")):
            if not RUN_ID.match(f.parent.name):
                continue
            m = json.loads(f.read_text(encoding="utf-8"))
            if "baseline" not in m or m["smoke"]:  # other run types, such as cache builds
                continue
            name = f.parent.parent.relative_to(root / group).as_posix()
            key = (group, config_key(group, name, f.parent))
            configs.setdefault(key, []).append((f.parent, m))
    return configs


def run_row(group: str, config: str, run_dir: Path, m: dict) -> dict:
    info = m["per_fold"][0]["baseline_info"]
    row = {"group": group, "config": config, "run": run_dir.as_posix(), "seed": m["seed"],
           "split_mode": m["split_mode"], "n_subjects": m["aggregate"]["r"]["n_subjects"],
           "wall_seconds": m.get("wall_seconds")}
    for k in METRICS:
        row[f"{k}_mean"], row[f"{k}_sd"] = m["aggregate"][k]["mean"], m["aggregate"][k]["sd"]
    row["pooled_r_secondary"] = float(np.mean([v["mean"] for v in m["pooled_r_secondary"].values()]))
    row["n_parameters"] = info.get("n_parameters")
    row["oracle_timing"] = bool(info.get("oracle_timing", False))
    row["cue_timing"] = bool(info.get("cue_timing", False))
    return row


def per_subject(runs: list[tuple[Path, dict]], metric: str) -> pd.Series:
    """Per-subject value of ``metric`` averaged over the runs (seeds) of one configuration."""
    frame = pd.DataFrame([{s: v[metric]["mean"] for s, v in m["per_subject"].items()} for _, m in runs])
    return frame.mean(axis=0)


def load_residuals() -> pd.DataFrame | None:
    """The newest residual table, or None if it was not computed yet."""
    tables = sorted(RESIDUAL_ROOT.glob("2*/residual.csv"))
    return pd.read_csv(tables[-1]) if tables else None


def residual_per_subject(table: pd.DataFrame, run_names: list[str], metric: str) -> pd.Series:
    """Per-subject value of a residual metric averaged over the given runs (seeds)."""
    rows = table[table.run.isin(run_names)]
    return rows.pivot(index="run", columns="subject", values=metric).mean(axis=0)


def model_row(group: str, config: str, runs: list[tuple[Path, dict]], floors: dict[str, pd.Series],
              residuals: pd.DataFrame | None = None) -> dict:
    rows = [run_row(group, config, d, m) for d, m in runs]
    row = {"group": group, "config": config, "n_runs": len(runs), "split_mode": rows[0]["split_mode"],
           "n_parameters": rows[0]["n_parameters"], "oracle_timing": rows[0]["oracle_timing"],
           "cue_timing": rows[0]["cue_timing"]}
    for k in METRICS:
        v = per_subject(runs, k)
        row[f"{k}_mean"], row[f"{k}_sd"] = float(v.mean()), float(v.std(ddof=1))
    aggs = [r["r_mean"] for r in rows]
    row["r_seed_sd"] = float(np.std(aggs, ddof=1)) if len(aggs) > 1 else float("nan")
    row["pooled_r_secondary"] = float(np.mean([r["pooled_r_secondary"] for r in rows]))
    r = per_subject(runs, "r")
    for floor, fr in floors.items():
        diff = (r - fr.reindex(r.index)).dropna()
        row[f"delta_r_vs_{floor}"] = float(diff.mean())
        row[f"n_above_{floor}"] = int((diff > 0).sum())
    if residuals is not None:
        names = [d.as_posix() for d, _ in runs]
        for metric in RESIDUAL_METRICS:
            v = residual_per_subject(residuals, names, metric)
            row[f"{metric}_mean"] = float(v.mean()) if v.notna().any() else float("nan")
            row[f"{metric}_sd"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else float("nan")
    row["runs"] = ", ".join(d.as_posix() for d, _ in runs)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=sorted(SOURCES), default="data-baselines")
    stage = parser.parse_args().stage
    out_dir = OUTPUT / stage / "summary"
    configs = collect(stage)
    floors = {f: per_subject(configs[("baselines", f)], "r") for f in FLOORS if ("baselines", f) in configs}
    runs = pd.DataFrame([run_row(g, c, d, m) for (g, c), rs in configs.items() for d, m in rs])
    residuals = load_residuals() if stage == "diagnostics" else None
    models = pd.DataFrame([model_row(g, c, rs, floors, residuals) for (g, c), rs in configs.items()])
    models = models.sort_values(["group", "r_mean"], ascending=[False, False])
    models["n_parameters"] = models["n_parameters"].astype("Int64")
    subj = pd.DataFrame({f"{g}/{c}": per_subject(rs, "r") for (g, c), rs in configs.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "runs.csv", index=False)
    models.to_csv(out_dir / "models.csv", index=False)
    subj.to_csv(out_dir / "per_subject_r.csv", index_label="subject")
    if residuals is not None:
        for metric in ("residual_r", "residual_r_specific"):
            pd.DataFrame({f"{g}/{c}": residual_per_subject(residuals, [d.as_posix() for d, _ in rs], metric)
                          for (g, c), rs in configs.items()}).to_csv(out_dir / f"per_subject_{metric}.csv", index_label="subject")

    cols = ["group", "config", "n_runs", "oracle_timing", "cue_timing", "n_parameters", "r_mean", "r_sd", "r_seed_sd",
            "r2_mean", "nrmse_mean", "residual_r_mean", "residual_r_sd", "residual_r_oracle_mean", "residual_r_other_trial_mean", "residual_r_specific_mean", "residual_r_specific_sd", "pooled_r_secondary", "delta_r_vs_template", "n_above_template",
            "delta_r_vs_stereotypy", "n_above_stereotypy", "delta_r_vs_template_cue", "n_above_template_cue"]
    md = models[[c for c in cols if c in models]].to_markdown(index=False, floatfmt=".3f")
    run_list = "\n".join(f"- `{c}`: {r}" for c, r in zip(models["group"] + "/" + models["config"], models["runs"]))
    (out_dir / "models.md").write_text(
        f"# {stage} summary\n\n"
        "LOSO over the 10 non-screening subjects. Per-trial Pearson r per axis, mean over trials, axes, then "
        "subjects (D7); +- SD across subjects. With n_runs > 1 (seeds) per-subject values are averaged over "
        "seeds first; r_seed_sd = SD of the aggregate r across seeds. pooled_r_secondary is pooled over "
        "concatenated trials and inflated; secondary only. delta_r_vs_X = mean over subjects of (r - r of X); "
        "n_above_X = subjects where r > r of X. oracle_timing = uses the test trial's kinematic onset. "
        "cue_timing = uses the test trial's LED-on time, which a real decoder can know. "
        "residual_r = per-trial r after subtracting the cue-locked template from prediction and target; NaN for "
        "the cue template itself. residual_r_oracle = same with the oracle-aligned template; analysis only. "
        "residual_r_other_trial = control pairing each predicted residual with another trial of the subject. "
        "residual_r_specific = residual_r minus that control.\n\n"
        + md + "\n\n## Run directories\n\n" + run_list + "\n", encoding="utf-8")
    print(md)
    print(f"\n{len(runs)} runs, {len(models)} configurations -> {out_dir}")


if __name__ == "__main__":
    main()
