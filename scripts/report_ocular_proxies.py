"""Ocular proxies (SSL-3): correlation of VEOG and HEOG from the CAR cache with the removed ICA ocular components.

Proxies: VEOG = mean(Fp1, Fp2), HEOG = F7 - F8, on the broadband CAR cache, lift windows plus 2 s of pre-lift
history. Components: the time courses of the components removed by the saved ICA (``ica_broadband_ica_car.npz``),
recovered from the uncleaned and cleaned caches (``eegtraj.preprocessing.ocular.sources_from_removed``). No labels
are used. A subject is flagged when the best |r| of a proxy over its components is below the threshold.

Writes to a new output/proposed-model/data/ocular-proxies/<run-id>/: correlations.csv (every proxy x component),
best.csv (best |r| per subject and proxy), summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\report_ocular_proxies.py [--subjects P1 P2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eegtraj.data.unlabeled import load_ica
from eegtraj.data.windows import input_channel_names, load_subject_samples
from eegtraj.preprocessing.ocular import PROXY_ORDER, proxy_rows, sources_from_removed
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/proposed-model/data/ocular-proxies")
CONFIG = {"cache_dir": "dataset/cache/98f5fa6e66", "band": "broadband", "cleaned_band": "broadband_ica",
          "reference": "car", "ica": "ica_broadband_ica_car.npz", "history": 200, "flag_below_abs_r": 0.5,
          "subjects": [f"P{i}" for i in range(1, 13)]}


def subject_table(cache_dir: Path, subject: str, cfg: dict) -> tuple[list[dict], float]:
    """Correlation of each proxy with each removed component of one subject, and the recovery residual."""
    kw = dict(channel_set="all", target="position", history=cfg["history"], pre_lift=True)
    raw = load_subject_samples(cache_dir, subject, cfg["band"], cfg["reference"], **kw)
    clean = load_subject_samples(cache_dir, subject, cfg["cleaned_band"], cfg["reference"], **kw)
    names = input_channel_names(cache_dir, subject, cfg["band"], "all")
    ica = load_ica(cache_dir / subject / cfg["ica"])
    if list(ica.info["names"]) != names:
        raise ValueError(f"{subject}: ICA channel order differs from the cache")
    proxies = proxy_rows(raw.eeg, names)
    src, resid = sources_from_removed(raw.eeg, clean.eeg, ica.mixing[:, ica.selected])
    rows = []
    for k, comp in enumerate(ica.selected):
        for p, name in enumerate(PROXY_ORDER):
            r = float(np.corrcoef(proxies[p], src[k])[0, 1])
            rows.append({"subject": subject, "component": comp, "proxy": name, "r": r, "abs_r": abs(r),
                         "fit_data_abs_r": float(ica.correlations[name][comp])})
    return rows, resid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    cfg = {**CONFIG, "subjects": args.subjects or CONFIG["subjects"]}
    run_dir = setup_run(OUTPUT, cfg)
    print(f"Run directory: {run_dir}")
    cache_dir = Path(cfg["cache_dir"])
    rows, resid = [], {}
    for s in cfg["subjects"]:
        r, resid[s] = subject_table(cache_dir, s, cfg)
        rows += r
    corr = pd.DataFrame(rows)
    corr.to_csv(run_dir / "correlations.csv", index=False)
    idx = corr.groupby(["subject", "proxy"]).abs_r.idxmax()
    best = corr.loc[idx].pivot(index="subject", columns="proxy", values="abs_r").reindex(cfg["subjects"])[list(PROXY_ORDER)]
    best["n_components"] = corr.groupby("subject").component.nunique().reindex(best.index)
    best["fit_residual"] = pd.Series(resid)
    thr = cfg["flag_below_abs_r"]
    best["flag"] = (best[list(PROXY_ORDER)] < thr).any(axis=1)
    best.to_csv(run_dir / "best.csv")
    flagged = best.index[best.flag].tolist()
    flagged_text = ", ".join(flagged) if flagged else "none"
    (run_dir / "summary.md").write_text(
        "# Ocular proxies vs ICA ocular components\n\n"
        "VEOG = mean(Fp1, Fp2), HEOG = F7 - F8 from the broadband CAR cache (lift windows + 2 s pre-lift history). "
        "Components: time courses of the removed ICA components, recovered from the uncleaned and cleaned caches. "
        f"Best |r| over each subject's removed components; flag when below {thr}. `fit_residual` checks the recovery "
        "(near 0 when the two caches differ only by the removed components).\n\n"
        + best.round(3).to_markdown() + f"\n\nFlagged: {flagged_text}.\n\n## All pairs\n\n"
        + corr.round(3).to_markdown(index=False) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"best_abs_r": best[list(PROXY_ORDER)].round(4).to_dict(orient="index"), "flagged": flagged,
                    "fit_residual": resid}, indent=2) + "\n", encoding="utf-8")
    print(best.round(3).to_string())
    print(f"Flagged: {flagged_text}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
