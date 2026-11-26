"""``residual_r_matched`` for decoder runs scored before design.md SSL-27.

Runs from before SSL-27 lack ``residual_r_matched`` in metrics.json. ``scripts/rescore_matched.py`` adds it from the
saved predictions and ``scripts/rescore_occlusion.py`` adds the occlusion drops from the saved fold models; summary
scripts read the newest table of each through these helpers. A run scored after SSL-27 has the value in metrics.json,
which takes precedence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MATCHED = Path("output/proposed-model/rescoring/matched")
OCCLUSION = Path("output/proposed-model/rescoring/occlusion")
KEY = "residual_r_matched"


def _newest(root: Path, name: str) -> Path | None:
    tables = sorted(root.glob(f"2*/{name}"))
    return tables[-1] if tables else None


def matched_scores() -> tuple[dict[tuple[str, str], float], str | None]:
    """{(run path posix, subject): residual_r_matched} from the newest rescoring table, and the table's path."""
    path = _newest(MATCHED, "scores.csv")
    if path is None:
        return {}, None
    t = pd.read_csv(path)
    return {(r, s): v for r, s, v in zip(t.run, t.subject, t[KEY])}, path.parent.as_posix()


def occlusion_drops() -> tuple[dict[tuple[str, str, str], float], str | None]:
    """{(run path posix, subject, group): drop in residual_r_matched} from the newest occlusion rescoring table."""
    path = _newest(OCCLUSION, "drops.csv")
    if path is None:
        return {}, None
    t = pd.read_csv(path)
    return {(r, s, g): v for r, s, g, v in zip(t.run, t.subject, t.group, t[f"drop_{KEY}"])}, path.parent.as_posix()


def subject_matched(run: Path, subject: str, scores: dict, table: dict) -> float:
    """``residual_r_matched`` of one test subject: metrics.json value if present, else the rescoring table."""
    if KEY in scores:
        return float(scores[KEY])
    try:
        return float(table[(Path(run).as_posix(), subject)])
    except KeyError:
        raise SystemExit(f"{run} {subject}: no residual_r_matched; run scripts/rescore_matched.py first") from None


ALPHA = 0.05


def verdicts(pairs: pd.DataFrame, index: list[str], old: str = "residual_r_specific", new: str = KEY) -> pd.DataFrame:
    """Old (rrs) against new (matched) reading of paired comparisons (design.md SSL-27).

    ``pairs``: long table with ``index`` columns, ``metric``, ``mean_diff``, ``n_pos``, ``p_wilcoxon``. A trial-specific
    difference counts as a finding only when its Wilcoxon p on ``residual_r_matched`` is below 0.05 (uncorrected, as
    in the summaries); significant on rrs only reads "not shown". ``changed``: the two metrics disagree on significance
    or, both significant, on the sign.
    """
    t = pairs[pairs.metric.isin([old, new])].pivot_table(index=index, columns="metric",
                                                         values=["mean_diff", "n_pos", "p_wilcoxon"], aggfunc="first")
    t.columns = [f"{v}_{'rrs' if m == old else 'matched'}" for v, m in t.columns]
    t = t.reset_index()
    sig_old, sig_new = t.p_wilcoxon_rrs < ALPHA, t.p_wilcoxon_matched < ALPHA
    sign = (t.mean_diff_rrs > 0) == (t.mean_diff_matched > 0)

    def word(sig: pd.Series, diff: pd.Series) -> pd.Series:
        return pd.Series(["finding (" + ("+" if d > 0 else "-") + ")" if s else "not shown" for s, d in zip(sig, diff)],
                         index=sig.index)

    t["reading_rrs"], t["reading_matched"] = word(sig_old, t.mean_diff_rrs), word(sig_new, t.mean_diff_matched)
    t["changed"] = (sig_old != sig_new) | (sig_old & sig_new & ~sign)
    return t
