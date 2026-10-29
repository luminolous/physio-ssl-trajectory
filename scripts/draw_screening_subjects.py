"""One-time seeded draw of the reserved screening subjects (decision D6).

Screening subjects are used for Exp 1 only and never appear as a test fold in
Exp 2. The draw uses ``numpy.random.default_rng(seed).choice`` over the subject
list in the config order, without replacement.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\draw_screening_subjects.py --config configs\\splits\\default.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from eegtraj.data.splits import draw_screening as draw

OUT_PATH = Path("output/data-baselines/splits/screening_draw.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/splits/default.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    chosen = draw(cfg["subjects"], cfg["n_screening"], cfg["seed"])
    record = {
        "drawn_at": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "subjects": cfg["subjects"],
        "n_screening": cfg["n_screening"],
        "method": "numpy.random.default_rng(seed).choice(len(subjects), size=n_screening, replace=False)",
        "numpy_version": np.__version__,
        "screening_subjects": chosen,
        "loso_subjects": [s for s in cfg["subjects"] if s not in chosen],
    }
    if OUT_PATH.exists():
        previous = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if previous["screening_subjects"] != chosen:
            raise RuntimeError(f"Re-draw differs from the recorded draw {previous['screening_subjects']}")
        print(f"Already drawn on {previous['drawn_at']}; re-draw matches.")
    else:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
