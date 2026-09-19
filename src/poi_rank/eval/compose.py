"""`compose`: the ONLY writer of `results/metrics.json`.

Every pipeline stage writes its own `results/parts/<stage>.json`; this module merges them.
The old design had `evaluate` write `metrics.json` and later stages (`lodo`, ...) read-modify-
write the same file -- a lost-update hazard that clobbered results twice. One writer over
independent inputs cannot lose an update, and a part that is missing is reported as missing
rather than silently absent from the scorecard.

Layout of the composed file:
  * `evaluate.json` is spread at the top level (the evaluation harness payload -- systems,
    bias-gap table, personalization, ablations, ... -- keeps the keys downstream code/docs read).
  * every other part nests under its stem name (`lodo`, `gate_dgp`, `gate_representation`,
    `representation`, `decision_register`, ...).
  * `parts_present` lists exactly which parts were composed.

Deterministic: sorted keys, no timestamps, byte-identical for identical parts
(`tests/test_compose.py`, `make audit`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PARTS_DIRNAME = "parts"
METRICS_FILENAME = "metrics.json"
EVALUATE_PART = "evaluate"

# stem -> key in metrics.json. `evaluate` is spread flat (None). Parts absent from disk are
# skipped and named in `parts_present`; a REQUIRED part missing is an error.
COMPOSED_PARTS: dict[str, str | None] = {
    EVALUATE_PART: None,
    "train": "train",
    "lodo": "lodo",
    "dgp_gate": "gate_dgp",
    "dgp_diagnostics": "dgp_diagnostics",
    "gate_representation": "gate_representation",
    "representation": "representation",
    "decision_register": "decision_register",
    "seed_replication": "seed_replication",
}
REQUIRED_PARTS = (EVALUATE_PART,)


def parts_dir(results_dir: Path) -> Path:
    return results_dir / PARTS_DIRNAME


def write_part(results_dir: Path, stage: str, payload: dict[str, Any]) -> Path:
    """Write one stage's part file (deterministic JSON). Stages call this; only `compose_metrics`
    ever writes `metrics.json`."""
    directory = parts_dir(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def compose_metrics(results_dir: Path) -> Path:
    directory = parts_dir(results_dir)
    for stem in REQUIRED_PARTS:
        if not (directory / f"{stem}.json").exists():
            raise FileNotFoundError(
                f"required part results/{PARTS_DIRNAME}/{stem}.json is missing -- run the "
                f"`{stem}` stage before `compose`."
            )
    composed: dict[str, Any] = {}
    present: list[str] = []
    for stem, key in COMPOSED_PARTS.items():
        path = directory / f"{stem}.json"
        if not path.exists():
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        present.append(stem)
        if key is None:
            composed.update(content)
        else:
            composed[key] = content
    composed["parts_present"] = present
    out = results_dir / METRICS_FILENAME
    out.write_text(json.dumps(composed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
