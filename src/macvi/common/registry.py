"""Append-only JSONL registry of experiment results.

One line per finished experiment: ``{"id", "task", "config", "metrics", "artifacts", "time"}``.
JSONL keeps the history diff-friendly and safe to append from concurrent Colab sessions; appends
and reads hold ``<registry>.lock`` (two GPU lanes in one Kaggle session share one registry).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from macvi.common.filelock import locked


@dataclass
class Result:
    id: str
    task: str
    config: dict[str, Any]
    metrics: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)
    time: float = field(default_factory=time.time)


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, result: Result) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with locked(self.path), open(self.path, "a") as f:  # two GPU lanes may append at once
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    def all(self, task: str | None = None) -> list[Result]:
        if not self.path.exists():
            return []
        with locked(self.path, optional=True):  # never half a line from a concurrent append
            text = self.path.read_text()
        rows = [Result(**json.loads(line)) for line in text.splitlines() if line.strip()]
        return [r for r in rows if task is None or r.task == task]

    def ids(self, task: str | None = None) -> set[str]:
        return {r.id for r in self.all(task)}

    def best(self, task: str, metric: str, higher_is_better: bool = True) -> Result | None:
        rows = [r for r in self.all(task) if metric in r.metrics]
        if not rows:
            return None
        key = lambda r: r.metrics[metric]  # noqa: E731
        return max(rows, key=key) if higher_is_better else min(rows, key=key)
