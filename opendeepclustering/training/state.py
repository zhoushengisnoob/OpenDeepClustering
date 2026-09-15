"""Serializable state recorded during estimator fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FitState:
    """Mutable progress state shared with callbacks."""

    stage: str = "initializing"
    epoch: int = 0
    step: int = 0
    n_iter: int = 0
    converged: bool = False
    stop_requested: bool = False
    stop_reason: str | None = None
    history: dict[str, list[float]] = field(default_factory=dict)

    def record(self, name: str, value: float) -> None:
        self.history.setdefault(name, []).append(float(value))

    def request_stop(self, reason: str = "callback") -> None:
        """Request a cooperative stop from an epoch-end callback."""
        self.stop_requested = True
        self.stop_reason = reason

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
