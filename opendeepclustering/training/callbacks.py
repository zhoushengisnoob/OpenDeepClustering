"""Minimal callbacks with no default filesystem side effects."""

from __future__ import annotations

from typing import Protocol

from opendeepclustering.training.state import FitState


class Callback(Protocol):
    """Observer interface for fit progress."""

    def on_event(self, event: str, estimator, state: FitState) -> None:
        """Observe a fit event without mutating estimator parameters."""


def emit(callbacks, event: str, estimator, state: FitState) -> None:
    for callback in callbacks or ():
        if hasattr(callback, "on_event"):
            callback.on_event(event, estimator, state)
        else:
            callback(event, estimator, state)
