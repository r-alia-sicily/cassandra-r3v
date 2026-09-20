"""Cooperative cancellation shared by the GUI and the numerical engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event


class CancelledError(RuntimeError):
    """Raised at a safe checkpoint after the user requests cancellation."""


@dataclass
class CancellationToken:
    """Small thread-safe cancellation token.

    Numerical routines call :meth:`checkpoint` between indivisible operations.
    This allows Cassandra to stop without terminating the process or corrupting
    an artifact that is currently being written.
    """

    _event: Event = field(default_factory=Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self, message: str = "Processing stopped by the user") -> None:
        if self.cancelled:
            raise CancelledError(message)


NEVER_CANCEL = CancellationToken()
