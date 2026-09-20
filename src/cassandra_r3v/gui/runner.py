"""One-at-a-time background execution with cooperative global stop."""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable

from ..cancellation import CancellationToken, CancelledError


class TaskRunner:
    def __init__(self, root, event_callback: Callable[[str, Any], None]):
        self.root = root
        self.event_callback = event_callback
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.token: CancellationToken | None = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.title = ""
        self._poll_scheduled = False

    def start(self, title: str, worker: Callable, on_success: Callable[[Any], None] | None = None) -> bool:
        if self.running:
            return False
        self.running = True
        self.title = title
        self.token = CancellationToken()

        def progress(value: float, message: str) -> None:
            self.events.put(("progress", (float(value), str(message))))

        def log(message: str, tag: str = "") -> None:
            self.events.put(("log", (str(message), str(tag))))

        def target() -> None:
            try:
                result = worker(self.token, progress, log)
                self.events.put(("success", (result, on_success)))
            except CancelledError as exc:
                self.events.put(("cancelled", str(exc)))
            except Exception as exc:
                self.events.put(("error", (exc, traceback.format_exc())))
            finally:
                self.events.put(("finished", None))

        self.thread = threading.Thread(target=target, name=f"cassandra-{title}", daemon=True)
        self.thread.start()
        if not self._poll_scheduled:
            self._poll_scheduled = True
            self.root.after(80, self._poll)
        self.event_callback("started", title)
        return True

    def cancel(self) -> bool:
        if not self.running or self.token is None:
            return False
        self.token.cancel()
        self.event_callback("stop_requested", self.title)
        return True

    def _poll(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "success":
                    result, callback = payload
                    if callback:
                        callback(result)
                if event == "finished":
                    self.running = False
                    self.token = None
                    self.thread = None
                self.event_callback(event, payload)
        except queue.Empty:
            pass
        if self.running or not self.events.empty():
            self.root.after(80, self._poll)
        else:
            self._poll_scheduled = False
