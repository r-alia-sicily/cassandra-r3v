"""Small reusable widgets that make the Cassandra workflow self-explaining."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class HelpLabel(ttk.Frame):
    """Ordinary-looking label with a local, click-open explanation."""

    def __init__(self, parent, text: str, help_text: str, *, width: int = 48, **kwargs):
        super().__init__(parent, **kwargs)
        self.help_text = help_text
        self.width = width
        self.popup: tk.Toplevel | None = None
        self.label = ttk.Label(self, text=text, cursor="hand2")
        self.label.pack(side=tk.LEFT)
        self.info = ttk.Label(self, text="  ⓘ", cursor="hand2", style="Help.TLabel")
        self.info.pack(side=tk.LEFT)
        for widget in (self, self.label, self.info):
            widget.bind("<Button-1>", self._toggle, add=True)
            widget.bind("<Return>", self._toggle, add=True)
        self.info.configure(takefocus=True)

    def _toggle(self, _event=None):
        if self.popup and self.popup.winfo_exists():
            self._close()
            return
        self.popup = tk.Toplevel(self)
        self.popup.overrideredirect(True)
        self.popup.attributes("-topmost", True)
        shell = tk.Frame(self.popup, bg="#17324d", padx=1, pady=1)
        shell.pack(fill=tk.BOTH, expand=True)
        body = tk.Frame(shell, bg="#f7fbff", padx=12, pady=10)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            body,
            text=self.help_text,
            justify=tk.LEFT,
            wraplength=self.width * 8,
            bg="#f7fbff",
            fg="#17212b",
            font=("TkDefaultFont", 10),
        ).pack(anchor=tk.W)
        tk.Label(
            body,
            text="Click outside to close",
            bg="#f7fbff",
            fg="#607080",
            font=("TkDefaultFont", 8),
        ).pack(anchor=tk.E, pady=(8, 0))
        self.popup.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 4
        screen_width = self.winfo_screenwidth()
        popup_width = self.popup.winfo_reqwidth()
        if x + popup_width > screen_width - 20:
            x = max(10, screen_width - popup_width - 20)
        self.popup.geometry(f"+{x}+{y}")
        self.popup.bind("<FocusOut>", lambda _e: self._close(), add=True)
        self.popup.focus_set()

    def _close(self):
        if self.popup and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None


class RequirementBanner(ttk.Frame):
    def __init__(self, parent, go_to: Callable[[], None]):
        super().__init__(parent, style="Banner.TFrame", padding=(10, 8))
        self.message = ttk.Label(self, style="Banner.TLabel")
        self.message.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.button = ttk.Button(self, command=go_to)
        self.button.pack(side=tk.RIGHT, padx=(10, 0))

    def set_ready(self, text: str = "Requirements satisfied") -> None:
        self.message.configure(text=f"✓ {text}", foreground="#1d6b43")
        self.button.pack_forget()

    def set_missing(self, text: str, action: str) -> None:
        self.message.configure(text=f"Required step: {text}", foreground="#9a4b00")
        if not self.button.winfo_manager():
            self.button.pack(side=tk.RIGHT, padx=(10, 0))
        self.button.configure(text=action)


class LogPanel(ttk.Frame):
    def __init__(self, parent, height: int = 12):
        super().__init__(parent)
        self.text = tk.Text(
            self,
            height=height,
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=8,
            pady=8,
            background="#f7f8fa",
            foreground="#1f2933",
        )
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.tag_configure("success", foreground="#116530")
        self.text.tag_configure("warning", foreground="#9a4b00")
        self.text.tag_configure("error", foreground="#a61b1b")
        self.text.tag_configure("muted", foreground="#607080")
        self.text.configure(state=tk.DISABLED)

    def append(self, message: str, tag: str = "") -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, message.rstrip() + "\n", tag)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)


class StageStrip(ttk.Frame):
    STAGES = (
        ("data", "1  Data"),
        ("prepared", "2  Preparation"),
        ("model", "3  Model"),
        ("structure", "4  AutoStructure"),
        ("evaluation", "5  Evaluation"),
        ("full_test", "6  Full Test"),
    )

    def __init__(self, parent, on_select: Callable[[int], None]):
        super().__init__(parent)
        self.labels: dict[str, ttk.Label] = {}
        for index, (key, text) in enumerate(self.STAGES):
            label = ttk.Label(self, text=text, style="StageMissing.TLabel", cursor="hand2", padding=(9, 5))
            label.grid(row=0, column=index, padx=(0, 5), sticky="ew")
            label.bind("<Button-1>", lambda _event, value=index: on_select(value))
            self.labels[key] = label
            self.columnconfigure(index, weight=1)

    def update_states(self, states: dict[str, str]) -> None:
        for key, _ in self.STAGES:
            state = states.get(key, "missing")
            style = {
                "complete": "StageComplete.TLabel",
                "partial": "StagePartial.TLabel",
                "optional": "StageOptional.TLabel",
                "missing": "StageMissing.TLabel",
            }.get(state, "StageMissing.TLabel")
            self.labels[key].configure(style=style)


def labeled_entry(parent, row: int, text: str, variable, help_text: str, width: int = 10):
    label = HelpLabel(parent, text, help_text)
    label.grid(row=row, column=0, sticky="w", pady=3)
    entry = ttk.Entry(parent, textvariable=variable, width=width)
    entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
    return entry
