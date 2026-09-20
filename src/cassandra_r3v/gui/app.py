"""Main Cassandra desktop application."""

from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from importlib import resources
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .._version import __version__
from ..artifacts import ArtifactStore, atomic_json
from ..banner import print_banner
from ..config import PROTOCOLS, SUBSETS
from ..data import DatasetRepository
from .runner import TaskRunner
from .widgets import HelpLabel, StageStrip


class CassandraApp:
    def __init__(self, workspace: Path | None = None, root: tk.Tk | None = None):
        self.project_root = Path(__file__).resolve().parents[3]
        source_workspace = self.project_root / "cassandra_workspace"
        installed_workspace = Path.home() / "Cassandra_R3v" / "cassandra_workspace"
        default_workspace = source_workspace if (self.project_root / "pyproject.toml").is_file() else installed_workspace
        self.workspace = Path(workspace or default_workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.workspace / "settings.json"
        self.settings = self._load_settings()
        self.root = root or tk.Tk()
        self.root.title(f"Cassandra R3v {__version__} · Interpretable prognostics")
        self.root.geometry("1380x900")
        self.root.minsize(1080, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.store = ArtifactStore(self.workspace)
        self.repository = DatasetRepository(self.workspace)
        self.subset_vars = {
            subset: tk.BooleanVar(value=bool(self.settings.get("subsets", {}).get(subset, True)))
            for subset in SUBSETS
        }
        self.protocol_vars = {
            protocol: tk.BooleanVar(value=bool(self.settings.get("protocols", {}).get(protocol, True)))
            for protocol in PROTOCOLS
        }
        self._selection_guard = False
        self._configure_style()
        self.runner = TaskRunner(self.root, self._task_event)
        self.active_log = None
        self._build_ui()
        for variable in (*self.subset_vars.values(), *self.protocol_vars.values()):
            variable.trace_add("write", self._selection_changed)
        self.refresh_all()

    def _load_settings(self) -> dict[str, Any]:
        try:
            with self.settings_path.open("r", encoding="utf-8") as handle:
                values = json.load(handle)
            return values if isinstance(values, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_settings(self) -> None:
        atomic_json(
            self.settings_path,
            {
                "schema": 1,
                "subsets": {key: bool(value.get()) for key, value in self.subset_vars.items()},
                "protocols": {key: bool(value.get()) for key, value in self.protocol_vars.items()},
            },
        )

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")
        self.root.configure(background="#eef2f5")
        style.configure("TFrame", background="#eef2f5")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Banner.TFrame", background="#f8fbfd")
        style.configure("Banner.TLabel", background="#f8fbfd")
        style.configure("Title.TLabel", background="#17324d", foreground="#ffffff", font=("TkDefaultFont", 18, "bold"))
        style.configure("Subtitle.TLabel", background="#17324d", foreground="#d9e8f5", font=("TkDefaultFont", 10))
        style.configure("Header.TFrame", background="#17324d")
        style.configure("Help.TLabel", foreground="#1266a3")
        style.configure("Section.TLabel", font=("TkDefaultFont", 12, "bold"), foreground="#17324d")
        style.configure("StageComplete.TLabel", background="#d9f0e3", foreground="#155c38")
        style.configure("StagePartial.TLabel", background="#fff0c9", foreground="#7a4b00")
        style.configure("StageOptional.TLabel", background="#e6edf3", foreground="#506273")
        style.configure("StageMissing.TLabel", background="#f3f5f7", foreground="#77838e")
        style.configure("Danger.TButton", foreground="#8f1717")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("Treeview", rowheight=25)

    def _build_ui(self) -> None:
        self._build_menu()
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(16, 12))
        header.pack(fill=tk.X)
        title_block = ttk.Frame(header, style="Header.TFrame")
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_block, text="CASSANDRA", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            title_block,
            text=f"Rul3volution · Rule, Evolution, Stability · v{__version__}",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W)
        self.stop_button = ttk.Button(
            header,
            text="■  Stop processing",
            style="Danger.TButton",
            command=self.request_stop,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.RIGHT, padx=(12, 0))

        selection = ttk.Frame(self.root, padding=(14, 9))
        selection.pack(fill=tk.X)
        HelpLabel(
            selection,
            "Shared scope",
            "These selections apply simultaneously to every tab. Cassandra creates and "
            "keeps separate artifacts for each subset/protocol pair.",
        ).pack(side=tk.LEFT, padx=(0, 12))
        for subset in SUBSETS:
            ttk.Checkbutton(selection, text=subset, variable=self.subset_vars[subset]).pack(side=tk.LEFT, padx=3)
        ttk.Separator(selection, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Checkbutton(selection, text="Full RUL (uncapped)", variable=self.protocol_vars["uncapped"]).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(selection, text="RUL cap125", variable=self.protocol_vars["cap125"]).pack(side=tk.LEFT, padx=4)
        self.job_count_label = ttk.Label(selection)
        self.job_count_label.pack(side=tk.RIGHT)

        stage_shell = ttk.Frame(self.root, padding=(14, 0, 14, 8))
        stage_shell.pack(fill=tk.X)
        self.stage_strip = StageStrip(stage_shell, self.go_to_tab)
        self.stage_strip.pack(fill=tk.X)

        self.global_progress = ttk.Progressbar(self.root, mode="determinate", maximum=1.0)
        self.global_progress.pack(fill=tk.X, padx=14, pady=(0, 5))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
        from .tabs import AutoStructureTab, DataTab, EvaluateTab, FullTestTab, PrepareTab, TrainTab

        self.tabs = [
            DataTab(self.notebook, self),
            PrepareTab(self.notebook, self),
            TrainTab(self.notebook, self),
            AutoStructureTab(self.notebook, self),
            EvaluateTab(self.notebook, self),
            FullTestTab(self.notebook, self),
        ]
        for tab in self.tabs:
            self.notebook.add(tab.frame, text=tab.title)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self.refresh_all())

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN, padding=(8, 4)).pack(fill=tk.X)

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        self.root.configure(menu=menu)
        file_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open workspace", command=self.open_workspace)
        file_menu.add_command(label="Save settings", command=self.save_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        help_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Interface guide", command=lambda: self.open_document("GUI_GUIDE.md"))
        help_menu.add_command(label="Scientific alignment", command=lambda: self.open_document("METHOD_ALIGNMENT.md"))
        help_menu.add_separator()
        help_menu.add_command(label="About Cassandra", command=self.show_about)

    def selected_subsets(self) -> tuple[str, ...]:
        return tuple(key for key in SUBSETS if self.subset_vars[key].get())

    def selected_protocols(self) -> tuple[str, ...]:
        return tuple(key for key in PROTOCOLS if self.protocol_vars[key].get())

    def selected_jobs(self) -> tuple[tuple[str, str], ...]:
        return tuple((subset, protocol) for subset in self.selected_subsets() for protocol in self.selected_protocols())

    def _selection_changed(self, *_args) -> None:
        if self._selection_guard:
            return
        self._selection_guard = True
        try:
            if not self.selected_subsets():
                self.subset_vars["FD001"].set(True)
            if not self.selected_protocols():
                self.protocol_vars["uncapped"].set(True)
        finally:
            self._selection_guard = False
        self.save_settings()
        self.refresh_all()

    def go_to_tab(self, index: int) -> None:
        self.notebook.select(index)

    def raw_ready(self) -> bool:
        status = self.repository.status()["subsets"]
        return all(status[subset] for subset in self.selected_subsets())

    def all_jobs_have(self, stage: str) -> bool:
        jobs = self.selected_jobs()
        return bool(jobs) and all(self.store.stage_status(*job)[stage] for job in jobs)

    def refresh_all(self) -> None:
        jobs = self.selected_jobs()
        self.job_count_label.configure(text=f"{len(jobs)} separate experiments")
        raw = self.repository.status()["subsets"]
        data_done = sum(bool(raw[subset]) for subset in self.selected_subsets())
        counts = {stage: sum(self.store.stage_status(*job)[stage] for job in jobs) for stage in ("prepared", "model", "structure", "evaluation")}

        def state(done: int, total: int, optional: bool = False) -> str:
            if total and done == total:
                return "complete"
            if done:
                return "partial"
            return "optional" if optional else "missing"

        latest_report = self.store.latest_full_test_report()
        full_test_state = "optional"
        if latest_report is not None:
            try:
                report_text = latest_report.read_text(encoding="utf-8")
                report_lines = set(report_text.splitlines())
                subset_line = f"Selected subsets: {', '.join(self.selected_subsets())}"
                protocol_line = f"Selected protocols: {', '.join(self.selected_protocols())}"
                matching_scope = subset_line in report_lines and protocol_line in report_lines
                passing = "OVERALL STATUS: PASS" in report_text
                full_test_state = "complete" if matching_scope and passing else "partial"
            except OSError:
                full_test_state = "partial"

        self.stage_strip.update_states(
            {
                "data": state(data_done, len(self.selected_subsets())),
                "prepared": state(counts["prepared"], len(jobs)),
                "model": state(counts["model"], len(jobs)),
                "structure": state(counts["structure"], len(jobs), optional=bool(counts["model"])),
                "evaluation": state(counts["evaluation"], len(jobs)),
                "full_test": full_test_state,
            }
        )
        for tab in getattr(self, "tabs", []):
            tab.refresh()

    def start_task(self, title: str, tab, worker, on_success=None) -> bool:
        if self.runner.running:
            messagebox.showinfo(
                "Cassandra is busy",
                "Another operation is already running. You can stop it with the global button.",
            )
            return False
        self.active_log = getattr(tab, "log", None)
        started = self.runner.start(title, worker, on_success=on_success)
        if started:
            self.stop_button.configure(state=tk.NORMAL)
            self.global_progress.configure(value=0.0)
            self.refresh_all()
        return started

    def request_stop(self) -> None:
        self.runner.cancel()

    def _task_event(self, event: str, payload: Any) -> None:
        if event == "started":
            self.status_var.set(f"Running: {payload}")
        elif event == "progress":
            value, message = payload
            self.global_progress.configure(value=max(0.0, min(1.0, value)))
            self.status_var.set(message)
        elif event == "log" and self.active_log is not None:
            message, tag = payload
            self.active_log.append(message, tag)
        elif event == "stop_requested":
            self.status_var.set("Stop requested: Cassandra will stop at the next safe checkpoint")
            if self.active_log is not None:
                self.active_log.append("Stop requested; waiting for the current safe checkpoint.", "warning")
        elif event == "cancelled":
            self.status_var.set("Processing stopped without closing Cassandra")
            if self.active_log is not None:
                self.active_log.append(str(payload), "warning")
        elif event == "error":
            error, trace = payload
            self.status_var.set(f"Error: {error}")
            if self.active_log is not None:
                self.active_log.append(str(error), "error")
                self.active_log.append(trace, "muted")
            messagebox.showerror("Error", str(error))
        elif event == "finished":
            self.stop_button.configure(state=tk.DISABLED)
            self.global_progress.configure(value=0.0)
            if not self.status_var.get().startswith(("Error", "Processing stopped")):
                self.status_var.set("Processing completed")
            self.refresh_all()

    def open_workspace(self) -> None:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(self.workspace)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.workspace)])
            else:
                subprocess.Popen(["xdg-open", str(self.workspace)])
        except OSError as exc:
            messagebox.showerror("Unable to open workspace", str(exc))

    def open_document(self, name: str) -> None:
        path = self.project_root / "docs" / name
        if not path.is_file():
            try:
                text = resources.files("cassandra_r3v.resources").joinpath(name).read_text(encoding="utf-8")
                path = self.workspace / "help" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            except (OSError, FileNotFoundError):
                messagebox.showerror("Guide unavailable", f"{name} was not found in this installation.")
                return
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["notepad", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Unable to open document", str(exc))

    def show_about(self) -> None:
        messagebox.showinfo(
            f"Cassandra R3v {__version__}",
            "Cassandra is the first software implementation of the Rul3volution framework.\n\n"
            "This release keeps uncapped and cap125 separate, exposes the four quantities "
            "of every rule, and controls structural growth through evidence.\n\n"
            "License: GNU GPL v3 or later.",
        )

    def _on_close(self) -> None:
        if self.runner.running:
            if not messagebox.askyesno(
                "Processing in progress",
                "Cassandra is working. Request a stop and keep the application open?",
            ):
                return
            self.request_stop()
            return
        self.save_settings()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch(workspace: Path | None = None) -> None:
    print_banner()
    CassandraApp(workspace=workspace).run()
