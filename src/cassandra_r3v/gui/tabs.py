"""The six workflow and diagnostic tabs of Cassandra's desktop interface."""

from __future__ import annotations

import json
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..artifacts import atomic_json, utc_now
from ..autostructure import (
    ComputeBudgetExceeded,
    estimate_pairwise_candidates,
    grow_structure,
)
from ..config import AutoStructureConfig, ComputeBudget, TrainingConfig
from ..data import prepare_subset
from ..diagnostics import FullTestResult, run_full_test
from ..experiment import compare_prepared_models, evaluate_prepared, explain_test_engine, save_comparison, train_prepared
from ..model import R3vModel
from .widgets import HelpLabel, LogPanel, RequirementBanner, labeled_entry


class BaseTab:
    title = ""

    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent, padding=12)
        self._build()

    def _heading(self, title: str, subtitle: str) -> None:
        ttk.Label(self.frame, text=title, style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(self.frame, text=subtitle, foreground="#506273", wraplength=1120).pack(anchor=tk.W, pady=(2, 9))

    def refresh(self) -> None:
        pass


class DataTab(BaseTab):
    title = "1 · Data"

    def _build(self) -> None:
        self._heading(
            "Dataset NASA C-MAPSS",
            "Import a previously downloaded NASA archive or download it from the official source. "
            "The code does not redistribute the data.",
        )
        card = ttk.LabelFrame(self.frame, text="Source and availability", padding=10)
        card.pack(fill=tk.X)
        path_row = ttk.Frame(card)
        path_row.pack(fill=tk.X)
        HelpLabel(
            path_row,
            "Local data folder",
            "The twelve TRAIN, TEST, and RUL files are extracted here. Experiments read only this local copy.",
        ).pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value=str(self.app.repository.raw_dir))
        ttk.Entry(path_row, textvariable=self.path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        buttons = ttk.Frame(card)
        buttons.pack(fill=tk.X, pady=(10, 5))
        self.import_button = ttk.Button(
            buttons,
            text="Import NASA archive…",
            style="Accent.TButton",
            command=self._choose_archive,
        )
        self.import_button.pack(side=tk.LEFT)
        self.download_button = ttk.Button(buttons, text="Download from NASA", command=self._download)
        self.download_button.pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Open workspace", command=self.app.open_workspace).pack(side=tk.LEFT)

        columns = ("subset", "train", "test", "rul", "status")
        self.tree = ttk.Treeview(card, columns=columns, show="headings", height=4)
        for key, label, width in (
            ("subset", "Subset", 100),
            ("train", "TRAIN", 120),
            ("test", "TEST", 120),
            ("rul", "RUL endpoint", 140),
            ("status", "Status", 220),
        ):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(self.frame, text="Log", style="Section.TLabel").pack(anchor=tk.W, pady=(12, 4))
        self.log = LogPanel(self.frame, height=13)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.append("Cassandra accepts both NASA's outer archive and the inner CMAPSSData.zip file.", "muted")

    def _choose_archive(self) -> None:
        path = filedialog.askopenfilename(
            title="Select the NASA C-MAPSS archive",
            filetypes=[("ZIP archives", "*.zip"), ("All files", "*")],
        )
        if path:
            self._import(Path(path))

    def _import(self, path: Path) -> None:
        self.log.clear()

        def worker(token, progress, log):
            log(f"Importing from {path}")
            result = self.app.repository.import_archive(path, token=token, progress=progress)
            log(f"Archive verified: {result['extracted']} useful files extracted.", "success")
            return result

        self.app.start_task("dataset import", self, worker)

    def _download(self) -> None:
        if not messagebox.askyesno(
            "Download dataset",
            "Download the C-MAPSS archive from the official NASA source?",
        ):
            return
        self.log.clear()

        def worker(token, progress, log):
            log("Connecting to the official NASA source…")
            result = self.app.repository.download_official(token=token, progress=progress)
            log("Download, verification, and extraction completed.", "success")
            return result

        self.app.start_task("dataset download", self, worker)

    def refresh(self) -> None:
        if not hasattr(self, "tree"):
            return
        selected = set(self.app.selected_subsets())
        status = self.app.repository.status()
        self.tree.delete(*self.tree.get_children())
        for subset, ready in status["subsets"].items():
            mark = "✓" if ready else "—"
            label = "available" if ready else "missing"
            tags = ("selected",) if subset in selected else ()
            self.tree.insert("", tk.END, values=(subset, mark, mark, mark, label), tags=tags)
        self.tree.tag_configure("selected", background="#edf6fc")
        state = tk.DISABLED if self.app.runner.running else tk.NORMAL
        self.import_button.configure(state=state)
        self.download_button.configure(state=state)


class PrepareTab(BaseTab):
    title = "2 · Preparation"

    def _build(self) -> None:
        self._heading(
            "Leakage-safe preparation",
            "Standardization and fuzzy partitions use TRAIN only. Parameters are then estimated "
            "on five pseudo-endpoints per engine.",
        )
        self.banner = RequirementBanner(self.frame, lambda: self.app.go_to_tab(0))
        self.banner.pack(fill=tk.X, pady=(0, 9))
        card = ttk.LabelFrame(self.frame, text="Experimental contract", padding=10)
        card.pack(fill=tk.X)
        HelpLabel(
            card,
            "Selected protocols",
            "Uncapped retains full RUL; cap125 applies min(RUL, 125). Results are always saved "
            "in separate rows and files.",
        ).grid(row=0, column=0, sticky="w")
        self.scope_var = tk.StringVar()
        ttk.Label(card, textvariable=self.scope_var, foreground="#1266a3").grid(row=0, column=1, sticky="w", padx=10)
        HelpLabel(
            card,
            "TRAIN-only pseudo-endpoints",
            "For each engine, Cassandra samples 35%, 50%, 65%, 80%, and 92% of its observed life. "
            "TEST data do not determine preprocessing or structure.",
        ).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Label(card, text="35% · 50% · 65% · 80% · 92%").grid(row=1, column=1, sticky="w", padx=10)
        HelpLabel(
            card,
            "Physical inputs",
            "The 21 sensors and 3 operating conditions use canonical names. Cycle and RUL are not model inputs.",
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(card, text="24 channels · no temporal gradients").grid(row=2, column=1, sticky="w", padx=10)

        self.run_button = ttk.Button(
            self.frame,
            text="Prepare selected experiments",
            style="Accent.TButton",
            command=self._run,
        )
        self.run_button.pack(anchor=tk.W, pady=10)
        self.tree = self._status_tree(self.frame)
        self.tree.pack(fill=tk.X)
        ttk.Label(self.frame, text="Log", style="Section.TLabel").pack(anchor=tk.W, pady=(12, 4))
        self.log = LogPanel(self.frame, height=11)
        self.log.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _status_tree(parent):
        tree = ttk.Treeview(parent, columns=("subset", "protocol", "status"), show="headings", height=8)
        for key, text, width in (
            ("subset", "Subset", 130),
            ("protocol", "Protocol", 180),
            ("status", "Preparation", 380),
        ):
            tree.heading(key, text=text)
            tree.column(key, width=width, anchor=tk.CENTER if key != "status" else tk.W)
        return tree

    def _run(self) -> None:
        jobs = self.app.selected_jobs()
        self.log.clear()

        def worker(token, progress, log):
            outputs = []
            for index, (subset, protocol) in enumerate(jobs):
                token.checkpoint()
                log(f"[{subset} · {protocol}] loading and preparing")
                dataset = prepare_subset(self.app.repository.raw_dir, subset, protocol, token=token)
                archived = self.app.store.invalidate_after("prepared", subset, protocol)
                path = self.app.store.save_prepared(dataset)
                log(
                    f"[{subset} · {protocol}] {dataset.metadata['n_fit_pseudo_endpoints']} pseudo-endpoints, "
                    f"{dataset.metadata['n_test_engines']} TEST engines · {path}",
                    "success",
                )
                if archived:
                    log(f"Previous dependent artifacts archived: {len(archived)}", "warning")
                outputs.append(str(path))
                progress((index + 1) / len(jobs), f"Prepared {subset} · {protocol}")
            return outputs

        self.app.start_task("experiment preparation", self, worker)

    def refresh(self) -> None:
        if not hasattr(self, "tree"):
            return
        protocols = ", ".join(self.app.selected_protocols())
        self.scope_var.set(protocols)
        ready = self.app.raw_ready()
        if ready:
            self.banner.set_ready("raw data are available for every selected subset")
        else:
            self.banner.set_missing("import the missing raw data", "Go to Data")
        self.tree.delete(*self.tree.get_children())
        for subset, protocol in self.app.selected_jobs():
            done = self.app.store.prepared_path(subset, protocol).is_file()
            self.tree.insert("", tk.END, values=(subset, protocol, "✓ ready" if done else "not prepared"))
        self.run_button.configure(state=tk.NORMAL if ready and not self.app.runner.running else tk.DISABLED)


class TrainTab(BaseTab):
    title = "3 · Model"

    def _build(self) -> None:
        self._heading(
            "R3v parametric learning",
            "Centered, bounded Ridge initializes the consequents; Adam alternates between consequents "
            "and log-authorities on the declared loss.",
        )
        self.banner = RequirementBanner(self.frame, lambda: self.app.go_to_tab(1))
        self.banner.pack(fill=tk.X, pady=(0, 9))

        content = ttk.Frame(self.frame)
        content.pack(fill=tk.X)
        settings = ttk.LabelFrame(content, text="Configuration", padding=10)
        settings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.profile_var = tk.StringVar(value="Canonical R3v Full24")
        HelpLabel(
            settings,
            "Profile",
            "The canonical profile uses all 24 channels and NASA loss. The quick profile only reduces "
            "the epochs: it checks the workflow and does not produce publication results.",
        ).grid(row=0, column=0, sticky="w", pady=3)
        profile = ttk.Combobox(
            settings,
            textvariable=self.profile_var,
            values=("Canonical R3v Full24", "Quick check", "Custom"),
            state="readonly",
            width=25,
        )
        profile.grid(row=0, column=1, sticky="w", padx=(8, 0))
        profile.bind("<<ComboboxSelected>>", self._apply_profile)
        self.epochs_var = tk.IntVar(value=250)
        self.lr_var = tk.DoubleVar(value=0.025)
        self.ridge_var = tk.DoubleVar(value=0.9)
        self.patience_var = tk.IntVar(value=50)
        labeled_entry(settings, 1, "Maximum epochs", self.epochs_var, "Early stopping is enabled; this value is a ceiling, not a requirement.")
        labeled_entry(settings, 2, "Adam learning rate", self.lr_var, "Shared learning rate for the two alternating blocks: consequents and log-authorities.")
        labeled_entry(settings, 3, "Ridge regularization λ", self.ridge_var, "Penalizes departures of consequents from the TRAIN mean and stabilizes initialization.")
        labeled_entry(settings, 4, "Patience", self.patience_var, "Number of epochs without improvement before early stopping.")

        contract = ttk.LabelFrame(content, text="Fixed choices", padding=10)
        contract.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        for row, (name, value, help_text) in enumerate(
            (
                ("Task loss", "asymmetric NASA", "This is the primary objective to minimize; MAE and RMSE remain diagnostics."),
                ("Authority", "learned log-authorities", "a_i = exp(theta_i - mean(theta)); no artificial unit-sum constraint."),
                ("Consequents", "0 ≤ c ≤ TRAIN bound", "The upper bound is 125 for cap125 and max(TRAIN RUL) for uncapped."),
                ("Backbone", "120 Full24 unary rules", "Five linguistic levels for each of the 24 physical channels."),
            )
        ):
            HelpLabel(contract, name, help_text).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Label(contract, text=value, foreground="#1266a3").grid(row=row, column=1, sticky="w", padx=10)

        self.run_button = ttk.Button(self.frame, text="Train selected models", style="Accent.TButton", command=self._run)
        self.run_button.pack(anchor=tk.W, pady=10)
        self.tree = PrepareTab._status_tree(self.frame)
        self.tree.heading("status", text="Model")
        self.tree.pack(fill=tk.X)
        ttk.Label(self.frame, text="Log", style="Section.TLabel").pack(anchor=tk.W, pady=(12, 4))
        self.log = LogPanel(self.frame, height=10)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _apply_profile(self, _event=None) -> None:
        profile = self.profile_var.get()
        if profile == "Canonical R3v Full24":
            self.epochs_var.set(250)
            self.lr_var.set(0.025)
            self.ridge_var.set(0.9)
            self.patience_var.set(50)
        elif profile == "Quick check":
            self.epochs_var.set(25)
            self.lr_var.set(0.025)
            self.ridge_var.set(0.9)
            self.patience_var.set(10)

    def _config(self) -> TrainingConfig:
        return TrainingConfig(
            loss="nasa",
            authority_mode="learned",
            epochs=int(self.epochs_var.get()),
            learning_rate_consequents=float(self.lr_var.get()),
            learning_rate_authorities=float(self.lr_var.get()),
            ridge_lambda=float(self.ridge_var.get()),
            patience=int(self.patience_var.get()),
        ).validate()

    def _run(self) -> None:
        try:
            config = self._config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        jobs = self.app.selected_jobs()
        self.log.clear()

        def worker(token, progress, log):
            outputs = []
            for job_index, (subset, protocol) in enumerate(jobs):
                token.checkpoint()
                dataset = self.app.store.load_prepared(subset, protocol)
                log(f"[{subset} · {protocol}] 24 variables, 120 unary rules, NASA loss")

                def local_progress(value, text):
                    progress((job_index + value) / len(jobs), f"{subset} · {protocol} · {text}")

                model = train_prepared(dataset, config, token=token, progress=local_progress)
                archived = self.app.store.invalidate_after("model", subset, protocol)
                model_path = self.app.store.model_path(subset, protocol)
                backbone_path = self.app.store.backbone_model_path(subset, protocol)
                model.save(model_path)
                model.save(backbone_path)
                log(
                    f"[{subset} · {protocol}] best epoch {model.fit_result.best_epoch}; "
                    f"objective {model.fit_result.best_objective:.6g} · model saved",
                    "success",
                )
                if archived:
                    log(f"Previous structure/evaluation artifacts archived: {len(archived)}", "warning")
                outputs.append(str(model_path))
            return outputs

        self.app.start_task("R3v training", self, worker)

    def refresh(self) -> None:
        if not hasattr(self, "tree"):
            return
        ready = self.app.all_jobs_have("prepared")
        if ready:
            self.banner.set_ready("prepared data are available for every selected experiment")
        else:
            self.banner.set_missing("prepare every selected experiment", "Go to Preparation")
        self.tree.delete(*self.tree.get_children())
        for subset, protocol in self.app.selected_jobs():
            done = self.app.store.model_path(subset, protocol).is_file()
            self.tree.insert("", tk.END, values=(subset, protocol, "✓ trained" if done else "not trained"))
        self.run_button.configure(state=tk.NORMAL if ready and not self.app.runner.running else tk.DISABLED)


class AutoStructureTab(BaseTab):
    title = "4 · AutoStructure"

    def _build(self) -> None:
        self._heading(
            "Evidence-controlled structural growth",
            "Candidates are compound linguistic concepts. Support and novelty screen them; only matched "
            "engine-group validation can accept them.",
        )
        self.banner = RequirementBanner(self.frame, lambda: self.app.go_to_tab(2))
        self.banner.pack(fill=tk.X, pady=(0, 9))
        content = ttk.Frame(self.frame)
        content.pack(fill=tk.X)
        settings = ttk.LabelFrame(content, text="Thresholds", padding=10)
        settings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.support_var = tk.DoubleVar(value=0.01)
        self.novelty_var = tk.DoubleVar(value=0.005)
        self.improvement_var = tk.DoubleVar(value=0.01)
        self.max_order_var = tk.IntVar(value=3)
        self.parent_var = tk.DoubleVar(value=0.0002)
        self.prune_var = tk.DoubleVar(value=0.002)
        labeled_entry(settings, 0, "Minimum support", self.support_var, "Minimum fraction of samples where candidate activation exceeds 0.01.")
        labeled_entry(settings, 1, "Minimum novelty", self.novelty_var, "Residual activation energy after projection onto the current-rule space.")
        labeled_entry(settings, 2, "NASA improvement", self.improvement_var, "Minimum relative NASA-score reduction required in the matched parent-child comparison.")
        labeled_entry(settings, 3, "Maximum order", self.max_order_var, "Higher-order rules are tried only when the preceding order produced an acceptance.")
        labeled_entry(settings, 4, "Composition threshold", self.parent_var, "A weak rule may still parent a candidate; this threshold is intentionally lower than the pruning threshold.")
        labeled_entry(settings, 5, "Pruning threshold (proposal)", self.prune_var, "Rules below this share are listed as candidates in the report. They are not deleted automatically; pruning remains separate.")

        budget = ttk.LabelFrame(content, text="Computational guard", padding=10)
        budget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.minutes_var = tk.DoubleVar(value=60.0)
        self.candidates_var = tk.IntVar(value=10000)
        self.screen_var = tk.IntVar(value=24)
        self.validate_var = tk.IntVar(value=4)
        self.rounds_var = tk.IntVar(value=3)
        self.validation_epochs_var = tk.IntVar(value=100)
        labeled_entry(budget, 0, "Maximum time (min)", self.minutes_var, "At the limit, Cassandra stops safely and proposes ways to reduce the workload.")
        labeled_entry(budget, 1, "Maximum candidates", self.candidates_var, "Hard limit before screening; prevents unintended combinatorial explosions.")
        labeled_entry(budget, 2, "Screening top-k", self.screen_var, "Maximum number of candidates reaching the expensive validation stage.")
        labeled_entry(budget, 3, "Validations per seed", self.validate_var, "Maximum number of top candidates compared from cold starts with matched budgets.")
        labeled_entry(budget, 4, "Maximum rounds", self.rounds_var, "AutoStructure adds at most one rule per round and may stop immediately.")
        labeled_entry(budget, 5, "Comparison epochs", self.validation_epochs_var, "Parent and child receive exactly the same maximum number of epochs.")

        estimate = ttk.Frame(self.frame, padding=(0, 8))
        estimate.pack(fill=tk.X)
        HelpLabel(
            estimate,
            "Pre-run estimate",
            "This is a first-order estimate. Actual time depends on the subset, epochs, retained "
            "candidates, and computer.",
        ).pack(side=tk.LEFT)
        self.estimate_var = tk.StringVar()
        ttk.Label(estimate, textvariable=self.estimate_var, foreground="#7a4b00").pack(side=tk.LEFT, padx=10)
        for variable in (self.screen_var, self.validate_var, self.validation_epochs_var):
            variable.trace_add("write", lambda *_args: self._update_estimate())

        self.run_button = ttk.Button(self.frame, text="Run AutoStructure", style="Accent.TButton", command=self._run)
        self.run_button.pack(anchor=tk.W, pady=(0, 10))
        self.tree = PrepareTab._status_tree(self.frame)
        self.tree.heading("status", text="Structure")
        self.tree.pack(fill=tk.X)
        ttk.Label(self.frame, text="Log and workload-reduction suggestions", style="Section.TLabel").pack(anchor=tk.W, pady=(12, 4))
        self.log = LogPanel(self.frame, height=9)
        self.log.pack(fill=tk.BOTH, expand=True)
        self._update_estimate()

    def _update_estimate(self) -> None:
        try:
            generated = estimate_pairwise_candidates(120, 24)
            fits = len((7, 42, 99)) * 3 * (1 + max(1, int(self.validate_var.get())))
            self.estimate_var.set(f"up to {generated:,} pairs · about {fits} matched fits per round")
        except tk.TclError:
            self.estimate_var.set("complete the parameters to obtain an estimate")

    def _config(self) -> AutoStructureConfig:
        return AutoStructureConfig(
            min_support=float(self.support_var.get()),
            min_novelty=float(self.novelty_var.get()),
            min_relative_improvement=float(self.improvement_var.get()),
            max_order=int(self.max_order_var.get()),
            parent_relative_authority=float(self.parent_var.get()),
            prune_relative_authority=float(self.prune_var.get()),
            screen_top_k=int(self.screen_var.get()),
            validation_epochs=int(self.validation_epochs_var.get()),
            budget=ComputeBudget(
                max_seconds=float(self.minutes_var.get()) * 60.0,
                max_candidates=int(self.candidates_var.get()),
                max_validations_per_seed=int(self.validate_var.get()),
                max_rounds=int(self.rounds_var.get()),
            ),
        ).validate()

    def _run(self) -> None:
        try:
            config = self._config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        jobs = self.app.selected_jobs()
        self.log.clear()

        def worker(token, progress, log):
            outputs = []
            for job_index, (subset, protocol) in enumerate(jobs):
                token.checkpoint()
                dataset = self.app.store.load_prepared(subset, protocol)
                model = R3vModel.load(self.app.store.model_path(subset, protocol))
                log(f"[{subset} · {protocol}] AutoStructure starts from {len(model.rules)} rules")

                def local_progress(value, text):
                    progress((job_index + value) / len(jobs), f"{subset} · {protocol} · {text}")

                try:
                    child, result = grow_structure(dataset, model, config, token=token, progress=local_progress)
                except ComputeBudgetExceeded as exc:
                    log(str(exc), "warning")
                    for suggestion in exc.recommendations:
                        log(f"• {suggestion}", "warning")
                    raise
                self.app.store.invalidate_after("structure", subset, protocol)
                child.save(self.app.store.model_path(subset, protocol))
                report = {
                    "schema": 1,
                    "subset": subset,
                    "protocol": protocol,
                    "created_at": utc_now(),
                    "config": config.to_dict(),
                    "result": result.to_dict(),
                }
                path = self.app.store.structure_path(subset, protocol)
                atomic_json(path, report)
                if result.accepted_rules:
                    log(
                        f"[{subset} · {protocol}] accepted {len(result.accepted_rules)} rules; "
                        f"total {result.final_rule_count}",
                        "success",
                    )
                else:
                    log(f"[{subset} · {protocol}] no rule accepted: {result.stop_reason}", "success")
                if result.pruning_candidates:
                    log(
                        f"[{subset} · {protocol}] {len(result.pruning_candidates)} rules below the pruning threshold: "
                        "listed in the report, not deleted.",
                        "warning",
                    )
                outputs.append(str(path))
            return outputs

        self.app.start_task("AutoStructure", self, worker)

    def refresh(self) -> None:
        if not hasattr(self, "tree"):
            return
        ready = self.app.all_jobs_have("model")
        if ready:
            self.banner.set_ready("a parametric model is available for every experiment")
        else:
            self.banner.set_missing("train every selected model", "Go to Model")
        self.tree.delete(*self.tree.get_children())
        for subset, protocol in self.app.selected_jobs():
            path = self.app.store.structure_path(subset, protocol)
            label = "optional · not run"
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        result = json.load(handle)["result"]
                    label = f"✓ {result['final_rule_count']} rules · {len(result['accepted_rules'])} added"
                except (OSError, ValueError, KeyError):
                    label = "unreadable report"
            self.tree.insert("", tk.END, values=(subset, protocol, label))
        self.run_button.configure(state=tk.NORMAL if ready and not self.app.runner.running else tk.DISABLED)


class EvaluateTab(BaseTab):
    title = "5 · Evaluation"

    def _build(self) -> None:
        self._heading(
            "Paired official endpoints and faithful explanations",
            "The frozen backbone and final AutoStructure model are evaluated on identical TEST engines. "
            "NASA score is primary; positive gain means a lower final score.",
        )
        self.banner = RequirementBanner(self.frame, lambda: self.app.go_to_tab(2))
        self.banner.pack(fill=tk.X, pady=(0, 9))
        buttons = ttk.Frame(self.frame)
        buttons.pack(fill=tk.X, pady=(0, 8))
        self.run_button = ttk.Button(
            buttons,
            text="Evaluate selected experiments",
            style="Accent.TButton",
            command=self._run,
        )
        self.run_button.pack(side=tk.LEFT)
        HelpLabel(
            buttons,
            "What is compared?",
            "Cassandra compares model_backbone.json with model.json, uses a paired 20,000-resample "
            "engine bootstrap, and never uses TEST to accept or reject a rule.",
        ).pack(side=tk.LEFT, padx=12)

        columns = (
            "subset",
            "protocol",
            "rules",
            "backbone_nasa",
            "final_nasa",
            "gain",
            "interval",
            "verdict",
            "mae",
            "k90",
        )
        self.tree = ttk.Treeview(self.frame, columns=columns, show="headings", height=8, selectmode="browse")
        labels = {
            "subset": "Subset",
            "protocol": "Protocol",
            "rules": "Rules",
            "backbone_nasa": "Backbone NASA",
            "final_nasa": "Final NASA ↓",
            "gain": "Gain % ↑",
            "interval": "95% interval",
            "verdict": "Bootstrap",
            "mae": "Final MAE",
            "k90": "Median k90",
        }
        for key in columns:
            self.tree.heading(key, text=labels[key])
            width = 125 if key in ("backbone_nasa", "final_nasa", "interval") else 95
            self.tree.column(key, width=width, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X)

        explain = ttk.LabelFrame(self.frame, text="Endpoint explanation", padding=10)
        explain.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        row = ttk.Frame(explain)
        row.pack(fill=tk.X)
        ttk.Label(row, text="TEST engine:").pack(side=tk.LEFT)
        self.engine_var = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=1, to=1000, textvariable=self.engine_var, width=8).pack(side=tk.LEFT, padx=6)
        self.explain_button = ttk.Button(row, text="Explain selected row", command=self._explain)
        self.explain_button.pack(side=tk.LEFT)
        HelpLabel(
            row,
            "Rule ordering",
            "Rules are ordered by local responsibility β, not only by β·c: even a zero consequent "
            "can take responsibility away from other rules.",
        ).pack(side=tk.LEFT, padx=12)
        self.log = LogPanel(explain, height=13)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _run(self) -> None:
        jobs = self.app.selected_jobs()
        self.log.clear()

        def worker(token, progress, log):
            outputs = []
            for index, (subset, protocol) in enumerate(jobs):
                token.checkpoint()
                dataset = self.app.store.load_prepared(subset, protocol)
                final_model = R3vModel.load(self.app.store.model_path(subset, protocol))
                backbone = R3vModel.load(self.app.store.backbone_model_path(subset, protocol))
                result = evaluate_prepared(final_model, dataset)
                comparison = compare_prepared_models(backbone, final_model, dataset, token=token)
                evaluation_path = self.app.store.evaluation_path(subset, protocol)
                atomic_json(evaluation_path, result)
                comparison_path = save_comparison(self.app.store, comparison)
                interval = comparison["bootstrap"]["relative_nasa_improvement_percent"]
                conclusion = comparison["change"]["bootstrap_conclusion"]
                log_tag = {
                    "supported_improvement": "success",
                    "supported_degradation": "error",
                    "inconclusive": "warning",
                    "no_change": "muted",
                }[conclusion]
                log(
                    f"[{subset} · {protocol}] NASA {comparison['backbone']['nasa_score']:.3f} → "
                    f"{comparison['final_model']['nasa_score']:.3f} · "
                    f"gain={comparison['change']['nasa_improvement_percent']:+.3f}% · "
                    f"95% interval=[{interval['ci_95_low']:+.3f}%, "
                    f"{interval['ci_95_high']:+.3f}%] · {conclusion}",
                    log_tag,
                )
                counts = comparison["paired_engine_counts"]
                log(
                    f"    paired engines: {counts['lower_nasa_penalty']} improved · "
                    f"{counts['equal_nasa_penalty']} unchanged · {counts['higher_nasa_penalty']} worsened",
                    "muted",
                )
                outputs.extend((str(evaluation_path), comparison_path))
                progress((index + 1) / len(jobs), f"Evaluated {subset} · {protocol}")
            return outputs

        self.app.start_task("endpoint evaluation", self, worker)

    def _selected_job(self) -> tuple[str, str] | None:
        selection = self.tree.selection()
        if selection:
            values = self.tree.item(selection[0], "values")
            return str(values[0]), str(values[1])
        jobs = self.app.selected_jobs()
        return jobs[0] if jobs else None

    def _explain(self) -> None:
        job = self._selected_job()
        if not job:
            return
        try:
            subset, protocol = job
            dataset = self.app.store.load_prepared(subset, protocol)
            model = R3vModel.load(self.app.store.model_path(subset, protocol))
            result = explain_test_engine(model, dataset, int(self.engine_var.get()), top_k=12)
        except (OSError, ValueError, KeyError, tk.TclError) as exc:
            messagebox.showerror("Explanation unavailable", str(exc))
            return
        self.log.clear()
        self.log.append(
            f"{subset} · {protocol} · engine {result['engine_id']} · true RUL {result['true_rul']:.2f} · "
            f"predicted RUL {result['prediction']:.2f}",
            "success",
        )
        self.log.append(f"Σβ={result['responsibility_sum']:.12f} · reconstruction={result['reconstruction']:.6f}")
        self.log.append(result["note"], "warning")
        self.log.append("\nTop rules by local responsibility:")
        for rank, item in enumerate(result["top_rules"], start=1):
            self.log.append(
                f"{rank:02d}. {item['rule']}\n"
                f"    μ={item['activation_mu']:.5f} · a={item['relative_authority_a']:.5f} · "
                f"β={item['responsibility_beta']:.5f} · c={item['consequent_c']:.3f} · "
                f"βc={item['contribution_beta_c']:.3f}"
            )

    def refresh(self) -> None:
        if not hasattr(self, "tree"):
            return
        ready = self.app.all_jobs_have("model") and self.app.all_jobs_have("backbone")
        if ready:
            self.banner.set_ready("backbone and final models are available for paired evaluation")
        else:
            self.banner.set_missing("train the selected models to create both frozen artifacts", "Go to Model")
        self.tree.delete(*self.tree.get_children())
        for subset, protocol in self.app.selected_jobs():
            evaluation_path = self.app.store.evaluation_path(subset, protocol)
            comparison_path = self.app.store.comparison_path(subset, protocol)
            values: tuple[Any, ...] = (
                subset,
                protocol,
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
            )
            if evaluation_path.is_file() and comparison_path.is_file():
                try:
                    with evaluation_path.open("r", encoding="utf-8") as handle:
                        result = json.load(handle)
                    with comparison_path.open("r", encoding="utf-8") as handle:
                        comparison = json.load(handle)
                    interval = comparison["bootstrap"]["relative_nasa_improvement_percent"]
                    values = (
                        subset,
                        protocol,
                        f"{comparison['backbone']['rules']}→{comparison['final_model']['rules']}",
                        f"{comparison['backbone']['nasa_score']:.2f}",
                        f"{comparison['final_model']['nasa_score']:.2f}",
                        f"{comparison['change']['nasa_improvement_percent']:+.2f}%",
                        f"[{interval['ci_95_low']:+.2f}, {interval['ci_95_high']:+.2f}]",
                        comparison["change"]["bootstrap_conclusion"].replace("_", " "),
                        f"{result['mae']:.2f}",
                        f"{result['responsibility']['median_k90']:.1f}",
                    )
                except (OSError, ValueError, KeyError):
                    pass
            self.tree.insert("", tk.END, values=values)
        enabled = ready and not self.app.runner.running
        self.run_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        self.explain_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)


class FullTestTab(BaseTab):
    """Run comprehensive diagnostics and produce a shareable TXT report."""

    title = "6 · Full Test"

    def _build(self) -> None:
        self.latest_report_path: Path | None = None
        self._heading(
            "Comprehensive diagnostics and shareable report",
            "Full Test inspects every available component without replacing scientific artifacts. "
            "It automatically saves one complete TXT report for development review.",
        )

        coverage = ttk.LabelFrame(self.frame, text="Diagnostic coverage", padding=10)
        coverage.pack(fill=tk.X)
        entries = (
            (
                "Environment and source",
                "Checks Python, NumPy, pandas, Tkinter, packaged help, source syntax, and version consistency.",
            ),
            (
                "Automated tests",
                "Runs the complete source test suite when it is available and records every test result.",
            ),
            (
                "NASA material",
                "Parses every selected, available FD001–FD004 raw dataset and records file hashes, "
                "dimensions, and engine counts.",
            ),
            (
                "Artifacts and pipeline",
                "Validates saved artifacts, compares backbone and final models with a paired 20,000-resample "
                "bootstrap, then runs a short non-destructive pipeline smoke test.",
            ),
        )
        for index, (name, explanation) in enumerate(entries):
            HelpLabel(coverage, name, explanation, width=55).grid(
                row=index // 2,
                column=(index % 2) * 2,
                sticky="w",
                padx=(0, 28),
                pady=4,
            )

        actions = ttk.Frame(self.frame)
        actions.pack(fill=tk.X, pady=(10, 7))
        self.run_button = ttk.Button(
            actions,
            text="Run Full Test",
            style="Accent.TButton",
            command=self._run,
        )
        self.run_button.pack(side=tk.LEFT)
        self.save_button = ttk.Button(actions, text="Save Report As…", command=self._save_report)
        self.save_button.pack(side=tk.LEFT, padx=7)
        HelpLabel(
            actions,
            "Safe stop and partial report",
            "The global Stop processing button remains active. A stop request is handled at a safe "
            "checkpoint, and the diagnostics completed so far are saved as a partial TXT report.",
            width=58,
        ).pack(side=tk.LEFT, padx=10)

        self.summary_var = tk.StringVar(value="No Full Test report has been created yet.")
        ttk.Label(self.frame, textvariable=self.summary_var, foreground="#1266a3", wraplength=1120).pack(
            anchor=tk.W,
            pady=(0, 6),
        )
        ttk.Label(self.frame, text="Complete diagnostic log", style="Section.TLabel").pack(anchor=tk.W, pady=(4, 4))
        self.log = LogPanel(self.frame, height=18)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _run(self) -> None:
        self.log.clear()
        subsets = self.app.selected_subsets()
        protocols = self.app.selected_protocols()

        def worker(token, progress, log):
            return run_full_test(
                project_root=self.app.project_root,
                repository=self.app.repository,
                store=self.app.store,
                selected_subsets=subsets,
                selected_protocols=protocols,
                token=token,
                progress=progress,
                log=log,
            )

        self.app.start_task("Full Test", self, worker, on_success=self._completed)

    def _completed(self, result: FullTestResult) -> None:
        self.latest_report_path = Path(result.report_path)
        self.summary_var.set(
            f"Last report: {self.latest_report_path.name} · {result.overall_status} · "
            f"PASS={result.counts['PASS']}, WARN={result.counts['WARN']}, "
            f"SKIP={result.counts['SKIP']}, FAIL={result.counts['FAIL']}"
        )
        self.save_button.configure(state=tk.NORMAL)

    def _save_report(self) -> None:
        source = self.latest_report_path or self.app.store.latest_full_test_report()
        if source is None or not source.is_file():
            messagebox.showinfo("No report available", "Run Full Test before saving a report copy.")
            return
        destination = filedialog.asksaveasfilename(
            title="Save the complete Full Test report",
            defaultextension=".txt",
            initialfile=source.name,
            filetypes=[("Text reports", "*.txt"), ("All files", "*")],
        )
        if not destination:
            return
        try:
            shutil.copyfile(source, Path(destination))
        except OSError as exc:
            messagebox.showerror("Unable to save report", str(exc))
            return
        messagebox.showinfo("Report saved", f"Complete report saved to:\n{destination}")

    def refresh(self) -> None:
        if not hasattr(self, "run_button"):
            return
        latest = self.app.store.latest_full_test_report()
        if latest is not None:
            self.latest_report_path = latest
            try:
                status = "unknown status"
                summary = ""
                for line in latest.read_text(encoding="utf-8").splitlines():
                    if line.startswith("OVERALL STATUS:"):
                        status = line.partition(":")[2].strip()
                    elif line.startswith("SUMMARY:"):
                        summary = line.partition(":")[2].strip()
                suffix = f" · {summary}" if summary else ""
                self.summary_var.set(f"Last report: {latest.name} · {status}{suffix}")
            except OSError:
                self.summary_var.set(f"Last report: {latest.name} · unreadable")
        state = tk.DISABLED if self.app.runner.running else tk.NORMAL
        self.run_button.configure(state=state)
        self.save_button.configure(state=tk.NORMAL if latest is not None and latest.is_file() else tk.DISABLED)
