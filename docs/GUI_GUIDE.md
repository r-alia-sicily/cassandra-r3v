# Cassandra 1.0.0 — interface guide

## Launch banner

The source launcher, the `cassandra-r3v gui` command, and the dedicated GUI
entry point print Cassandra's stable ASCII banner before the desktop window
opens. The artwork is ASCII-only for terminal compatibility, and the displayed
release number comes directly from the application version. Headless commands
do not print the banner because their structured output must remain clean.

## Shared selections

The four subsets and two target protocols are selected once in the top bar. The choices apply to every tab. Cassandra prevents users from accidentally deselecting every protocol or every subset.

On first launch, both `uncapped` and `cap125` are active. Every subset/protocol pair produces separate files.

## Dependency strip

The numbered strip shows the state of the complete selected scope:

- green: every required artifact exists;
- yellow: only some experiments are ready;
- light gray: a required stage is missing;
- blue gray: an optional stage, such as AutoStructure after a backbone exists;
- Full Test turns green after a passing report and yellow after a failed or cancelled report.

When a tab cannot run, its opening banner identifies the missing prerequisite and links directly to the tab that resolves it.

## Local help

Labels marked with `ⓘ` are clickable. They open an explanation beside the relevant item instead of moving the user to a general guide. A second click, or a click outside the popup, closes the explanation.

## Global stop

During a calculation, **Stop processing** becomes active. The command does not close Cassandra. The operation ends at the next safe checkpoint, and a temporary file is never mistaken for a complete result.

## AutoStructure and computational cost

The tab shows the maximum number of first-order pairs and an estimate of matched fits per round. Support and novelty thresholds reduce screening; time, candidate, validation, and round limits are hard bounds.

The composition threshold is lower than the pruning threshold: a temporarily weak rule can still participate in building a useful concept. Rules below the pruning threshold are only proposed in the report; pruning is separate and never automatic.

## Paired evaluation

The Evaluation tab first compares the frozen `model_backbone.json` with the final `model.json` on exactly the same official TEST endpoints. It reports the backbone and final NASA scores, the percentage gain, and a paired 95% bootstrap interval based on 20,000 engine-level resamples. Positive gain means that the final NASA score is lower.

The displayed conclusion has four possible values:

- `no change`: the two models produce identical endpoint predictions;
- `supported improvement`: the complete 95% interval is above zero;
- `supported degradation`: the complete 95% interval is below zero;
- `inconclusive`: the interval includes zero.

This comparison is evaluative only. TEST data never accept, reject, or tune a rule; AutoStructure decisions remain TRAIN-only.

## Explanations

For the selected endpoint, Cassandra reports:

- the semantic rule and its order;
- antecedent activation `mu`;
- positive global relative authority `a`;
- normalized local responsibility `beta`;
- consequent `c`;
- contribution `beta*c`.

Rules are ordered by `beta`. The prediction reconstruction is exact, but it is not claimed to be causal.

## Full Test

The sixth tab checks the complete software environment and every item that is actually available:

1. runtime dependencies and source integrity;
2. built-in numerical contracts and the source test suite;
3. all selected FD001–FD004 raw files found in the workspace;
4. every saved preparation, backbone, final model, structure report, evaluation, and paired comparison;
5. a fresh backbone–AutoStructure comparison with a 20,000-resample paired bootstrap;
6. a two-epoch, non-destructive Full24 training/evaluation path for every selected experiment;
7. a bounded AutoStructure screening and matched-validation path.

The short smoke-test metrics are diagnostic values, not manuscript results. Missing NASA data or scientific artifacts are marked `SKIP`, while any present item is parsed and validated.

A complete TXT report is saved automatically in `cassandra_workspace/reports`. **Save Report As…** creates a copy at a user-selected location. If **Stop processing** is pressed, Cassandra saves the checks completed so far in a report clearly marked `CANCELLED`.
