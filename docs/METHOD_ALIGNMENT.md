# R3v method and Cassandra R3v 1.0.0

This document maps the public methodological contract to this implementation. Cassandra R3v 1.0.0 is a public software reference for the R3v method and is intended to support reproducibility and alignment with the associated article.

| Methodological contract | Canonical implementation |
| --- | --- |
| Normalized zero-order Takagi–Sugeno model | `R3vModel._from_mu` |
| Antecedent activation `mu_i` | `SemanticRule.activation` with product t-norm |
| Consequent `c_i` | `R3vModel.consequents` |
| Positive relative authority `a_i` | `exp(theta_i - mean(theta))` |
| Local responsibility `beta_i` | exact normalization of `a_i * mu_i` |
| No `sum(a)=1` constraint | centered log-authorities; shares are diagnostics only |
| Canonical task loss | NASA asymmetric loss |
| Consequent bounds | `[0,125]` for `cap125`; `[0,max(y_TRAIN)]` for `uncapped` |
| Five stable linguistic levels | extreme shoulders plus three internal triangles |
| Physical inputs | `OS1...OS3` and `S1...S21`; cycle and RUL are excluded |
| TRAIN-only pseudo-endpoints | 35%, 50%, 65%, 80%, and 92% of each observed engine life |
| Grouped validation | complete engines are assigned to folds |
| Structural screening | support, residualization, novelty ratio, directional score |
| Matched comparison | parent and child start cold with identical budgets |
| Consensus | seeds 7, 42, and 99; default requirement 2 of 3 |
| Growth stopping | stop immediately when no candidate passes a stage |
| Pruning | proposals are reported; no rule is deleted automatically |
| Final structural effect | paired official TEST endpoints with a 20,000-resample engine bootstrap |

## Protocol separation

`uncapped` and `cap125` use separate preparations, models, structure reports, evaluations, paired comparisons, interface rows, and Full Test lines. No reporting function combines their metrics.

## Structural semantics

AutoStructure creates conjunctions using `AND`. A candidate is accepted only after support and novelty screening followed by matched engine-group validation and the configured seed consensus. At most one rule is accepted per round.

The lower parent-authority threshold keeps a weak rule eligible as a component before the higher pruning-proposal threshold is applied. A pruning proposal is descriptive evidence only. Disjunctive `OR` consolidation and automatic deletion are outside the 1.0.0 contract.

## Public release contract

Scientific results are attributable to Cassandra only when all of the following are preserved:

1. tag `v1.0.0` and its commit SHA;
2. the unmodified `examples/article_profile.json`, or a fully disclosed replacement;
3. hashes of the C-MAPSS input files;
4. the preparation, model, structure, and evaluation artifacts for each experiment;
5. the execution environment and complete command line;
6. a passing Full Test report from the same tagged source.

The software must not infer or manufacture a structure from aggregate metrics. Every interaction claimed in the article must occur in its saved `autostructure.json` and final `model.json`.

## Evaluation boundary

AutoStructure accepts rules exclusively through grouped TRAIN-only matched validation. After the final structure is frozen, Evaluation compares `model_backbone.json` and `model.json` on the same official TEST engines. The deterministic paired bootstrap quantifies uncertainty in NASA-score improvement; it does not retrospectively alter the structure.

A positive point estimate whose 95% interval includes zero is reported as `inconclusive`, not as a demonstrated improvement. A Full Test `PASS` for a paired comparison means that the calculation and artifacts are valid, not that the predictive effect is favorable.

## Declared limitations

- Fuzzy partitions remain fixed after fitting on TRAIN.
- The canonical model uses instantaneous inputs and adds neither age nor temporal gradients.
- Candidate generation is combinatorial, although filters and hard budgets bound it.
- Rule decompositions are faithful to the prediction but are not causal explanations.
- Automatic pruning and `OR` consolidation are not implemented.
- A single paired TEST set can leave a structural effect statistically inconclusive.
- The release does not claim state-of-the-art accuracy against unrestricted black-box models.

## Diagnostic boundary

Full Test uses deliberately short smoke budgets to verify executable paths and numerical invariants. Its smoke metrics are acceptance evidence for the software, not experimental results for the article.
