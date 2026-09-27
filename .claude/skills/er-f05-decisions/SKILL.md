---
name: er-f05-decisions
description: Use when turning scored candidate pairs (s1, cand, p) into per-S1 match lists or matching_results.tsv, choosing a threshold or deciding empty lists for singletons, when one S2/S3 record is claimed by several Source-1 entities (exclusivity), when computing or breaking down macro F0.5 with metric.py, tuning decision parameters on OOF, applying them to France or test data, or when validation F0.5 disagrees with AUC or pair-level F1.
---

# F0.5 decisions and scoring (per-S1 macro F0.5)

## Overview
The score is an **average over S1 entities**, not over pairs, and every S2/S3 record belongs to **at most one** S1. So decide per S1: make the probabilities exclusivity-aware, keep them calibrated, then take the top-m candidates that maximise the expected per-S1 F0.5, with m = 0 meaning an empty list. Tune the 2-3 free parameters on train OOF only, and always report the score per country.

## When to use
- You have OOF scored pairs from **er-matcher-training** and need the validation macro F0.5, or test scored pairs and need final links.
- You are deciding on singletons (empty lists) or on records claimed by several S1s.
- You are about to trust a validation or leaderboard number, or to compare two pipeline variants.
- Not for: generating candidates (**er-blocking-candidates**), features or training (**er-pair-features**, **er-matcher-training**), or the TSV format and validators (**er-submission-packaging**).

## Quick reference
| Task | Call (from this folder: `metric.py`, `decide.py`; copied to `src/ber/`) |
|---|---|
| Score a prediction TSV against GT, on validation ids only, with breakdowns | `python metric.py --pred P.tsv --gt train_ground_truth.tsv --ids val_ids.txt --s1 train_source1.tsv --by-country --by-k --by-pred-size --json scores.json` |
| Check the metric edge cases | `python metric.py --selftest` |
| Score long frames `(s1, mid)` | `M.macro_f05(pred, gt, s1_ids)`, `M.f05_breakdown(pred, gt, s1_meta)`, `M.per_s1_scores(...)` |
| Stable fold ids (GroupKFold by S1) | `M.fold_of(s1_ids, n_folds=5, seed=0)` |
| Final links, exclusive, expected-F0.5 | `D.select_links(scored, "expected_f", exclusivity="none"/"soft", lam_missing=λ, empty_bias=b)` → `s1, mid` |
| Same, keeping `p` and `sel` (for shards and monitoring) | `D.resolve(scored, method="exact", exclusivity=..., lam_missing=λ, empty_bias=b)` |
| Exclusivity only | `D.assign_exclusive(scored, margin=0.0, mode="hard"/"soft")` |
| Per-S1 table (`m_star`, `ef_best`, `p_empty`) | `D.decide_exact(scored, λ, return_s1_table=True)` |
| O(n) approximation (same decisions in about 99% of S1) | `method="expected_f_fast"` (series) |
| Global-threshold baseline | `D.select_links(scored, "threshold", t=0.6)` (exclusivity is still enforced by the repair step) |
| Tests | `python -m pytest -q -p no:cacheprovider test_metric.py test_decide.py` (25 pass) |

## The metric, exactly
Per S1 with truth set T (size k) and predicted set P (size m, sets, so duplicates collapse), where tp = |P ∩ T|:

| Case | Score |
|---|---|
| T empty, P empty (correct singleton) | **1.0** |
| T empty, P non-empty (false merge on a singleton) | **0.0** |
| T non-empty, P empty | **0.0** (P undefined, R = 0) |
| both non-empty, disjoint | 0.0 |
| otherwise | 1.25·tp / (0.25·k + m), identical to 1.25PR/(0.25P+R) |

The score is macro-averaged over **all** evaluated S1: singletons count, and so do S1 with zero candidates.

Worked values (from `test_metric.py`):

| k | prediction | F0.5 |
|---|---|---|
| 2 | {a,b,c}, truth {a,c} (README example) | **0.714** |
| 4 | 2 of 4 correct, no FP | **0.833** |
| 4 | 3 of 4 correct | 0.9375 |
| 4 | all 4 correct + 1 wrong | 0.833 |
| 4 | 1 of 4 correct | 0.625 (empty would score 0) |
| 1 | the 1 correct + 1 wrong | 0.556 |

Why this changes the optimal decision:
- **Partial precise lists are cheap, empty lists are expensive.** 94.4% of S1 have matches, so an empty list on any of them costs a full 1.0. One extra wrong id costs about 0.17.
- **The bar for adding a candidate rises as the list gets better.** For fixed k, adding a candidate with probability p raises expected F only if **p > F_current / 1.25**. That means p > 0.8 once the list is perfect, and p > 0.64 at F = 0.8. A single global threshold cannot express this.
- **A lone best candidate is worth predicting when p1 is above about 0.5.** The exact break-even is 0.48 with λ = 0.1 missing matches, and 0.41-0.44 when the S1 also has weak candidates. Below that, an empty list wins.

## Procedure (on OOF first, then test)
1. **Build the evaluation frame.** You need OOF `(s1, cand, p, label)` for every train S1, plus `s1_meta` = **all** train S1 ids with country. S1 with no candidates have no pair rows but stay in the denominator.
2. **Tune on complete geographic shards, never on a random S1 sample.** Exclusivity needs every competing S1 of a record, and random sampling removes the competitors. Use 2-3 full states per country, or the full train OOF on AWS.
3. **Estimate λ (`lam_missing`) per country:** `(Σk − true links inside the candidate lists) / n_S1` on OOF. This is the blocking-miss rate per S1 (≈ 0.10 at 97% pair recall).
4. **Make p exclusivity-aware and calibrated.** Preferred: the matcher already has competition features (rank in S1, gap to the S1's best, number of claimants, best competing S1 score; **See also:** er-pair-features). Then use `exclusivity="none"`; the repair step still guarantees at most one S1 per record. Without such features, use soft exclusivity **followed by isotonic recalibration on OOF** (snippet below). Soft p' deflates probabilities, and expected-F on the uncalibrated p' lost 0.010 to a plain threshold in simulation.
5. **Tune `empty_bias` (grid 0.5 to 8) and pick the method** on OOF macro F0.5 over all OOF S1 ids. Keep a tuned global threshold as a baseline row.
6. **Report** overall, per country, per true-k bucket and per predicted size, plus the oracle ceiling (all in-candidate true links predicted). Log it in `work/experiments.md`.
7. **Apply to test with the chosen parameters.** France gets the pooled parameters. Then run the shift checks below.

```python
import numpy as np, polars as pl
from sklearn.isotonic import IsotonicRegression
from ber import decide as D, metric as M

def lam_missing(oof, gt, ids):                 # blocking misses per S1 (ids = ALL evaluated S1)
    found = oof.filter(pl.col("label") == 1).height
    return (gt.join(pl.DataFrame({"s1": ids}), on="s1", how="semi").height - found) / len(ids)

def soft_iso(oof, test=None):                  # only if the matcher has no competition features
    so = D.exclusivity_soft(oof)               # p' = o/(1+sum o over claimants), o = p/(1-p)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(so["p"], so["label"])
    fix = lambda f: f.with_columns(pl.Series("p", iso.predict(f["p"].to_numpy())))
    return fix(so), (fix(D.exclusivity_soft(test)) if test is not None else None)

def tune(oof, gt, meta, lam, biases=(0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 8)):
    rows = []
    for b in biases:
        links = D.select_links(oof, "expected_f", exclusivity="none", lam_missing=lam, empty_bias=b)
        rows.append((b, M.macro_f05(links, gt, meta["s1"])))
    for t in np.arange(0.30, 0.91, 0.02):     # baseline row
        links = D.select_links(oof, "threshold", exclusivity="none", t=float(t))
        rows.append((f"t={t:.2f}", M.macro_f05(links, gt, meta["s1"])))
    return sorted(rows, key=lambda r: -r[1])
```
Then call `M.f05_breakdown(best_links, gt, meta)` for the per-country and per-k table.

## Simulation evidence
Measured 2026-09-25 with the real train k distribution (5.58% singletons, mean 3.46), 97% blocking recall, confusable S1 groups that create exclusivity conflicts, and exact-Bayes calibrated p. Parameters were tuned on world A and evaluated on an independent world B (50k S1 each, about 21 candidates per S1). Full grid, generative model and more studies are in `reference-decision-theory.md`.

| Rule (all parameters tuned on A) | macro F0.5 on B (pair AUC 0.973) | pred/S1 |
|---|---|---|
| Global threshold (t = 0.59) | 0.8245 | 2.83 |
| Threshold + best-candidate fallback | 0.8257 | 2.84 |
| Expected-F exact, tuned bias (no exclusivity) | 0.8245 | 2.70 |
| Hard exclusivity + threshold | 0.8614 | 2.92 |
| Soft exclusivity + threshold | 0.8611 | 2.88 |
| Soft + expected-F on **uncalibrated** p' | 0.8514 | 2.44 |
| Soft + **isotonic** + `resolve(none, exact, bias=3)` | 0.8625 | 2.79 |
| Stage-2 competition features + threshold | 0.8620 | 2.88 |
| **Stage-2 competition features + `resolve(none, exact, bias=2)`** | **0.8652** | 2.85 |
| Oracle (every in-candidate true link) | 0.9911 | 3.36 |

What the table shows:
- Exclusivity is the big lever (+0.037).
- Per-S1 expected-F adds +0.001 to +0.003 over a tuned threshold, but **only when p is calibrated after exclusivity**.
- Competition features in the matcher beat post-hoc soft exclusivity.
- The fast series approximation gave the same score as exact (0.8246 vs 0.8245).

## Exclusivity (each S2/S3 id links to at most one S1)
- Fact: 0 violations in 7.64M train links (`reference-data-facts.md` in er-challenge-playbook). Country never crosses, so decide per country.
- **Soft:** `p' = o / (1 + Σ_claimants o)`. A record with a single claimant keeps p exactly, so the change only matters for contested records.
- **Hard:** keep only the best S1 per record. With `margin`, the record is also dropped when best − second < margin. The margin tuned to 0.0 in simulation.
- **`resolve()`:** exclusivity, then the per-S1 decision, then a repair loop. The repair removes losing edges and re-decides only the affected S1 (`repair_iters=2`). A final guarantee keeps at most one S1 per record even under ties. Every `select_links` call ends with this guarantee, whatever `exclusivity=` says.
- **Shards:** run `resolve` per blocking shard. If a record can be a candidate in two shards, concatenate the selected rows (with `p`) and run `D.exclusivity_hard(sel, margin=0)` once globally. Log how many links it removed.

## Singletons and `empty_bias`
- `EF[m=0] = P(truth empty) = Π(1−p_j)·e^(−λ)`. The independence assumption misjudges this for S1 with several weak candidates. `empty_bias` multiplies it (> 1 favours empty lists).
- Tuned values in simulation: 1.5-2 on plain calibrated p, 2 on stage-2 p, 3 after soft + isotonic. Always tune it; never hard-code it.
- An S1 with zero candidates automatically gets an empty list. It scores 1 if it is a singleton and 0 otherwise, so it must stay in the metric denominator: pass all S1 ids to `macro_f05` / `--ids`.
- Watch `singleton_empty_acc` and `nonsingleton_empty_rate` in the metric report. Both come from one trade-off: 5.58% singletons against a 1.0 loss per wrongly emptied matched S1.

## France and test-time shift
- France has no labels. Use **pooled** λ, `empty_bias` and method (tuned on US+India together). Never fit anything to France predictions. If you tuned per-country values for US and India, France still gets the pooled ones.
- Test pools are about 23% denser per S1 (S2+S3 per S1 is 5.8 on test vs 4.68 on train), so there are more distractors or more matches. After predicting, compare **test vs OOF per country**:
  - mean predicted links per S1
  - empty-list rate
  - candidates per S1 and Σp per S1
  - conflict rate (share of records with ≥ 2 claimants at p > 0.5)
  - mean `ef_best` from `decide_exact(..., return_s1_table=True)`, a label-free estimate of F
- If test pred/S1 or the empty rate moves by more than about 10% relative to OOF, look for the cause: blocking volume, calibration, or normalization on France. Log the diagnosis. Do **not** retune on the leaderboard.

## Common mistakes
| Mistake | Fix |
|---|---|
| One global threshold on raw p, exclusivity ignored | Exclusivity-aware p (competition features, or soft + isotonic), then per-S1 expected-F with tuned `empty_bias` (+0.04 in simulation) |
| Expected-F on soft-exclusivity p' without recalibrating | Isotonic on OOF p' first. Uncalibrated p' lost 0.010 to a threshold |
| Two S1 lists share an S2/S3 id | Always go through `resolve`/`select_links`; global `exclusivity_hard` after sharded decisions |
| Tuning thresholds/bias on the public leaderboard (a small test subset) | Tune on train OOF only; the leaderboard is a sanity check |
| Scoring only S1 that have predictions or candidates | Evaluate over all S1 ids of the split: `--ids` / `s1_meta` lists every S1; missing rows score as empty |
| Averaging F only over non-empty predictions or only matched S1 | The macro average includes singletons (1.0 when empty) and emptied matched S1 (0.0) |
| Pair-level AUC / pair-F1 / micro P-R as the objective | Report them as diagnostics only; select models and rules on macro F0.5 per S1 |
| Tuning on a random S1 sample or on in-sample (non-OOF) p | Complete geo shards; GroupKFold-by-S1 OOF probabilities (`fold_of`) |
| One number for all countries | `--by-country` / `f05_breakdown`; France gets pooled parameters and shift monitoring |
| Hard-coding `empty_bias=1` or λ = 0 | Estimate λ from OOF blocking misses; grid-tune `empty_bias` (optimum hit 3.0 in one setting) |

## Files in this folder
- `metric.py`: exact scorer (library + CLI, polars and pure-python engines that agree to 1e-12), plus breakdowns by country, true k and predicted size, and a loss decomposition into `singleton_false_merge`, `nonsingleton_empty_pred`, `nonsingleton_all_wrong` and `partial`.
- `decide.py`: threshold, exact and fast expected-F, soft/hard exclusivity, `resolve`, the contract functions `assign_exclusive` and `select_links`, and the TSV writer `write_id_lists`.
- `test_metric.py`, `test_decide.py`: include brute-force enumeration checks of the exact expected F.
- `reference-decision-theory.md`: derivations, the full simulation grid, the shift, miscalibration and transfer studies, and the speed benchmarks.

**REQUIRED SUB-SKILL:** er-matcher-training (calibrated GroupKFold OOF p is the input). **See also:** er-pair-features, er-error-analysis, er-submission-packaging.
