> Research notes captured 2026-09-25 while building the er-f05-decisions skill (agent b-decide), on the team laptop (i5-1235U, 16 GB, about 1.5-2 GB free). Simulation and benchmark scripts lived in a temporary scratch folder that is NOT part of the project. The generative model is described below so it can be rebuilt. Rely on the numbers and conclusions, and re-measure on real OOF predictions once the matcher exists.

# Decision theory for per-S1 macro F0.5 (reference for er-f05-decisions)

## 1. Metric identities
- Per S1: `F = 1.25·tp / (0.25·k + m)` when tp > 0; 1.0 when k = m = 0; 0 otherwise. `metric.f_beta_counts` implements exactly this. `test_formula_equivalence` checks it against `1.25PR/(0.25P+R)` for every tp ≤ 7, k ≤ 11, m ≤ 14.
- Loss decomposition (`metric.py` `loss_points`, which sum to 1 − macro F):
  - `singleton_false_merge`: k = 0 and m > 0, loses 1 each
  - `nonsingleton_empty_pred`: k > 0 and m = 0, loses 1 each
  - `nonsingleton_all_wrong`: tp = 0 and m > 0, loses 1 each
  - `partial`: loses 1 − F
- Use the decomposition to see where the loss comes from before changing anything. Singleton false merges and emptied matched S1 each cost 1.0 per S1. Missing links inside a precise list cost little: missing one link of k = 4 costs 0.0625, and missing one of k = 7 costs 0.032.

## 2. Optimal per-S1 decision
Model: Z_j ~ Bernoulli(p_j) independent for the candidates, and Q = number of true matches outside the list (blocking misses plus the low-p tail beyond rank L) ~ Poisson(λ + Σ_tail p). Then |T| = ΣZ + Q.
- **Probability ranking principle.** Under independence the best set is always a top-m set by p. `test_top_m_is_optimal_among_all_subsets` checks all 2^n subsets for n ≤ 7. So the only decision per S1 is m ∈ {0..L}.
- **Exact E[F(top-m)]** (`expected_f_topm`): the prefix pmf of tp (first m candidates) is convolved with the suffix-plus-background pmf of the remaining true count, then summed against G(a,c) = 1.25a/(0.25(a+c)+m). Cost is O(B·L²·(L+R)) with L = 16, R = 12. `test_exact_matches_brute_force` checks it against enumeration over 60 random cases to 1e-10.
- **m = 0:** `EF[0] = P(T empty) = Π(1−p_j)·e^(−λ)`. `empty_bias` multiplies this term before the argmax.
- **Approximations** (`expected_f_approx`):
  - `plugin`: ratio of expectations
  - `series`: leave-one-out plus a 2nd-order delta method

  Measured by `test_approximations_close_to_exact` on 5000 random S1 lists (Beta(0.4, 0.8) p, λ ~ U(0, 0.3)): series picks the same m as exact in **98.6%** of S1 with mean expected-F loss 1e-5; plugin agrees in 94.7% with loss 1.5e-4. In the simulation grid, series+bias and exact+bias differ by ≤ 0.0001 macro F.
- **Marginal add rule.** With k fixed, adding a candidate of probability p to a list with current tp, m raises expected F iff `p·(0.25k + m) > tp`, i.e. **p > F_current/1.25**.

  | current F | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 1.0 |
  |---|---|---|---|---|---|---|
  | min p to add | 0.48 | 0.56 | 0.64 | 0.72 | 0.76 | 0.80 |

- **Break-even for a lone best candidate** (exact E[F], predict {top-1} vs {}):

  | λ (missing) | other candidates | break-even p1 |
  |---|---|---|
  | 0.0 | none | 0.500 |
  | 0.0 | one at 0.1 | 0.478 |
  | 0.0 | three at 0.1 | 0.434 |
  | 0.1 | none | 0.479 |
  | 0.1 | one at 0.1 | 0.457 |
  | 0.1 | three at 0.1 | 0.413 |
  | 0.3 | none | 0.438 |
  | 0.3 | one at 0.1 | 0.416 |
  | 0.3 | three at 0.1 | 0.373 |

  Weak other candidates and missed matches both make "predict the best one" attractive at lower p1, because they lower P(T empty).

## 3. Exclusivity
- **Fact:** each S2/S3 id is in at most one S1 list (0 violations in 7.64M train links).
- **Soft** (`exclusivity_soft`): `p'_i = o_i / (1 + Σ_l o_l)`, where o = p/(1−p) and the sum runs over all S1 claiming the record (including i). This is the posterior when each claimant's odds are independent a-priori evidence and at most one claimant is right. A lone claimant keeps p exactly.

  **It is not calibrated.** The matcher's p was learned on data that already contains competitors, so dividing by 1 + Σo double-counts the competition. Two claimants at 0.8 and 0.6 become 0.62 and 0.23. In simulation, expected-F on raw p' predicted 2.44 links/S1 instead of 2.8 and lost 0.010 macro F to a threshold. Isotonic recalibration on OOF (fit p' → label) fixes it: 0.8514 → 0.8625.
- **Hard** (`exclusivity_hard`): keep the best claimant, optionally only when best − second ≥ margin. A margin grid {0, 0.05, 0.1, 0.2} tuned on world A chose 0.0.
- **`resolve()`:**
  1. Soft, hard or no exclusivity.
  2. Per-S1 decision.
  3. Up to `repair_iters` rounds: a selected edge whose record is also selected by a higher-p S1 is removed, and only the affected S1 are re-decided.
  4. Final guarantee: one S1 per record, highest p wins, ties broken by s1 id.

  `select_links` always goes through this, so exclusivity holds even with `exclusivity="none"` or `"threshold"`.
- **Best option: competition features inside the matcher** (rank of the edge in its S1, gap to the S1's best, number of claimants, best competing claimant's score, soft p' as a feature). The model then learns exclusivity with calibrated output, and soft on top slightly hurts (double counting): 0.8652 with none vs 0.8644 with soft.

## 4. Simulation model (rebuild this if needed)
One "world" = n S1 entities.
- **Matches:** k ~ the real train distribution: 0: 5.58%, 1: 5.40%, 2: 17.00%, 3: 24.05%, 4: 21.94%, 5: 14.59%, 6: 7.47%, 7: 2.90%, 8: 0.85%, 9: 0.19%, 10: 0.02%.
- **Confusable groups:** S1 are grouped into clusters of size 1-5 with probabilities (0.60, 0.20, 0.10, 0.06, 0.04), like many "Primary Care Group" S1. Each own record of a sibling appears in my list with probability q_sib = 0.5. Sibling edges are the source of exclusivity conflicts.
- **Blocking recall:** 0.97 per own record. Missed records count in k but cannot be predicted.
- **Negatives:** hard orphan negatives ~ Poisson(0.6) per S1; random distractors ~ Poisson(15). That gives about 21 candidates per S1.
- **Scores:** s = μ + N(0,1), where μ is:
  - d for easy positives (90% of positives)
  - 0.4d for hard positives (aliases, native script)
  - 0.55d for sibling and orphan negatives
  - 0 for random distractors
- **Probabilities:** p = the exact Bayes posterior under the world-A mixture. It is calibrated marginally but ignores competition, just like a pair-level matcher.
- **Parameter tuning:** all parameters (t, t0, bias, margin, isotonic map, stage-2 model) are fitted on world A and evaluated on an independent world B with the same configuration. The stage-2 model is scikit-learn HistGradientBoosting, 200 iterations, 31 leaves, on 13 context features. For tuning it uses 2-fold OOF by S1 on A.
- **Quality levels:** d ∈ {3, 4, 5, 6.5} changes pair AUC from about 0.93 to 0.99.

## 5. Simulation results
### 5.1 Main grid, d = 5 (n = 50k S1 per world, seed 11; pair AUC 0.973, AP 0.904; stage-2 AP 0.930; λ = 0.105; 21.4 edges/S1)
| rule | F | F singletons | F non-singletons | empty rate | pred/S1 | tuned |
|---|---|---|---|---|---|---|
| threshold | 0.8245 | 0.747 | 0.829 | 0.066 | 2.83 | t = 0.59 |
| threshold + fallback | 0.8257 | 0.693 | 0.833 | 0.056 | 2.84 | t = 0.59, t0 = 0.50 |
| exact, bias 1 | 0.8242 | 0.592 | 0.838 | 0.044 | 2.71 | |
| exact + bias | 0.8245 | 0.646 | 0.835 | 0.051 | 2.70 | bias 1.5 |
| series + bias | 0.8246 | 0.646 | 0.835 | 0.051 | 2.71 | bias 1.5 |
| hard + exact + bias | 0.8515 | 0.795 | 0.855 | 0.062 | 2.51 | margin 0, bias 1 |
| hard + threshold | 0.8614 | 0.797 | 0.865 | 0.063 | 2.92 | t = 0.42 |
| soft + threshold | 0.8611 | 0.810 | 0.864 | 0.065 | 2.88 | t = 0.42 |
| soft + exact + bias (uncalibrated p') | 0.8514 | 0.799 | 0.854 | 0.062 | 2.44 | bias 1 |
| soft + isotonic + exact, bias 1 | 0.8608 | 0.662 | 0.872 | 0.045 | 2.82 | |
| soft + isotonic + exact + bias | 0.8623 | 0.802 | 0.866 | 0.062 | 2.80 | bias 3 |
| soft + isotonic → resolve(none, exact, bias) | 0.8625 | 0.806 | 0.866 | 0.063 | 2.79 | bias 3 |
| stage-2 + threshold | 0.8620 | 0.836 | 0.864 | 0.070 | 2.88 | t = 0.65 |
| stage-2 + exact + bias | 0.8652 | 0.777 | 0.870 | 0.058 | 2.85 | bias 2 |
| **stage-2 + resolve(none, exact, bias)** | **0.8652** | 0.778 | 0.870 | 0.058 | 2.85 | bias 2 |
| stage-2 + resolve(soft, exact, bias) | 0.8644 | 0.763 | 0.870 | 0.056 | 2.77 | bias 1.25 |
| oracle (all in-candidate true links) | 0.9911 | 1.000 | 0.991 | 0.056 | 3.36 | |

Extending the bias grid to 8 reproduced these numbers exactly: bias 3 is an interior optimum.

### 5.2 Other quality levels and seeds
PENDING (batch running).

## 6. Speed and memory (decide.py)
PENDING.

## 7. Re-measure on real data
When the matcher exists, rerun on OOF of 2-3 complete states per country:
- threshold vs `resolve(none/soft, exact, bias)`
- soft + isotonic
- λ and `empty_bias` per country vs pooled

Log each result in `work/experiments.md`. The simulation fixes the ranking of the ideas, not the exact numbers.
