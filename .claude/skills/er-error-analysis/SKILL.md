---
name: er-error-analysis
description: Use when the Master Bolt entity-resolution validation macro F0.5 stalls or drops, when deciding what to improve next, when analysing false merges or missed matches, when leaderboard and validation scores disagree, when checking France or other test data without labels for distribution shift, or when logging experiments in work/experiments.md.
---

# Error analysis and the iteration loop

## Overview
Improve by **expected gain, not by guess**: decompose `1 − macro F0.5` into exact loss components, find the bucket that holds the most loss, tag the errors by cause, make **one** change, re-measure on the same frozen folds, log it. Every claim of improvement needs a logged number.

## When to use
- After every OOF run; before choosing the next change; when test/leaderboard behaviour looks different from OOF; before the final submission.

## Quick reference
| Task | Command (copy `error_report.py` to `src/ber/` or run from this folder) |
|---|---|
| Labelled report (validation fold) | `python error_report.py report --scored work/oof/scored.parquet --links work/oof/links.parquet --gt work/cache/gt_long.parquet --s1 work/norm/train_s1.parquet --pool work/norm/train_pool.parquet --ids work/folds/val_fold0.txt --out work/reports/err_<date>_fold0.md` |
| Label-free shift (test / France vs OOF) | `python error_report.py shift --ref-scored ... --ref-links ... --ref-s1 ... --cur-scored work/test/scored.parquet --cur-links work/test/links.parquet --cur-s1 work/norm/test_s1.parquet --out work/reports/shift.md` (+ `--ref-features/--cur-features` for PSI) |
| Manual sheet | `python error_report.py inspect --scored ... --links ... --s1 ... --pool ... --country France --n 50 --out work/reports/france_inspect.md` |
| Library | `build_report`, `render_markdown`, `per_s1_table`, `loss_decomposition`, `tag_error_pairs`, `unlabeled_stats`, `shift_report`, `psi` |
| Log format | `experiments_template.md` (this folder) → `work/experiments.md` |
| Tests | `python -m pytest -q -p no:cacheprovider test_error_report.py` (13 pass; 12k-S1 mini report in ~34 s) |

## What the report gives you
- Macro F0.5 overall / per country / per true-k bucket (0,1,…,7+) / per predicted-size bucket; blocking ceiling F and oracle-cut F (best possible with this ranking) — the gaps say whether to work on blocking, the model or the decision rule.
- `loss_decomposition`: components that sum exactly to `1 − F`: FP on singletons, empty prediction on matched S1, partial recall, false merges, blocking misses.
- Error pairs split into FP / FN_model / FN_blocking and tagged: `generic_name_same_city`, `house_number_mismatch`, `alias_name`, `native_script`, `domain_form`, `truncated_name`, `empty_address`, `legal_form_only_diff`, `cross_duplicate`, `competition_loss` (candidate went to another S1).

## The loop
1. Freeze the folds (er-data-loading `make_folds`, seed 0). Never change them mid-competition.
2. Run the report; pick the bucket with the largest `share × loss`.
3. Read 20 tagged examples of that bucket; write one hypothesis.
4. Make one change (normalization rule, blocking pass/K, feature, param, decision parameter).
5. Re-run the affected stages on the dev slice, then full OOF; log a line in `work/experiments.md` with Δ macro F0.5 overall/US/India, blocking recall, runtime, RAM. Keep the change only if OOF improves (or is neutral and simplifies).

Where the points usually are (from the data facts): matched S1 predicted empty (costs 1.0 each; 94% of S1 have matches), generic names in the same city (US medical "… group" names ×200), house-number noise (1804→1804A, 3412→3411), native-script names (India), alias names that only match on address, and FP on the 5.58% singletons.

## France and test without labels
- LOCO (train US → India and back) is the proxy for zero-shot France (**See also:** er-matcher-training).
- `shift` report: predicted links per S1, empty-rate, score histograms and feature PSI per country, test vs OOF. France far below the others in links/S1 means blocking or normalization is failing there; far above means false merges (generic French vocabulary: club, ecole, amicale, sarl…).
- `inspect` sheet: read 50 France S1 with their top-3 candidates; count obvious errors; fix normalization/blocking rules that are language-agnostic.
- Test pools are ~23% denser per S1 than train: if the test links/S1 rate exceeds OOF, suspect distractors → tighten slightly rather than loosen.

## Common mistakes
| Mistake | Fix |
|---|---|
| Changing several things at once | one change per log line |
| Tuning to the public leaderboard (a small subset of test) | OOF on frozen folds is the objective; leaderboard = sanity check |
| Pair-level metrics (AUC, pair F1) as the target | macro F0.5 per S1 incl. singletons |
| Ignoring S1 with zero candidates | they stay in the denominator (blocking ceiling) |
| Fixing France by country-specific hacks that break the open-set rule | language-agnostic rules; generic fallbacks |
| No record of what was tried | `work/experiments.md`, append-only |
