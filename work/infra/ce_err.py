# -*- coding: utf-8 -*-
"""ce_err.py -- where the remaining OOF loss of the best stage-2 model sits (v1v2 by default).
Counterfactual macro F0.5 on the OOF sample (331,205 S1): actual decision; + every missed in-candidate true link added;
- every false link removed; both (= oracle on the candidates); and the same fixes restricted to stage-1 p ranges
(what a wider cross-encoder band could reach at most). Also per-S1 loss by error type.
    python infra/ce_err.py [--variant v1v2]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="v1v2")
a = ap.parse_args()
W, S2D = Path("/vol/work_v4"), Path("/vol/exp/ce2/s2")
tune = json.load(open(S2D / f"tune_{a.variant}.json"))
lam, eb = float(tune["lam_missing"]), float(tune["eb_best_oof"])
oof = pl.read_parquet(S2D / f"oof_{a.variant}.parquet")                    # s1, cand, label, p (final), country
p1 = pl.read_parquet(W / "model" / "oof.parquet").select("s1", "cand", pl.col("p").alias("p1"))
oof = oof.join(p1, on=["s1", "cand"], how="left")
ids = oof["s1"].unique()
gt = bio.scan_split(W, "train")["gt"].collect().join(ids.to_frame("s1"), on="s1", how="semi")
links = D.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=lam, empty_bias=eb)
sel = oof.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left").with_columns(
    pl.col("sel").fill_null(False))
F = lambda lk: M.macro_f05(lk.select("s1", pl.col("cand").alias("mid")) if "cand" in lk.columns else lk, gt, ids)  # noqa: E731
base = F(sel.filter("sel"))
fn = sel.filter(~pl.col("sel") & (pl.col("label") == 1))
fp = sel.filter(pl.col("sel") & (pl.col("label") == 0))
tp = sel.filter(pl.col("sel") & (pl.col("label") == 1))
out = {"variant": a.variant, "n_s1": len(ids), "true_links": gt.height, "in_candidates": int(oof["label"].sum()),
       "selected": int(sel["sel"].sum()), "tp": tp.height, "fp": fp.height, "fn_in_candidates": fn.height,
       "fn_blocking": gt.height - int(oof["label"].sum()), "F_actual": base,
       "F_add_all_fn": F(pl.concat([tp, fp, fn])), "F_drop_all_fp": F(tp), "F_oracle": F(pl.concat([tp, fn]))}
bands = [("<0.001", 0, 1e-3), ("0.001-0.01", 1e-3, 0.01), ("0.01-0.99 (CE band)", 0.01, 0.99 + 1e-9),
         ("0.99-0.999", 0.99 + 1e-9, 0.999), (">=0.999", 0.999, 2)]
rows = []
for name, lo, hi in bands:
    inb = pl.col("p1").is_between(lo, hi, closed="left")
    fix = pl.concat([tp, fp.filter(~inb), fn.filter(inb)])      # fix FN and FP of pairs in this stage-1 band only
    rows.append({"stage1_band": name, "fp": fp.filter(inb).height, "fn": fn.filter(inb).height,
                 "F_if_band_perfect": F(fix), "gain": F(fix) - base})
out["by_stage1_band"] = rows
print(json.dumps(out, indent=1))
json.dump(out, open(S2D / f"err_{a.variant}.json", "w"), indent=1)
