# -*- coding: utf-8 -*-
"""ce_err2.py -- empty-address share of the remaining OOF loss of the best stage-2 model (v1v2).
    python infra/ce_err2.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402

W, S2D = Path("/vol/work_v4"), Path("/vol/exp/ce2/s2")
tune = json.load(open(S2D / "tune_v1v2.json"))
oof = pl.read_parquet(S2D / "oof_v1v2.parquet")
ids = oof["s1"].unique()
gt = bio.scan_split(W, "train")["gt"].collect().join(ids.to_frame("s1"), on="s1", how="semi")
links = D.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft",
                       lam_missing=float(tune["lam_missing"]), empty_bias=float(tune["eb_best_oof"]))
emp = pl.concat([pl.read_parquet(W / "norm" / f"train_{c}_pool.parquet", columns=["entity_id", "business_address"])
                 for c in ("India", "US")]).select(
    pl.col("entity_id").alias("cand"), (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty_addr"))
sel = (oof.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left")
          .with_columns(pl.col("sel").fill_null(False)).join(emp, on="cand", how="left"))
tp = sel.filter(pl.col("sel") & (pl.col("label") == 1))
fp = sel.filter(pl.col("sel") & (pl.col("label") == 0))
fn = sel.filter(~pl.col("sel") & (pl.col("label") == 1))
F = lambda d: M.macro_f05(d.select("s1", pl.col("cand").alias("mid")), gt, ids)  # noqa: E731
base = F(sel.filter("sel"))
per = M.per_s1_scores(sel.filter("sel").select("s1", pl.col("cand").alias("mid")), gt, ids)
npred = sel.group_by("s1").agg(pl.col("sel").sum().alias("msel"), pl.col("label").sum().alias("k_in"))
per = per.join(npred, on="s1", how="left").with_columns(pl.col("msel").fill_null(0), pl.col("k_in").fill_null(0))
n = len(ids)
out = {"F": base,
       "fn_total": fn.height, "fn_empty_addr": int(fn["empty_addr"].sum()),
       "fp_total": fp.height, "fp_empty_addr": int(fp["empty_addr"].sum()),
       "F_fix_empty_addr_fn": F(pl.concat([tp, fp, fn.filter("empty_addr")])),
       "F_fix_empty_addr_fn_fp": F(pl.concat([tp, fp.filter(~pl.col("empty_addr")), fn.filter("empty_addr")])),
       "F_fix_nonempty_fn": F(pl.concat([tp, fp, fn.filter(~pl.col("empty_addr"))])),
       "loss_empty_list_with_true_candidate": float(per.filter((pl.col("msel") == 0) & (pl.col("k_in") > 0))
                                                     .select((1 - pl.col("f")).sum()).item()) / n,
       "n_empty_list_with_true_candidate": per.filter((pl.col("msel") == 0) & (pl.col("k_in") > 0)).height,
       "loss_singleton_fp": float(per.filter(pl.col("msel") > 0).join(gt.select("s1").unique(), on="s1", how="anti")
                                  .select((1 - pl.col("f")).sum()).item()) / n}
print(json.dumps(out, indent=1))
