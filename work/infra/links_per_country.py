import json, sys
sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl
from ber import decide as D
S2D = "/vol/exp/ce2/s2"
t = json.load(open(f"{S2D}/tune_v1v2.json"))
oof = pl.read_parquet(f"{S2D}/oof_v1v2.parquet")
lk = D.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=float(t["lam_missing"]), empty_bias=float(t["eb_best_oof"]))
lk = lk.join(oof.select("s1", "cand", "label", "country").rename({"cand": "mid"}), on=["s1", "mid"], how="left")
n = oof.group_by("country").agg(pl.col("s1").n_unique().alias("n"))
sp = oof.filter(pl.col("p") >= 1e-3).group_by("country").agg(pl.col("p").sum().alias("sum_p"))
r = lk.group_by("country").agg(pl.len().alias("links"), (1 - pl.col("label")).sum().alias("fp")).join(n, on="country").join(sp, on="country")
print(r.with_columns((pl.col("links") / pl.col("n")).alias("links_per_s1"), (pl.col("fp") / pl.col("n")).alias("fp_per_s1"), (pl.col("sum_p") / pl.col("n")).alias("sum_p_per_s1")))
