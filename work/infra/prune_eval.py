# -*- coding: utf-8 -*-
"""prune_eval.py -- candidate-set size vs quality when the stage-1 matcher is used as the last blocking filter
(keep pairs with stage-1 p1 >= tau). Test: pairs per S1 kept, v5 final links outside the kept set. OOF sample: true
links kept (pair recall), oracle ceiling, and the v1v2 decision re-run on the kept pairs only (macro F0.5).
    python infra/prune_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402

W4, W5, CE, S2D = Path("/vol/work_v4"), Path("/vol/work_v5"), Path("/vol/exp/ce"), Path("/vol/exp/ce2/s2")
TAUS = [1e-4, 1e-3, 3e-3, 1e-2]
out = {"test": {}, "oof": {}}
for c in ("France", "India", "US"):
    p1 = pl.read_parquet(CE / f"pfull_test_{c}.parquet", columns=["s1", "cand", "p"])
    links = pl.read_parquet(W5 / "pred" / f"test_{c}_links.parquet").rename({"mid": "cand"})
    n = pl.scan_parquet(W4 / "norm" / f"test_{c}_s1.parquet").select(pl.len()).collect().item()
    lk = links.join(p1, on=["s1", "cand"], how="left")
    r = {"n_s1": n, "pairs_per_s1_all": p1.height / n, "links": links.height}
    for t in TAUS:
        r[f"tau {t}"] = {"pairs_per_s1": p1.filter(pl.col("p") >= t).height / n,
                         "links_below_tau": int((lk["p"] < t).sum())}
    out["test"][c] = r
    print(c, json.dumps(r), flush=True)
tune = json.load(open(S2D / "tune_v1v2.json"))
oof = pl.read_parquet(S2D / "oof_v1v2.parquet").join(
    pl.read_parquet(W4 / "model" / "oof.parquet").select("s1", "cand", pl.col("p").alias("p1")), on=["s1", "cand"], how="left")
ids = oof["s1"].unique()
gt = bio.scan_split(W4, "train")["gt"].collect().join(ids.to_frame("s1"), on="s1", how="semi")
kw = dict(exclusivity="soft", lam_missing=float(tune["lam_missing"]), empty_bias=float(tune["eb_best_oof"]))
base = D.select_links(oof.select("s1", "cand", "p"), "expected_f", **kw)
out["oof"]["all"] = {"pairs_per_s1": oof.height / ids.len(), "F": M.macro_f05(base, gt, ids),
                     "true_links_in_cands": int(oof["label"].sum()), "true_links": gt.height}
for t in TAUS:
    k = oof.filter(pl.col("p1") >= t)
    lk = D.select_links(k.select("s1", "cand", "p"), "expected_f", **kw)
    oracle = M.macro_f05(k.filter(pl.col("label") == 1).select("s1", pl.col("cand").alias("mid")), gt, ids)
    out["oof"][f"tau {t}"] = {"pairs_per_s1": k.height / ids.len(), "true_links_kept": int(k["label"].sum()),
                              "F": M.macro_f05(lk, gt, ids), "oracle": oracle,
                              "links_changed_vs_all": int(lk.join(base, on=["s1", "mid"], how="anti").height
                                                          + base.join(lk, on=["s1", "mid"], how="anti").height)}
    print(t, json.dumps(out["oof"][f"tau {t}"]), flush=True)
print(json.dumps(out["oof"]["all"]))
json.dump(out, open(S2D / "prune_eval.json", "w"), indent=1)
