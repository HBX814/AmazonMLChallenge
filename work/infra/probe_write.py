# -*- coding: utf-8 -*-
"""probe_write.py -- leaderboard probes from the expB test scores (v3 stage-1 + ber/stage2.py G1):
    s2g1_density  : stage-2 G1 + adapt density on the HARD classes (source = OOF stage-2 priors, adapt_source_s2.json)
    s2g1_rule     : stage-2 G1 + targeted rule (drop HARD links when the S1 has an exact-house-number link, p < 0.99)
Candidate pairs = the v3 scored pairs (unchanged). Writes /vol/exp/probes/<name>/{matching_results,candidate_pairs}.tsv
"""
import json
import os
import sys

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                     # noqa: E402

from ber import adapt as A              # noqa: E402
from ber import decide as D             # noqa: E402
from ber import outputs as O            # noqa: E402

W3, EB, OUT = "/vol/work_v3", "/vol/exp/expB", "/vol/exp/probes"
LAM = 0.05303060038344832
AR = json.load(open(f"{EB}/a_results.json"))
EB2 = max((1.0, 1.5, 2.0), key=lambda eb: AR["mine_OOF_G1"][f"eb{eb}"]["ALL"])
SRC = A.load_source(f"{EB}/adapt_source_s2.json")
ACFG = A.AdaptConfig(method="density", classes=list(A.HARD_CLASSES))
res = {"eb": EB2}
scored_all, links = [], {"s2g1_density": [], "s2g1_rule": []}
for c in ("France", "India", "US"):
    sc = pl.read_parquet(f"{EB}/test_{c}_s2.parquet").select("s1", "cand", pl.col("p_g1").alias("p"))
    n_s1 = pl.scan_parquet(f"{W3}/norm/test_{c}_s1.parquet").select(pl.len()).collect().item()
    tagged = A.tag_pairs(sc, f"{W3}/norm/test_{c}_s1.parquet", f"{W3}/norm/test_{c}_pool.parquet", p_min=ACFG.p_min)
    ad, est = A.adapt(tagged, SRC, ACFG, country=c, n_s1=n_s1)
    lk_d = D.select_links(ad.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB2)
    lk_g1 = D.select_links(sc, "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB2)
    lk_r = A.targeted_rule(lk_g1, tagged)
    links["s2g1_density"].append(lk_d)
    links["s2g1_rule"].append(lk_r)
    scored_all.append(sc)
    res[c] = {"n_s1": n_s1, "links_per_s1_g1": round(lk_g1.height / n_s1, 4),
              "links_per_s1_density": round(lk_d.height / n_s1, 4), "links_per_s1_rule": round(lk_r.height / n_s1, 4),
              "estimates": est.filter(pl.col("factor") != 1.0).select("hn_rel", "pairs_per_s1", "pairs_per_s1_src",
                                                                        "prior_s", "prior_t", "factor").to_dicts()}
    print(c, json.dumps({k: v for k, v in res[c].items() if k != "estimates"}), res[c]["estimates"], flush=True)
scored = pl.concat(scored_all)
s1_ids = O.read_s1_order("/root/proj/student_resource/dataset/test/test_source1.tsv")
for name, parts in links.items():
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    rep = O.write_submission(s1_ids, pl.concat(parts), scored, d)
    print(name, rep, flush=True)
json.dump(res, open(f"{OUT}/probes.json", "w"), indent=1, default=str)
