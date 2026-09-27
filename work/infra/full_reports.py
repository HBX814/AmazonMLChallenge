# -*- coding: utf-8 -*-
"""full_reports.py -- reports after a full run (run from the project root inside the container).

    python infra/full_reports.py oof     # OOF links (best decision) + labelled error report on the train sample
    python infra/full_reports.py shift   # label-free: test per country vs OOF (links/S1, empty rate, p histogram)

Outputs: work/model/oof_links.parquet, work/reports/err_full_oof.md, work/reports/shift_<country>.md,
work/reports/test_country_stats.json
"""
import json
import os
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "business_entity_resolution", "src"))
sys.path.insert(0, os.path.join(HERE, "..", ".claude", "skills", "er-error-analysis"))

from ber import decide as D   # noqa: E402

W = os.environ.get("BER_WORK", "work")


def _safe(c):
    return "".join(ch if ch.isalnum() else "_" for ch in (c or "EMPTY"))


def _countries(split):
    return sorted(pl.scan_parquet(f"{W}/cache/{split}_source1.parquet").select("country").unique().collect()["country"].to_list())


def _best():
    return json.load(open(f"{W}/model/decision.json", encoding="utf-8"))["best"]


def oof_links():
    best = _best()
    oof = pl.read_parquet(f"{W}/model/oof.parquet")
    kw = {k: best[k] for k in ("empty_bias", "lam_missing", "t") if k in best}
    links = D.select_links(oof, best["method"], exclusivity=best.get("exclusivity", "none"), **kw)
    links.write_parquet(f"{W}/model/oof_links.parquet")
    return oof, links


def cmd_oof():
    import error_report as ER
    oof, links = oof_links()
    ids = oof.select("s1").unique()
    s1 = pl.concat([pl.read_parquet(f"{W}/norm/train_{_safe(c)}_s1.parquet") for c in _countries("train")]) \
        .join(ids.rename({"s1": "entity_id"}), on="entity_id", how="semi")
    cand_ids = oof.select(pl.col("cand").alias("entity_id")).unique()
    gt = pl.read_parquet(f"{W}/cache/train_gt_long.parquet").join(ids, on="s1", how="semi")
    pool = pl.concat([pl.read_parquet(f"{W}/norm/train_{_safe(c)}_pool.parquet") for c in _countries("train")]) \
        .join(pl.concat([cand_ids, gt.select(pl.col("mid").alias("entity_id"))]).unique(), on="entity_id", how="semi")
    # evaluated S1 = the sampled train S1 (from decision.json population): OOF S1 plus sampled S1 without candidates
    rep = ER.write_report(f"{W}/reports/err_full_oof.md", scored=oof, links=links, gt=gt, s1=s1, pool=pool,
                          s1_ids=None, n_examples=6, max_tag_pairs=60_000, seed=0, title="full OOF (train sample)")
    h = rep["headline"]
    print(f"full OOF macro F0.5={h['macro_f']:.5f} ceiling={h['blocking_ceiling_f']:.5f} oracle_cut={h['oracle_cut_f']:.5f}")


def cmd_shift():
    import error_report as ER
    oof, links = (pl.read_parquet(f"{W}/model/oof.parquet"), pl.read_parquet(f"{W}/model/oof_links.parquet")) \
        if os.path.exists(f"{W}/model/oof_links.parquet") else oof_links()
    s1_tr = pl.concat([pl.read_parquet(f"{W}/norm/train_{_safe(c)}_s1.parquet", columns=["entity_id", "country"])
                       for c in _countries("train")]) \
        .join(oof.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id", how="semi")   # OOF population
    stats = {}
    for c in _countries("test"):
        sc = pl.read_parquet(f"{W}/pred/test_{_safe(c)}_scored.parquet")
        lk = pl.read_parquet(f"{W}/pred/test_{_safe(c)}_links.parquet")
        s1 = pl.read_parquet(f"{W}/norm/test_{_safe(c)}_s1.parquet", columns=["entity_id", "country"])
        n = s1.height
        per = lk.group_by("s1").len()
        stats[c] = {"n_s1": n, "pairs": sc.height, "cands_per_s1": sc.height / n, "links": lk.height,
                    "links_per_s1": lk.height / n, "empty_rate": 1 - per.height / n,
                    "p_hist": sc.select(pl.col("p").cut([0.1, 0.3, 0.5, 0.7, 0.9]).value_counts()).unnest("p").sort("p")
                                .with_columns(pl.col("p").cast(pl.Utf8)).rows()}
        md, _ = ER.shift_report({"scored": oof, "links": links, "s1": s1_tr},
                                {"scored": sc, "links": lk, "s1": s1}, None, None, frac=0.1)
        with open(f"{W}/reports/shift_{_safe(c)}.md", "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        print(f"[shift] {c}: {json.dumps({k: v for k, v in stats[c].items() if k != 'p_hist'})}", flush=True)
    per = links.group_by("s1").len()
    n_oof = oof.select("s1").n_unique()
    stats["OOF(train sample, S1 with >=1 candidate)"] = {"n_s1": n_oof, "links_per_s1": links.height / n_oof,
                                                         "empty_rate": 1 - per.height / n_oof,
                                                         "cands_per_s1": oof.height / n_oof}
    with open(f"{W}/reports/test_country_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, default=str)
    print(json.dumps(stats["OOF(train sample, S1 with >=1 candidate)"]))


if __name__ == "__main__":
    os.makedirs(f"{W}/reports", exist_ok=True)
    {"oof": cmd_oof, "shift": cmd_shift}[sys.argv[1]]()
