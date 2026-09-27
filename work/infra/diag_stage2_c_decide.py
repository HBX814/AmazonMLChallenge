# -*- coding: utf-8 -*-
"""diag_stage2_c_decide.py -- cheap decision-level ideas on stage-1 p (no new model), on the OOF population (grp 0,
OOF p) and on the honest holdout HB (grp 1, final-model p, full-frame competition). Needs diag_stage2_a_prep outputs.

Ideas: lam_missing grid, calibration power p^g, hard exclusivity with margins, soft exclusivity against the FULL
frame (soft_ext = test-like competition), per-S1 caps, threshold + fallback, expected-F list cap L, sibling
propagation (add / raise a candidate whose same-name-key sibling is selected with high p).
Writes /vol/diag/stage2/decide_ideas.json
"""
import json
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import polars as pl           # noqa: E402
import psutil                 # noqa: E402

from ber import decide as D   # noqa: E402
import diag_stage2_lib as L   # noqa: E402

OUT = "/vol/diag/stage2"
LAM, EB = 0.05303060038344832, 2.0
T0 = time.time()


def log(*a):
    rss = psutil.Process().memory_info().rss / 2 ** 30
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s rss {rss:5.1f}G]", *a, flush=True)


def cap_links(links, scored, k):
    x = links.join(scored.select("s1", pl.col("cand").alias("mid"), "p"), on=["s1", "mid"], how="left")
    x = x.sort(["s1", "p", "mid"], descending=[False, True, False]).with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
    return x.filter(pl.col("_r") < k).select("s1", "mid")


def thr_fallback(scored, t, ft):
    e = scored.filter(pl.col("p") >= min(ft, t)).sort(["s1", "p", "cand"], descending=[False, True, False])
    e = e.with_columns(pl.int_range(pl.len()).over("s1").alias("_r"), pl.col("p").max().over("s1").alias("_m"))
    sel = (pl.col("p") >= t) | ((pl.col("_m") < t) & (pl.col("_r") == 0) & (pl.col("p") >= ft))
    s = e.filter(sel).select("s1", "cand", "p")
    s = s.sort(["cand", "p"], descending=[False, True]).unique("cand", keep="first")   # exclusivity guarantee
    return s.select("s1", pl.col("cand").alias("mid"))


def sib_add(scored, links, names, labels, t_low, pd_min=0.9, empty_addr_only=False):
    sel = links.select("s1", pl.col("mid").alias("cand"), pl.lit(True).alias("_sel"))
    e = (scored.join(sel, on=["s1", "cand"], how="left").with_columns(pl.col("_sel").fill_null(False))
               .join(names.select("cand", "nk", "ad"), on="cand", how="left")
               .with_columns(pl.col("nk").fill_null(""), pl.col("ad").fill_null("")))
    anchors = e.filter(pl.col("_sel") & (pl.col("p") >= pd_min) & (pl.col("nk") != "")).select("s1", "nk").unique()
    add = e.filter(~pl.col("_sel") & (pl.col("p") >= t_low) & (pl.col("nk") != "")).join(anchors, on=["s1", "nk"], how="semi")
    if empty_addr_only:
        add = add.filter(pl.col("ad") == "")
    add = add.join(links.select(pl.col("mid").alias("cand")), on="cand", how="anti")
    add = add.sort(["cand", "p"], descending=[False, True]).unique("cand", keep="first")
    lab = add.join(labels, on=["s1", "cand"], how="left")["label"].fill_null(0)
    out = pl.concat([links, add.select("s1", pl.col("cand").alias("mid"))])
    return out, {"n_added": add.height, "added_precision": float(lab.mean()) if add.height else None}


def sib_raise(scored, names, alpha, pd_min=0.9):
    e = scored.join(names.select("cand", "nk"), on="cand", how="left").with_columns(pl.col("nk").fill_null(""))
    mx = e.filter((pl.col("p") >= pd_min) & (pl.col("nk") != "")).group_by("s1", "nk").agg(pl.col("p").max().alias("_pk"))
    e = e.join(mx, on=["s1", "nk"], how="left").with_columns(
        pl.max_horizontal(pl.col("p"), alpha * pl.col("_pk").fill_null(0.0)).alias("p"))
    return e.select("s1", "cand", "p")


def run(tag, scored, labels, gt, ev, names, osum_ext):
    R = {}

    def rec(name, links, extra=None):
        s = L.score(links, gt, ev)
        R[name] = {**s, "links_per_s1": round(links.height / ev.height, 4), **(extra or {})}
        log(f"[{tag}] {name:34s} {s}" + (f" {extra}" if extra else ""))

    base = L.select_fast(scored, "soft", LAM, EB)
    rec("baseline soft eb2 lam.053", base)
    rec("none eb2", L.select_fast(scored, "none", LAM, EB))
    for lam in (0.0, 0.02, 0.08, 0.12, 0.2):
        rec(f"soft eb2 lam{lam}", L.select_fast(scored, "soft", lam, EB))
    for eb in (1.5, 3.0):
        rec(f"soft_ext eb{eb}", L.select_fast(scored, "soft_ext", LAM, eb, osum_ext=osum_ext))
    rec("soft_ext eb2", L.select_fast(scored, "soft_ext", LAM, EB, osum_ext=osum_ext))
    for g in (0.7, 0.85, 1.15, 1.3):
        sc = scored.with_columns(pl.col("p").clip(0.0, 1.0).pow(g))
        rec(f"power {g} soft eb2", L.select_fast(sc, "soft", LAM, EB))
    for m in (0.0, 0.05, 0.1, 0.2, 0.3):
        rec(f"hard margin {m} eb2", L.select_fast(scored, "hard", LAM, EB, hard_margin=m))
    for k in (3, 4, 5, 6, 8, 10):
        rec(f"cap {k} on baseline", cap_links(base, scored, k))
    rec("L=24 soft eb2", L.select_fast(scored, "soft", LAM, EB, L=24))
    for t, ft in ((0.65, 0.2), (0.65, 0.3), (0.7, 0.2), (0.7, 0.3), (0.7, 0.4), (0.75, 0.3)):
        rec(f"thr {t} fallback {ft}", thr_fallback(scored, t, ft))
    for t_low in (0.02, 0.05, 0.1, 0.2, 0.3):
        for eo in (False, True):
            lk, ex = sib_add(scored, base, names, labels, t_low, empty_addr_only=eo)
            rec(f"sib_add t{t_low}{' emptyaddr' if eo else ''}", lk, ex)
    for alpha in (0.5, 0.7, 0.9):
        rec(f"sib_raise a{alpha} soft eb2", L.select_fast(sib_raise(scored, names, alpha), "soft", LAM, EB))
    return R


def main():
    meta = pl.read_parquet(f"{OUT}/meta.parquet")
    gt = pl.read_parquet(f"{OUT}/gt.parquet")
    names = pl.read_parquet(f"{OUT}/names.parquet")
    osum_full = pl.read_parquet(f"{OUT}/osum_full.parquet")
    pf = pl.scan_parquet(f"{OUT}/p_full.parquet")
    res = {}
    for grp, tag in ((0, "OOF"), (1, "HB")):
        ev = meta.filter(pl.col("grp") == grp).select("s1", "country")
        g = gt.join(ev, on="s1", how="semi")
        fr = pf.filter(pl.col("grp") == grp).select("s1", "cand", "label", "p1").collect()
        scored = fr.select("s1", "cand", pl.col("p1").alias("p"))
        labels = fr.select("s1", "cand", "label")
        ext = L.external_osum(osum_full, fr)
        log(f"[{tag}] {ev.height:,} S1, {fr.height:,} edges")
        res[tag] = run(tag, scored, labels, g, ev, names, ext)
        json.dump(res, open(f"{OUT}/decide_ideas.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
