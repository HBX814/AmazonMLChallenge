# -*- coding: utf-8 -*-
"""measure_blocking.py -- measure blocking recall on a labelled slice: per-pass table, cumulative union (marginal
recall of each pass), leave-one-out loss, K / cap trade-off curves, timings and peak RSS.

Contract functions used: blocking.generate_candidates, blocking.candidate_recall (this folder),
normalize.name_variants / address_all / name_flags (er-text-normalization) via normalize_frame() below.

Input slice = three parquet (or TSV) files with raw columns:
  <slice>_s1.parquet    entity_id, business_name, business_address, country
  <slice>_pool.parquet  entity_id, business_name, business_address, country   (S2 u S3; extra columns ignored)
  <slice>_links.parquet s1, mid                                              (ground truth, long form)
The slice's own ceiling (links whose target is not in the slice pool) is reported first: blocking cannot beat it.

Usage (Windows, project venv):
  set PYTHONDONTWRITEBYTECODE=1 & set PYTHONIOENCODING=utf-8
  .venv\\Scripts\\python .claude\\skills\\er-blocking-candidates\\measure_blocking.py --slice IN_KA ^
      --bench-dir <dir with IN_KA_*.parquet> --out-dir work\\blocking --native-dict work\\native_token_dict.tsv
  options: --n-jobs 3 (normalization workers) --cap 80 --no-save --passes key_name,key_tok,... (union order)
Outputs in --out-dir: <slice>_{s1,pool}_norm.parquet (cached, reused), <slice>_cands.parquet (capped union with a
`label` column), <slice>_blocking_report.json, and the tables on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "er-text-normalization"))

_NORM_READY = False


# =====================================================================================================
# normalization of a slice (distinct strings only, spawn-safe multiprocessing)
# =====================================================================================================
def _init_worker(native_dict):
    global _NORM_READY
    import normalize as N
    if native_dict and os.path.exists(native_dict):
        N.load_native_dict(native_dict)
    _NORM_READY = True


def _norm_names(chunk):
    import normalize as N
    out = []
    for name, country in chunk:
        v = N.name_variants(name, country)
        fl = N.name_flags(name)
        script = "indic" if fl.get("native_script") else ("latin" if any(ch.isalpha() for ch in name) else "other")
        out.append((name, country, v["norm"], v["core"], v["glued"], script, bool(fl.get("domain") or fl.get("handle"))))
    return out


def _pick_city(parts, ck):
    import normalize as N
    cands = parts.get("city_candidates") or []
    if ck == "france":
        for c in cands:
            if c in N.FR_CITIES:
                return c
    return cands[-1] if cands else ""


def _norm_addrs(chunk):
    import normalize as N
    out = []
    for addr, country in chunk:
        norm, parts = N.address_all(addr, country)
        hn = list(parts.get("house_numbers") or [])
        for v in (parts.get("designated_numbers") or {}).values():
            hn.extend(v)
        out.append((addr, country, norm, hn, _pick_city(parts, N.country_key(country)), parts.get("state_canonical") or ""))
    return out


def _parallel_map(fn, items, n_jobs, native_dict):
    if not items:
        return []
    size = max(2000, len(items) // (max(1, n_jobs) * 8) + 1)
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    if n_jobs <= 1:
        _init_worker(native_dict)
        return [r for ch in chunks for r in fn(ch)]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_jobs, initializer=_init_worker, initargs=(native_dict,)) as pool:
        return [r for res in pool.imap(fn, chunks, chunksize=1) for r in res]


def normalize_frame(df: pl.DataFrame, n_jobs: int = 3, native_dict: str | None = None) -> pl.DataFrame:
    """Raw source frame -> + contract normalized columns (name_norm, name_core, name_compact, addr_norm, house_nums,
    city_key, state_key, script, name_is_domain). Each DISTINCT (string, country) is normalized once.
    Prefer ber.normalize.add_normalized_columns in the pipeline; this is the slice-measurement equivalent."""
    df = df.with_columns([pl.col(c).cast(pl.Utf8).fill_null("") for c in ("business_name", "business_address", "country")])
    names = df.select("business_name", "country").unique().rows()
    t = time.time()
    rn = _parallel_map(_norm_names, names, n_jobs, native_dict)
    nm = pl.DataFrame(rn, schema={"business_name": pl.Utf8, "country": pl.Utf8, "name_norm": pl.Utf8, "name_core": pl.Utf8,
                                  "name_compact": pl.Utf8, "script": pl.Utf8, "name_is_domain": pl.Boolean}, orient="row")
    print(f"  normalized {len(names):,} distinct names in {time.time() - t:.1f}s", flush=True)
    addrs = df.select("business_address", "country").unique().rows()
    t = time.time()
    ra = _parallel_map(_norm_addrs, addrs, n_jobs, native_dict)
    ad = pl.DataFrame(ra, schema={"business_address": pl.Utf8, "country": pl.Utf8, "addr_norm": pl.Utf8,
                                  "house_nums": pl.List(pl.Utf8), "city_key": pl.Utf8, "state_key": pl.Utf8}, orient="row")
    print(f"  normalized {len(addrs):,} distinct addresses in {time.time() - t:.1f}s", flush=True)
    return (df.join(nm, on=["business_name", "country"], how="left", maintain_order="left")
              .join(ad, on=["business_address", "country"], how="left", maintain_order="left"))


# =====================================================================================================
# reporting
# =====================================================================================================
def _read(path):
    if path.endswith(".parquet"):
        return pl.read_parquet(path)
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)   # never quote-parse these TSVs
    return df.with_columns(pl.all().fill_null(""))


def _fmt(r):
    return (f"recall {r['pair_recall']:.4f}  full-S1 {r['s1_full_recall']:.4f}  zero-S1 {r['s1_zero_recall']:.4f}  "
            f"cands/S1 {r['cands_per_s1_mean']:6.1f} (p50 {r['cands_per_s1_p50']:.0f}, p95 {r['cands_per_s1_p95']:.0f})  "
            f"oracleF0.5 {r['oracle_f05']:.4f}")


def pass_table(cands: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.DataFrame, order: list) -> list:
    """per pass alone, cumulative union in `order`, and leave-one-out loss (links found ONLY by that pass)."""
    import blocking as B
    rows = []
    union = pl.lit(False)
    lab = cands.join(gt.select("s1", pl.col("mid").alias("cand"), pl.lit(True).alias("y")), on=["s1", "cand"], how="left") \
               .with_columns(pl.col("y").fill_null(False))
    n_true = gt.join(s1_ids.select(pl.col("entity_id").alias("s1")), on="s1", how="semi").height
    hits = lab.select([B.pass_hit_expr(p).alias(p) for p in order] + ["y", "s1"])
    n_s1 = s1_ids.height
    for p in order:
        alone = hits.filter(pl.col(p))
        union = union | pl.col(p)
        cum = hits.filter(union)
        others = pl.any_horizontal([pl.col(o) for o in order if o != p]) if len(order) > 1 else pl.lit(False)
        only = hits.filter(pl.col(p) & ~others & pl.col("y")).height
        rows.append({"pass": p, "pairs": alone.height, "alone_recall": alone["y"].sum() / n_true,
                     "alone_cands_per_s1": alone.height / n_s1, "cum_recall": cum["y"].sum() / n_true,
                     "cum_cands_per_s1": cum.height / n_s1, "only_this_pass": only / n_true})
    prev = 0.0
    for r in rows:
        r["marginal"] = r["cum_recall"] - prev
        prev = r["cum_recall"]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--native-dict", default=None)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--cap", type=int, default=80, help="per-S1 cap for the saved candidate file")
    ap.add_argument("--caps", default="10,20,30,40,60,80,100,150")
    ap.add_argument("--passes", default=None, help="comma list: union order of the table (default = config order)")
    ap.add_argument("--tfidf", default=None, help="comma list of TF-IDF fields (default config)")
    ap.add_argument("--keys", default=None, help="comma list of key passes (default config)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--city-alt", action="store_true", help="BlockingConfig.city_alt_blocks (city-voted extra blocks)")
    a = ap.parse_args(argv)
    import blocking as B

    os.makedirs(a.out_dir, exist_ok=True)
    t_all = time.time()
    s1 = _read(os.path.join(a.bench_dir, f"{a.slice}_s1.parquet"))
    pool = _read(os.path.join(a.bench_dir, f"{a.slice}_pool.parquet"))
    gt = _read(os.path.join(a.bench_dir, f"{a.slice}_links.parquet")).select(pl.col("s1").cast(pl.Utf8), pl.col("mid").cast(pl.Utf8))
    keep = ["entity_id", "business_name", "business_address", "country"]
    s1, pool = s1.select(keep), pool.select(keep)
    print(f"[{a.slice}] S1={s1.height:,} pool={pool.height:,} true links={gt.height:,}  rss={B._rss_gb():.2f}GB")

    # ---- slice ceiling -------------------------------------------------------------------------------
    in_pool = gt.join(pool.select(pl.col("entity_id").alias("mid")), on="mid", how="semi").height
    in_s1 = gt.join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi").height
    ceiling = {"links": gt.height, "target_in_pool": in_pool, "s1_in_slice": in_s1, "ceiling_recall": in_pool / max(1, gt.height)}
    print(f"[{a.slice}] slice ceiling: {in_pool:,}/{gt.height:,} link targets inside the slice pool = {ceiling['ceiling_recall']:.4f}")

    # ---- normalization (cached) --------------------------------------------------------------------------
    paths = {k: os.path.join(a.out_dir, f"{a.slice}_{k}_norm.parquet") for k in ("s1", "pool")}
    if all(os.path.exists(p) for p in paths.values()):
        s1n, pooln = pl.read_parquet(paths["s1"]), pl.read_parquet(paths["pool"])
        print(f"[{a.slice}] reused cached normalized frames")
        t_norm = 0.0
    else:
        t = time.time()
        s1n = normalize_frame(s1, a.n_jobs, a.native_dict)
        pooln = normalize_frame(pool, a.n_jobs, a.native_dict)
        t_norm = time.time() - t
        s1n.write_parquet(paths["s1"]); pooln.write_parquet(paths["pool"])
        print(f"[{a.slice}] normalized in {t_norm:.1f}s  rss={B._rss_gb():.2f}GB")

    # ---- geo-block ceiling --------------------------------------------------------------------------------
    geo = (gt.join(s1n.select(pl.col("entity_id").alias("s1"), pl.col("state_key").alias("st1")), on="s1")
             .join(pooln.select(pl.col("entity_id").alias("mid"), pl.col("state_key").alias("st2")), on="mid"))
    geo_c = {"same_block": float((geo["st1"] == geo["st2"]).mean()), "pool_unknown": float((geo["st2"] == "").mean()),
             "other_block": float(((geo["st1"] != geo["st2"]) & (geo["st2"] != "")).mean()),
             "s1_unknown": float((s1n["state_key"] == "").mean()), "pool_rows_unknown": float((pooln["state_key"] == "").mean())}
    print(f"[{a.slice}] geo blocks (state_key): true links same block {geo_c['same_block']:.4f}, pool-side unknown "
          f"{geo_c['pool_unknown']:.4f} (fallback), other block {geo_c['other_block']:.4f} (TF-IDF cannot reach; key passes can)")

    # ---- candidates ------------------------------------------------------------------------------------
    cfg = B.BlockingConfig(max_cands_per_s1=None, n_threads=a.threads, city_alt_blocks=a.city_alt)
    if a.tfidf is not None:
        cfg.tfidf_passes = tuple(x for x in a.tfidf.split(",") if x)
    if a.keys is not None:
        cfg.key_passes = tuple(x for x in a.keys.split(",") if x)
    stats = {}
    t = time.time()
    cands = B.generate_candidates(s1n, pooln, cfg, stats)
    t_block = time.time() - t
    s1_ids = s1n.select("entity_id", "country")

    order = a.passes.split(",") if a.passes else (
        [f"key_{k}" for k in cfg.key_passes] + [f"{f}_{d}" for f in cfg.tfidf_passes for d in ("fwd", "rev")])
    rows = pass_table(cands, gt, s1_ids, order)
    print(f"\n[{a.slice}] per pass (alone) and cumulative union in this order; 'only' = links no other pass finds")
    print(f"{'pass':<14}{'pairs':>11}{'alone_rec':>10}{'alone_c/S1':>11}{'cum_rec':>9}{'marginal':>9}{'cum_c/S1':>9}{'only':>8}")
    for r in rows:
        print(f"{r['pass']:<14}{r['pairs']:>11,}{r['alone_recall']:>10.4f}{r['alone_cands_per_s1']:>11.1f}{r['cum_recall']:>9.4f}"
              f"{r['marginal']:>9.4f}{r['cum_cands_per_s1']:>9.1f}{r['only_this_pass']:>8.4f}")
    timing = {k: v["seconds"] for c in stats.get("by_country", {}).values() for k, v in c.get("passes", {}).items()}
    print(f"[{a.slice}] seconds per pass: {timing}")

    full = B.candidate_recall(cands, gt, s1_ids)
    print(f"\n[{a.slice}] UNION (no cap): {_fmt(full)}")
    curve = []
    for cap in [int(x) for x in a.caps.split(",") if x]:
        r = B.candidate_recall(B.cap_candidates(cands, cap, B.reserve_expr(cfg, cands.columns), cfg.reserve_slots), gt, s1_ids)
        curve.append({"cap": cap, **{k: r[k] for k in ("pair_recall", "s1_full_recall", "cands_per_s1_mean", "cands_per_s1_p95", "oracle_f05")}})
        print(f"  cap {cap:>4}: {_fmt(r)}")

    # K curve for the TF-IDF passes (uses the rank columns of the same run; no recomputation)
    kcurve = []
    for f in cfg.tfidf_passes:
        for kf in (1, 3, 5, 10, 20):
            if kf > cfg.k_fwd.get(f, 0):
                continue
            sub = cands.filter(pl.col(f"p_{f}_rank") <= kf)
            r = B.candidate_recall(sub, gt, s1_ids)
            kcurve.append({"pass": f"{f}_fwd", "k": kf, "recall": r["pair_recall"], "cands_per_s1": r["cands_per_s1_mean"]})
        for kr in (1, 2, 3):
            if kr > cfg.k_rev.get(f, 0):
                continue
            sub = cands.filter(pl.col(f"p_{f}_rrank") <= kr)
            r = B.candidate_recall(sub, gt, s1_ids)
            kcurve.append({"pass": f"{f}_rev", "k": kr, "recall": r["pair_recall"], "cands_per_s1": r["cands_per_s1_mean"]})
    print(f"\n[{a.slice}] TF-IDF top-K curves (pass alone):")
    for r in kcurve:
        print(f"  {r['pass']:<10} k={r['k']:<3} recall {r['recall']:.4f}  cands/S1 {r['cands_per_s1']:.1f}")

    capped = B.cap_candidates(cands, a.cap, B.reserve_expr(cfg, cands.columns), cfg.reserve_slots)   # + reserve slots
    fin = B.candidate_recall(capped, gt, s1_ids)
    print(f"\n[{a.slice}] FINAL (cap {a.cap}): {_fmt(fin)}")
    peak = max(stats.get("by_country", {}).get(c, {}).get("peak_rss_gb", 0) for c in stats.get("by_country", {"": {}}))
    print(f"[{a.slice}] blocking {t_block:.1f}s, normalization {t_norm:.1f}s, total {time.time() - t_all:.1f}s, peak RSS {peak:.2f}GB")

    if not a.no_save:
        lab = capped.join(gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1, dtype=pl.Int8).alias("label")),
                          on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
        out = os.path.join(a.out_dir, f"{a.slice}_cands.parquet")
        lab.write_parquet(out)
        print(f"[{a.slice}] saved {lab.height:,} labelled candidates -> {out}")
    rep = {"slice": a.slice, "ceiling": ceiling, "geo": geo_c, "passes": rows, "union": full, "cap_curve": curve,
           "k_curve": kcurve, "final": fin, "cap": a.cap, "timing_s": {"blocking": t_block, "normalize": t_norm, **timing},
           "peak_rss_gb": peak, "stats": stats, "config": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in cfg.__dict__.items()}}
    with open(os.path.join(a.out_dir, f"{a.slice}_blocking_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    return rep


if __name__ == "__main__":
    main()
