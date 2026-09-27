# -*- coding: utf-8 -*-
"""make_dev_slice.py -- build small-but-realistic development slices from the parquet cache.

Purpose
-------
Every pipeline change is first measured on a slice, then at full scale. Two slice kinds:

(a) geo slice  -- all train S1 whose parsed state (normalize.address_parts(...)["state_canonical"]) equals X,
    plus ALL same-country S2/S3 records whose parsed state is X or unknown (distractors included, nothing
    force-added), plus the GT links of those S1. Pool density is realistic, so candidate counts, precision and
    runtime transfer to full scale. Some true targets carry another/garbled state and fall outside the pool:
    the slice reports that loss as `recall_ceiling` (= share of the slice's GT links whose target is in the
    slice pool) -- this is also the recall cost of hard state sharding at full scale.
(b) random slice -- a deterministic hash sample of N S1 per country, with the FULL same-country pool. Honest
    recall AND precision (nothing is missing from the pool, ceiling = 1.0), but expensive: the pool is millions
    of rows (train US 6.19M, India 4.13M), i.e. blocking still runs against the whole country.

Outputs (under --out-dir):
    s1.parquet, pool.parquet (+ source, slice_state), links.parquet (s1, mid, in_pool), meta.json
    <split>/<split>_source{1,2,3}.tsv (+ train_ground_truth.tsv)  -- organiser layout, so the full pipeline runs
                                                                    unchanged with --data-dir <out-dir>
Everything streams via polars scan_parquet over <work-dir>/cache (build it with io.to_parquet_cache or
pass --data-dir). State parsing runs on distinct addresses in a spawn-safe process pool.

Contract functions
------------------
geo_slice(work_dir, split, country, state, out_dir, n_jobs=4, fmt="both") -> dict (meta)
random_slice(work_dir, split, n_s1, out_dir, countries=None, seed=0, fmt="both") -> dict (meta)
state_counts(work_dir, split, country, n_jobs=4) -> pl.DataFrame  (state, n_s1) to choose X

Usage
-----
    python make_dev_slice.py states --work-dir work --split train --country India
    python make_dev_slice.py geo    --work-dir work --split train --country India --state Karnataka --out-dir work/dev/IN_KA
    python make_dev_slice.py geo    --work-dir work --split test  --country France --state Hauts-de-France --out-dir work/dev/FR_HDF
    python make_dev_slice.py random --work-dir work --split train --countries India --n-s1 5000 --out-dir work/dev/IN_rand5k
    (add --data-dir student_resource/dataset to build/refresh the cache first)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_normalize(explicit: str | None = None) -> Path:
    for cand in ([Path(explicit)] if explicit else []) + [HERE / "normalize.py",
                                                         HERE.parent / "er-text-normalization" / "normalize.py"]:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError("normalize.py not found next to this file or in ../er-text-normalization/; pass --normalize")


def _deps(normalize_path: str | None = None):
    """(io module, normalize module). Package mode (`ber` importable) first, else sibling / skill-folder files.
    Our io module is loaded under the name `ber_io` because `import io` returns the standard library."""
    try:
        if normalize_path:
            raise ImportError
        from ber import io as bio, normalize as nz   # type: ignore
        return bio, nz
    except ImportError:
        bio = sys.modules.get("ber_io") or _load("ber_io", HERE / "io.py")
        npath = _find_normalize(normalize_path)
        nz = sys.modules.get("ber_normalize") or _load("ber_normalize", npath)
        return bio, nz


# ------------------------------------------------------------------------------------------------ state parsing
_NZ = None


def _init_worker(normalize_path: str) -> None:
    global _NZ
    _NZ = _load("ber_normalize", Path(normalize_path))


def _states_chunk(args) -> list[str]:
    country, addrs = args
    ap = _NZ.address_parts
    out = []
    for a in addrs:
        st = ap(a, country)["state_canonical"] if a else None
        out.append(st or "")
    return out


class StateParser:
    """address -> canonical state ("" = unknown), via normalize.address_parts; parallel over distinct values."""

    def __init__(self, normalize_path: Path, n_jobs: int = 4, chunk: int = 4000):
        self.path, self.n_jobs, self.chunk = str(normalize_path), max(1, n_jobs), chunk
        _init_worker(self.path)            # main process needs it too (canon, n_jobs=1)
        self.pool = None
        if self.n_jobs > 1:
            self.pool = mp.get_context("spawn").Pool(self.n_jobs, initializer=_init_worker, initargs=(self.path,))

    def canon(self, state: str, country: str) -> str:
        """User spelling ('KA', 'Karnataka', 'texas') -> the canonical key address_parts emits."""
        return _NZ.address_parts(state, country)["state_canonical"] or ""

    def parse(self, addresses: pl.Series, country: str) -> pl.Series:
        """Same length/order as `addresses`; parses each distinct address once."""
        uniq = addresses.fill_null("").unique(maintain_order=True)
        vals = uniq.to_list()
        tasks = [(country, vals[i:i + self.chunk]) for i in range(0, len(vals), self.chunk)]
        if self.pool is not None:
            res = [s for part in self.pool.imap(_states_chunk, tasks) for s in part]
        else:
            res = [s for t in tasks for s in _states_chunk(t)]
        lut = pl.DataFrame({"a": uniq, "st": pl.Series(res, dtype=pl.Utf8)})
        return (pl.DataFrame({"a": addresses.fill_null("")})
                  .join(lut, on="a", how="left", maintain_order="left")["st"].fill_null(""))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def _batches(lf: pl.LazyFrame, chunk_rows: int):
    yield from lf.collect_batches(chunk_size=chunk_rows, lazy=True)


def _canon_frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    # DuckDB-made or older caches may hold nulls for empty fields; the contract says ""
    return lf.with_columns([pl.col(c).cast(pl.Utf8).fill_null("") for c in
                            ("entity_id", "business_name", "business_address", "country")])


# ------------------------------------------------------------------------------------------------ writers
def _write_outputs(out_dir: Path, split: str, s1: pl.DataFrame, pool: pl.DataFrame, links: pl.DataFrame | None,
                   meta: dict, fmt: str, gt_in_pool_only: bool, bio) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("parquet", "both"):
        s1.write_parquet(out_dir / "s1.parquet", compression="zstd")
        pool.write_parquet(out_dir / "pool.parquet", compression="zstd")
        if links is not None:
            links.write_parquet(out_dir / "links.parquet", compression="zstd")
    if fmt in ("tsv", "both"):
        d = out_dir / split
        bio.write_source_tsv(s1, d / f"{split}_source1.tsv")
        for n in ("2", "3"):
            bio.write_source_tsv(pool.filter(pl.col("source") == f"S{n}"), d / f"{split}_source{n}.tsv")
        if links is not None:
            gl = links.filter(pl.col("in_pool")) if gt_in_pool_only else links
            bio.write_ground_truth_tsv(gl.select("s1", "mid"), s1["entity_id"], d / "train_ground_truth.tsv")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def _link_stats(links: pl.DataFrame, pool: pl.DataFrame, s1: pl.DataFrame) -> dict:
    n_links = links.height
    n_in = int(links["in_pool"].sum()) if n_links else 0
    linked = links.filter(pl.col("in_pool"))["mid"]
    return {"links": n_links, "links_in_pool": n_in,
            "recall_ceiling": round(n_in / n_links, 5) if n_links else None,
            "s1_with_links": links["s1"].n_unique(),
            "singleton_rate": round(1 - links["s1"].n_unique() / max(s1.height, 1), 5),
            "pool_linked_to_slice_s1": int(pool["entity_id"].is_in(linked.implode()).sum()),
            "pool_distractor_share": round(1 - linked.n_unique() / max(pool.height, 1), 4)}


# ------------------------------------------------------------------------------------------------ slices
def state_counts(work_dir, split: str, country: str, n_jobs: int = 4, chunk_rows: int = 200_000,
                 normalize_path: str | None = None) -> pl.DataFrame:
    """Parsed-state histogram of one country's S1 -> (state, n_s1), "" = unknown. Use it to pick X."""
    bio, _ = _deps(normalize_path)
    lz = bio.scan_split(work_dir, split)
    sp = StateParser(_find_normalize(normalize_path), n_jobs)
    try:
        parts = []
        for b in _batches(_canon_frame(lz["s1"]).filter(pl.col("country") == country).select("business_address"),
                          chunk_rows):
            parts.append(sp.parse(b["business_address"], country).value_counts(name="n_s1"))
    finally:
        sp.close()
    if not parts:
        return pl.DataFrame({"state": [], "n_s1": []}, schema={"state": pl.Utf8, "n_s1": pl.UInt32})
    return (pl.concat(parts).rename({"st": "state"}).group_by("state").agg(pl.col("n_s1").sum())
              .sort("n_s1", descending=True))


def geo_slice(work_dir, split: str, country: str, state: str, out_dir, *, n_jobs: int = 4, fmt: str = "both",
              chunk_rows: int = 200_000, gt_in_pool_only: bool = False, normalize_path: str | None = None,
              log=print) -> dict:
    t0 = time.time()
    bio, _ = _deps(normalize_path)
    lz = bio.scan_split(work_dir, split)
    sp = StateParser(_find_normalize(normalize_path), n_jobs)
    try:
        target = sp.canon(state, country)
        if not target:
            raise ValueError(f"state {state!r} is not recognised for country {country!r}; run the `states` command")
        s1_parts, n_s1_country, n_s1_unknown = [], 0, 0
        for b in _batches(_canon_frame(lz["s1"]).filter(pl.col("country") == country), chunk_rows):
            st = sp.parse(b["business_address"], country)
            n_s1_country += b.height
            n_s1_unknown += int((st == "").sum())
            s1_parts.append(b.filter(st == target))
        s1 = pl.concat(s1_parts) if s1_parts else _canon_frame(lz["s1"]).head(0).collect()
        log(f"[slice] S1 {country}/{target}: {s1.height:,} of {n_s1_country:,} ({time.time()-t0:.0f}s)")
        pool_parts, pool_unknown, other_states = [], 0, {}
        for key in ("s2", "s3"):
            for b in _batches(_canon_frame(lz[key]).filter(pl.col("country") == country), chunk_rows):
                st = sp.parse(b["business_address"], country)
                keep = (st == target) | (st == "")
                pool_unknown += int((st == "").sum())
                pool_parts.append(b.filter(keep).with_columns(slice_state=st.filter(keep)))
                # remember the state of every dropped record, to explain lost links below
                other_states[key] = other_states.get(key, []) + [
                    b.select("entity_id").with_columns(st=st).filter(~keep)]
            log(f"[slice] pool {key} done ({time.time()-t0:.0f}s)")
    finally:
        sp.close()
    pool = bio.add_source_column(pl.concat(pool_parts)) if pool_parts else None
    meta = {"kind": "geo", "split": split, "country": country, "state_arg": state, "state_canonical": target,
            "s1": s1.height, "s1_country_total": n_s1_country, "s1_country_state_unknown": n_s1_unknown,
            "pool": pool.height, "pool_by_source": dict(pool["source"].value_counts().iter_rows()),
            "pool_state_unknown_kept": int((pool["slice_state"] == "").sum()),
            "pool_per_s1": round(pool.height / max(s1.height, 1), 3)}
    links = None
    if split == "train" and "gt" in lz:
        links = (lz["gt"].filter(pl.col("s1").is_in(s1["entity_id"].implode())).collect()
                 .with_columns(in_pool=pl.col("mid").is_in(pool["entity_id"].implode())))
        meta.update(_link_stats(links, pool, s1))
        lost = links.filter(~pl.col("in_pool")).select("mid")
        dropped = pl.concat([f for fs in other_states.values() for f in fs]) if other_states else None
        if dropped is not None and lost.height:
            meta["lost_links_by_target_state"] = dict(
                lost.join(dropped, left_on="mid", right_on="entity_id", how="left")
                    .with_columns(pl.col("st").fill_null("<not same-country>"))
                    .group_by("st").len().sort("len", descending=True).head(15).iter_rows())
    meta["seconds"] = round(time.time() - t0, 1)
    _write_outputs(Path(out_dir), split, s1, pool, links, meta, fmt, gt_in_pool_only, bio)
    log(f"[slice] {json.dumps(meta, ensure_ascii=False)}")
    return meta


def random_slice(work_dir, split: str, n_s1: int, out_dir, *, countries=None, seed: int = 0, fmt: str = "both",
                 normalize_path: str | None = None, log=print) -> dict:
    """n_s1 S1 per country (deterministic stable_hash64 order) + the FULL same-country S2/S3 pool."""
    t0 = time.time()
    bio, _ = _deps(normalize_path)
    lz = bio.scan_split(work_dir, split)
    ids = _canon_frame(lz["s1"]).select("entity_id", "country").collect()
    if countries is None:
        countries = sorted(ids["country"].unique().to_list())          # open set: whatever the split holds
    ids = ids.filter(pl.col("country").is_in(countries))
    ids = ids.with_columns(h=pl.Series(bio.stable_hash64(ids["entity_id"], seed), dtype=pl.UInt64))
    pick = ids.sort("h").group_by("country", maintain_order=True).head(n_s1)["entity_id"]
    del ids
    s1 = _canon_frame(lz["s1"]).filter(pl.col("entity_id").is_in(pick.implode())).collect()   # file order kept
    pool = bio.add_source_column(pl.concat([
        _canon_frame(lz[k]).filter(pl.col("country").is_in(countries)).collect() for k in ("s2", "s3")]))
    pool = pool.with_columns(slice_state=pl.lit(""))
    meta = {"kind": "random", "split": split, "countries": countries, "n_s1_per_country": n_s1, "seed": seed,
            "s1": s1.height, "s1_by_country": dict(s1["country"].value_counts().iter_rows()),
            "pool": pool.height, "pool_by_source": dict(pool["source"].value_counts().iter_rows()),
            "pool_per_s1": round(pool.height / max(s1.height, 1), 1)}
    links = None
    if split == "train" and "gt" in lz:
        links = (lz["gt"].filter(pl.col("s1").is_in(s1["entity_id"].implode())).collect()
                 .with_columns(in_pool=pl.col("mid").is_in(pool["entity_id"].implode())))
        meta.update(_link_stats(links, pool, s1))
    meta["seconds"] = round(time.time() - t0, 1)
    _write_outputs(Path(out_dir), split, s1, pool, links, meta, fmt, False, bio)
    log(f"[slice] {json.dumps(meta, ensure_ascii=False)}")
    return meta


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build geo / random dev slices from the parquet cache.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work-dir", required=True, help="holds cache/ (io.to_parquet_cache)")
    common.add_argument("--data-dir", help="organiser dataset dir: build/refresh the cache first")
    common.add_argument("--split", default="train", choices=["train", "test"])
    common.add_argument("--normalize", help="path to normalize.py (auto-detected by default)")
    s = sub.add_parser("states", parents=[common], help="S1 count per parsed state of one country")
    s.add_argument("--country", required=True)
    s.add_argument("--n-jobs", type=int, default=4)
    g = sub.add_parser("geo", parents=[common], help="all S1 of state X + same-country pool with state X/unknown")
    g.add_argument("--country", required=True)
    g.add_argument("--state", required=True, help="any spelling address_parts understands: KA, Karnataka, TX, Texas")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--n-jobs", type=int, default=4)
    g.add_argument("--format", default="both", choices=["parquet", "tsv", "both"])
    g.add_argument("--gt-in-pool-only", action="store_true", help="TSV ground truth keeps only reachable links")
    r = sub.add_parser("random", parents=[common], help="N hash-sampled S1 per country + FULL country pool")
    r.add_argument("--n-s1", type=int, required=True, help="S1 per country")
    r.add_argument("--countries", help="comma list; default = every country present in the split")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--format", default="both", choices=["parquet", "tsv", "both"])
    a = ap.parse_args(argv)
    bio, _ = _deps(a.normalize)
    if a.data_dir:
        bio.to_parquet_cache(a.data_dir, a.work_dir, splits=(a.split,))
    if a.cmd == "states":
        with pl.Config(tbl_rows=80):
            print(state_counts(a.work_dir, a.split, a.country, a.n_jobs, normalize_path=a.normalize))
    elif a.cmd == "geo":
        geo_slice(a.work_dir, a.split, a.country, a.state, a.out_dir, n_jobs=a.n_jobs, fmt=a.format,
                  gt_in_pool_only=a.gt_in_pool_only, normalize_path=a.normalize)
    else:
        geo = a.countries.split(",") if a.countries else None
        random_slice(a.work_dir, a.split, a.n_s1, a.out_dir, countries=geo, seed=a.seed, fmt=a.format,
                     normalize_path=a.normalize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
