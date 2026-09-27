# -*- coding: utf-8 -*-
"""io.py -- safe loading of the Business Entity Resolution TSVs (Amazon ML Challenge 2026, team Master Bolt).

Purpose
-------
Read the organiser TSVs without corrupting a single field, cache them once as zstd parquet, explode the ground
truth to long form, and give every S1 a deterministic validation fold. Polars-first and memory-frugal: the
full-size path is `to_parquet_cache` (streaming sink) + `scan_split` (lazy frames), never a full in-memory load.

Format facts this module is built on (measured on the first 100k lines of all 7 files, 2026-09-25):
  * TAB separated, header row, UTF-8, LF, no BOM; exactly 3 tabs per source line (1 per GT line).
  * Literal "NULL" / "None" / "NA" occur as DATA (address noise tokens, even a whole business_name "NA");
    they must stay strings. Empty fields must become "" (never null / NaN).
  * The writer used CSV QUOTE_MINIMAL: the rare field that contains a double quote is wrapped in quotes with
    inner quotes doubled, e.g. raw  Q Q Q ehpad Club SAS Q  == value  Q ehpad Club SAS  (Q = one double-quote
    char; 1-6 fields per 100k rows, mostly France / junk-leading-quote noise).
    We parse with quoting DISABLED (a stray quote can never swallow tabs/newlines, so rows == lines - 1 always)
    and then decode only fields that are exactly a well-formed quoted token (`unquote=True`).
  * Apostrophes (' and U+2019) are common (~2% of fields) and are plain data.

Contract functions (CONTRACT.md)
--------------------------------
read_source(path)                     -> pl.DataFrame  entity_id, business_name, business_address, country (all Utf8, "" for missing)
read_ground_truth(path)               -> pl.DataFrame  long form s1, mid (singletons have no rows)
load_split(data_dir, split)           -> dict          {"s1","s2","s3"} (+ "gt" for train) eager frames (small data / big RAM only)
Extra public helpers
--------------------
scan_split(work_dir, split)           -> dict          same keys, LAZY frames over the parquet cache (laptop full-size path)
to_parquet_cache(data_dir, work_dir)  -> dict          convert each TSV once to <work_dir>/cache/*.parquet (skips if up to date)
add_source_column(df)                 -> frame         + `source` in {"S1","S2","S3"} from the id prefix
s1_ids_in_file_order(path)            -> pl.Series     S1 ids in file order (outputs must follow test_source1 order)
make_folds(s1_ids, n_folds, seed)     -> pl.DataFrame  s1, fold (Int8); deterministic hash (same as metric.fold_of)
gt_k_per_s1(gt, s1_ids)               -> pl.DataFrame  s1, k (0 for singletons)
count_data_lines(path)                -> int           naive newline count minus header (the `wc -l` - 1 check)
verify_source(df, prefix) / verify_ground_truth(gt, ...) -> dict  checklist numbers

Usage
-----
    from ber import io as bio            # NEVER `import io` -- that is the standard library module
    bio.to_parquet_cache("student_resource/dataset", "work")          # once; ~minutes; streaming, low RAM
    lz = bio.scan_split("work", "train")                               # lazy frames
    s1_us = lz["s1"].filter(pl.col("country") == "US").collect()
    gt = lz["gt"].collect()                                            # s1, mid
    folds = bio.make_folds(s1_us["entity_id"], n_folds=5, seed=0)

CLI
---
    python io.py verify --data-dir student_resource/dataset --head 100000   # row counts, quotes, countries
    python io.py cache  --data-dir student_resource/dataset --work-dir work # build / refresh parquet cache
    python io.py folds  --work-dir work --n-folds 5 --seed 0                # -> work/cache/train_folds.parquet
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import polars as pl

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GT_COLUMNS = ("source1_entity_id", "matched_entity_ids")
SPLITS = ("train", "test")
SOURCE_KEYS = ("s1", "s2", "s3")
CACHE_SCHEMA_VERSION = 1          # bump when the cached representation changes -> forces a rebuild

# A field that is exactly one CSV-quoted token: opening quote, (non-quote | doubled quote)*, closing quote.
_QUOTED_RE = r'^"(?:[^"]|"")*"$'
_ID_RE = {"s1": r"^S1-\d+$", "s2": r"^S2-\d+$", "s3": r"^S3-\d+$"}

# polars renamed missing_utf8_is_empty_string -> empty_string_is_null (inverted) in 1.43
_EMPTY_KW = ({"empty_string_is_null": False}
             if "empty_string_is_null" in inspect.signature(pl.scan_csv).parameters
             else {"missing_utf8_is_empty_string": True})
# polars 1.44 warns that explode's empty_as_null default flips in 2.0; pass it explicitly when supported
_EXPLODE_KW = {"empty_as_null": True} if "empty_as_null" in inspect.signature(pl.LazyFrame.explode).parameters else {}


def _csv_kwargs(encoding: str = "utf8") -> dict:
    return dict(separator="\t", quote_char=None, has_header=True, infer_schema=False, null_values=None,
                encoding=encoding, truncate_ragged_lines=False, raise_if_empty=True, **_EMPTY_KW)


# --------------------------------------------------------------------------------------------------------------
# file layout
# --------------------------------------------------------------------------------------------------------------
def source_files(data_dir: str | os.PathLike, split: str) -> dict[str, Path]:
    """Paths of the organiser files of one split: {"s1","s2","s3"} (+ "gt" for train when present)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    d = Path(data_dir) / split
    out = {k: d / f"{split}_source{k[1]}.tsv" for k in SOURCE_KEYS}
    gt = d / "train_ground_truth.tsv"
    if split == "train" and gt.exists():
        out["gt"] = gt
    return out


def cache_paths(work_dir: str | os.PathLike, split: str) -> dict[str, Path]:
    """Parquet cache locations under <work_dir>/cache/ (same keys as source_files; gt = LONG form)."""
    c = Path(work_dir) / "cache"
    out = {k: c / f"{split}_source{k[1]}.parquet" for k in SOURCE_KEYS}
    if split == "train":
        out["gt"] = c / "train_gt_long.parquet"
    return out


# --------------------------------------------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------------------------------------------
def decode_csv_quoted(expr: pl.Expr) -> pl.Expr:
    """Undo QUOTE_MINIMAL encoding on fields that are exactly one quoted token; leave everything else as is."""
    inner = expr.str.slice(1, expr.str.len_chars() - 2).str.replace_all('""', '"', literal=True)
    return pl.when(expr.str.starts_with('"') & expr.str.contains(_QUOTED_RE)).then(inner).otherwise(expr)


def _check_header(cols: list[str], expected: tuple[str, ...], where: str) -> dict[str, str]:
    """Map raw header names (BOM/space tolerant) to canonical names; raise a helpful error on a wrong file."""
    clean = [c.replace("﻿", "").strip() for c in cols]
    if len(clean) == 1 and ("," in clean[0] or ";" in clean[0]):
        raise ValueError(f"{where}: header {cols[0]!r} has no TAB -- file is not tab separated")
    missing = [c for c in expected if c not in clean]
    if missing:
        raise ValueError(f"{where}: header {clean} lacks {missing} (expected {list(expected)})")
    return {raw: cl for raw, cl in zip(cols, clean)}


def _source_lazy(lf: pl.LazyFrame, where: str, unquote: bool) -> pl.LazyFrame:
    ren = _check_header(lf.collect_schema().names(), SOURCE_COLUMNS, where)
    lf = lf.rename({k: v for k, v in ren.items() if k != v}).select(SOURCE_COLUMNS)
    exprs = []
    for c in SOURCE_COLUMNS:
        e = pl.col(c).cast(pl.Utf8).fill_null("")
        if c == SOURCE_COLUMNS[-1]:
            e = e.str.strip_chars_end("\r")              # stray CR from mixed line endings
        if unquote and c != "entity_id":
            e = decode_csv_quoted(e)
        exprs.append(e.alias(c))
    return lf.with_columns(exprs)


def _is_parquet(path) -> bool:
    return isinstance(path, (str, os.PathLike)) and str(path).lower().endswith(".parquet")


def scan_source(path, *, unquote: bool = True, n_rows: int | None = None, encoding: str = "utf8") -> pl.LazyFrame:
    """Lazy version of read_source (TSV, parquet cache file, or raw bytes of a TSV head)."""
    if _is_parquet(path):
        lf = pl.scan_parquet(path)
        if n_rows is not None:
            lf = lf.head(n_rows)
        return _source_lazy(lf, str(path), unquote=False)   # the cache is already decoded
    src = path if isinstance(path, (bytes, bytearray)) else str(path)
    lf = pl.scan_csv(src, n_rows=n_rows, **_csv_kwargs(encoding))
    return _source_lazy(lf, "<bytes>" if isinstance(path, (bytes, bytearray)) else str(path), unquote)


def read_source(path, *, columns: Iterable[str] | None = None, n_rows: int | None = None,
                unquote: bool = True, encoding: str = "utf8") -> pl.DataFrame:
    """Safe source reader: TAB, quoting disabled, every column Utf8, missing -> "", literal NULL/None kept.

    `path` may be a .tsv, a .parquet cache file, or bytes (e.g. the first 100k lines of a big TSV).
    `columns` projects early (e.g. ["entity_id", "country"]) to save memory.
    """
    lf = scan_source(path, unquote=unquote, n_rows=n_rows, encoding=encoding)
    if columns is not None:
        lf = lf.select(list(columns))
    return lf.collect()


def _gt_long_lazy(lf: pl.LazyFrame, where: str) -> pl.LazyFrame:
    names = lf.collect_schema().names()
    if {"s1", "mid"} <= set(names):                        # already long (cache file)
        return lf.select(pl.col("s1").cast(pl.Utf8), pl.col("mid").cast(pl.Utf8))
    ren = _check_header(names, GT_COLUMNS, where)
    lf = lf.rename({k: v for k, v in ren.items() if k != v})
    return (lf.select(pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars().alias("s1"),
                      pl.col("matched_entity_ids").cast(pl.Utf8).fill_null("").str.strip_chars_end("\r")
                      .str.split(",").alias("mid"))
              .explode("mid", **_EXPLODE_KW)
              .with_columns(pl.col("mid").str.strip_chars())
              .filter(pl.col("mid").is_not_null() & (pl.col("mid") != "")))


def scan_ground_truth(path, encoding: str = "utf8") -> pl.LazyFrame:
    if _is_parquet(path):
        return _gt_long_lazy(pl.scan_parquet(path), str(path))
    src = path if isinstance(path, (bytes, bytearray)) else str(path)
    return _gt_long_lazy(pl.scan_csv(src, **_csv_kwargs(encoding)), str(path)[:80])


def read_ground_truth(path, encoding: str = "utf8") -> pl.DataFrame:
    """Ground truth in LONG form: one row per link (s1, mid), file order kept; singletons have no rows.
    Accepts the organiser TSV (source1_entity_id, matched_entity_ids) or a cached long parquet (s1, mid)."""
    return scan_ground_truth(path, encoding).collect()


def gt_k_per_s1(gt: pl.DataFrame, s1_ids) -> pl.DataFrame:
    """s1, k (number of true links, 0 for singletons) for every id in s1_ids (order kept)."""
    ids = pl.DataFrame({"s1": pl.Series(s1_ids, dtype=pl.Utf8)})
    k = gt.group_by("s1").agg(pl.len().cast(pl.Int32).alias("k"))
    return ids.join(k, on="s1", how="left", maintain_order="left").with_columns(pl.col("k").fill_null(0))


def load_split(data_dir: str | os.PathLike, split: str, *, work_dir: str | os.PathLike | None = None,
               columns: Iterable[str] | None = None) -> dict[str, pl.DataFrame]:
    """Eager {"s1","s2","s3"} (+ "gt" long, train only). With work_dir the parquet cache is (re)built and used.

    Memory: the full train split is ~3.3 GB in RAM (see SKILL.md budget) -- on the laptop use scan_split and
    filter by country/state first; load_split is for mini data, dev slices, or a big AWS box.
    """
    if work_dir is not None:
        to_parquet_cache(data_dir, work_dir, splits=(split,))
        paths = cache_paths(work_dir, split)
    else:
        paths = source_files(data_dir, split)
    out = {}
    for k in SOURCE_KEYS:
        if not Path(paths[k]).exists():
            raise FileNotFoundError(paths[k])
        out[k] = read_source(paths[k], columns=columns)
    if "gt" in paths and Path(paths["gt"]).exists():
        out["gt"] = read_ground_truth(paths["gt"])
    return out


def scan_split(work_dir: str | os.PathLike, split: str) -> dict[str, pl.LazyFrame]:
    """Lazy frames over the parquet cache (build it first with to_parquet_cache). Filter, project, then collect."""
    paths = cache_paths(work_dir, split)
    missing = [str(p) for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"parquet cache missing {missing}; run to_parquet_cache(data_dir, work_dir) first")
    out = {k: pl.scan_parquet(paths[k]) for k in SOURCE_KEYS}
    if "gt" in paths:
        out["gt"] = pl.scan_parquet(paths["gt"])
    return out


# --------------------------------------------------------------------------------------------------------------
# ids, sources, folds
# --------------------------------------------------------------------------------------------------------------
def add_source_column(df):
    """Add `source` ("S1"/"S2"/"S3") from the entity_id prefix. Validates eagerly for DataFrames."""
    lf_or_df = df.with_columns(pl.col("entity_id").str.slice(0, 2).alias("source"))
    if isinstance(df, pl.DataFrame):
        bad = lf_or_df.filter(~pl.col("entity_id").str.contains(r"^S[123]-"))
        if bad.height:
            raise ValueError(f"{bad.height} entity_id without S1-/S2-/S3- prefix, e.g. {bad['entity_id'].head(3).to_list()}")
    return lf_or_df


def s1_ids_in_file_order(path) -> pl.Series:
    """S1 ids exactly in file order (TSV or parquet cache). Raises on empty / duplicate / non-S1 ids.
    matching_results.tsv and candidate_pairs.tsv are written by iterating over this series."""
    lf = scan_source(path, unquote=False).select(pl.col("entity_id").str.strip_chars().alias("s1"))
    s = lf.collect().to_series()
    bad = s.filter(~s.str.contains(r"^S1-\S+$"))
    if bad.len():
        raise ValueError(f"{path}: {bad.len()} ids are not S1 ids, e.g. {bad.head(3).to_list()}")
    dup = s.len() - s.n_unique()
    if dup:
        raise ValueError(f"{path}: {dup} duplicate S1 ids (the scorer rejects duplicate rows)")
    return s


def stable_hash64(ids, seed: int = 0):
    """Version-stable uint64 hash per id (numpy array): splitmix64 of the numeric id suffix, crc32 fallback for
    ids without digits. Same algorithm as er-f05-decisions metric.fold_of. polars Expr.hash is NOT used because
    its values may change between polars releases (folds/samples would silently change)."""
    import zlib
    import numpy as np
    s = ids.cast(pl.Utf8) if isinstance(ids, pl.Series) else pl.Series(list(ids), dtype=pl.Utf8)
    num = s.str.extract(r"(\d+)\s*$", 1).cast(pl.UInt64, strict=False)
    x = num.fill_null(0).to_numpy().astype(np.uint64)
    miss = num.is_null().to_numpy()
    if miss.any():
        x[miss] = np.array([zlib.crc32(v.encode()) for v in s.filter(pl.Series(miss)).to_list()], dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15) * np.uint64(seed + 1)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def make_folds(s1_ids, n_folds: int = 5, seed: int = 0) -> pl.DataFrame:
    """Deterministic fold per S1 -> DataFrame s1 (Utf8), fold (Int8), aligned with the input order.
    GroupKFold by S1: every candidate pair inherits the fold of its s1 (join on s1). Pools are never split."""
    import numpy as np
    if n_folds < 2 or n_folds > 127:
        raise ValueError("n_folds must be in [2, 127]")
    s = pl.Series("s1", s1_ids, dtype=pl.Utf8) if not isinstance(s1_ids, pl.Series) else s1_ids.cast(pl.Utf8).alias("s1")
    fold = (stable_hash64(s, seed) % np.uint64(n_folds)).astype(np.int8)
    return pl.DataFrame({"s1": s, "fold": pl.Series("fold", fold, dtype=pl.Int8)})


# --------------------------------------------------------------------------------------------------------------
# writers (mirror the organiser format: TAB, LF, no BOM, CSV QUOTE_MINIMAL only for fields containing a quote)
# --------------------------------------------------------------------------------------------------------------
def write_source_tsv(df: pl.DataFrame, path) -> None:
    """Write a source frame in the organiser format so read_source() gives back exactly the same values."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.select(SOURCE_COLUMNS).write_csv(path, separator="\t", quote_style="necessary", line_terminator="\n",
                                        include_header=True)


def write_ground_truth_tsv(gt: pl.DataFrame, s1_ids, path) -> None:
    """Write long-form links (s1, mid) back to the organiser wide format: one row per id in s1_ids (file order),
    matched_entity_ids comma-joined (empty for singletons). mid order inside a row follows gt order."""
    ids = pl.DataFrame({"source1_entity_id": pl.Series(s1_ids, dtype=pl.Utf8)})
    agg = gt.group_by("s1", maintain_order=True).agg(pl.col("mid").str.join(",").alias("matched_entity_ids"))
    out = (ids.join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left")
              .with_columns(pl.col("matched_entity_ids").fill_null("")))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="necessary", line_terminator="\n", include_header=True)


# --------------------------------------------------------------------------------------------------------------
# verification helpers
# --------------------------------------------------------------------------------------------------------------
def count_data_lines(path, chunk: int = 1 << 24) -> int:
    """Naive record count = newline count (+1 if no trailing newline) - 1 header. Streams in 16 MB chunks."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += b.count(b"\n")
            last = b[-1:]
    if last != b"\n":
        n += 1
    return max(n - 1, 0)


def sniff(path, nbytes: int = 1 << 16) -> dict:
    """Header / BOM / CRLF / tabs-per-line of the first 64 KB."""
    with open(path, "rb") as f:
        b = f.read(nbytes)
    lines = b.split(b"\n")[:-1] or [b]
    return {"bom": b.startswith(b"\xef\xbb\xbf"), "crlf": b"\r\n" in b,
            "header": lines[0].decode("utf-8", "replace").rstrip("\r").lstrip("﻿").split("\t"),
            "tabs_per_line": sorted({ln.count(b"\t") for ln in lines})}


def verify_source(df: pl.DataFrame, key: str | None = None) -> dict:
    """Checklist numbers for one source frame (key in s1/s2/s3 enables the id-prefix check)."""
    ids = df["entity_id"]
    res = {"rows": df.height, "dup_ids": df.height - ids.n_unique(),
           "null_cells": int(sum(df.null_count().row(0))),
           "empty_name": int((df["business_name"] == "").sum()),
           "empty_address": int((df["business_address"] == "").sum()),
           "fields_with_literal_NULL_None": int(df.select(
               pl.sum_horizontal(pl.col("business_name", "business_address").str.contains(r"\b(?:NULL|null|None)\b"))
           ).to_series().sum()),
           "countries": dict(df["country"].value_counts().sort("count", descending=True).iter_rows())}
    if key in _ID_RE:
        res["bad_id_prefix"] = int((~ids.str.contains(_ID_RE[key])).sum())
    return res


def verify_ground_truth(gt: pl.DataFrame, s1_ids=None, pool_ids=None) -> dict:
    """Links, exclusivity (each mid -> at most one s1), id prefixes, coverage against S1 / pool ids."""
    res = {"links": gt.height, "s1_with_links": gt["s1"].n_unique(),
           "dup_links": gt.height - gt.unique(["s1", "mid"]).height,
           "mids_linked_to_several_s1": int(gt.group_by("mid").agg(pl.col("s1").n_unique().alias("n"))
                                            .filter(pl.col("n") > 1).height),
           "bad_mid_prefix": int((~gt["mid"].str.contains(r"^S[23]-\d+$")).sum())}
    if s1_ids is not None:
        ids = pl.Series(s1_ids, dtype=pl.Utf8)
        res["gt_s1_not_in_source1"] = int((~gt["s1"].unique().is_in(ids.implode())).sum())
        res["singleton_rate"] = round(1 - gt["s1"].unique().is_in(ids.implode()).sum() / max(ids.len(), 1), 5)
    if pool_ids is not None:
        res["mids_not_in_pool"] = int((~gt["mid"].is_in(pl.Series(pool_ids, dtype=pl.Utf8).implode())).sum())
    return res


# --------------------------------------------------------------------------------------------------------------
# parquet cache
# --------------------------------------------------------------------------------------------------------------
def _meta_path(p: Path) -> Path:
    return p.with_name(p.name + ".meta.json")


def _up_to_date(src: Path, dst: Path, unquote: bool) -> bool:
    mp = _meta_path(dst)
    if not (dst.exists() and mp.exists()):
        return False
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    st = src.stat()
    return (m.get("size") == st.st_size and m.get("mtime_ns") == st.st_mtime_ns
            and m.get("unquote") == unquote and m.get("schema_version") == CACHE_SCHEMA_VERSION)


def _parquet_rows(p: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(p).metadata.num_rows


NL, CRLF = bytes([10]), bytes([13, 10])


def iter_tsv_chunks(path, chunk_bytes: int = 32 << 20):
    """Yield (chunk_bytes_with_header, n_newlines) pieces of a big TSV, cut at line ends. Every piece starts with
    the original header line, so each parses exactly like a small TSV. Bounded memory (~chunk_bytes)."""
    with open(path, "rb") as f:
        header = f.readline()
        if not header.endswith(NL):
            header += NL
        carry = b""
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            block = carry + block
            cut = block.rfind(NL)
            if cut < 0:
                carry = block
                continue
            carry = block[cut + 1:]
            yield header + block[:cut + 1], block.count(NL, 0, cut + 1)
        if carry.strip(CRLF):                  # last line without a trailing newline
            yield header + carry + NL, 1


def _convert_tsv(src: Path, tmp: Path, key: str, unquote: bool, chunk_bytes: int, row_group_size: int) -> dict:
    """Chunked TSV -> parquet. Returns counts: parsed rows (wide rows for gt), naive lines, written rows."""
    import pyarrow.parquet as pq
    writer, parsed, naive, written, s1_with_links = None, 0, 0, 0, 0
    try:
        for chunk, n_lines in iter_tsv_chunks(src, chunk_bytes):
            naive += n_lines
            if key == "gt":
                wide = pl.read_csv(chunk, **_csv_kwargs())
                parsed += wide.height
                df = _gt_long_lazy(wide.lazy(), str(src)).collect()
                s1_with_links += df["s1"].n_unique()
            else:
                df = read_source(chunk, unquote=unquote)
                parsed += df.height
            tbl = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), tbl.schema, compression="zstd")
            writer.write_table(tbl, row_group_size=row_group_size)
            written += df.height
            del df, tbl
        if writer is None:       # header-only file: still write an empty, correctly typed parquet
            cols = ("s1", "mid") if key == "gt" else SOURCE_COLUMNS
            pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in cols}).write_parquet(tmp)
    finally:
        if writer is not None:
            writer.close()
    return {"parsed_rows": parsed, "naive_data_lines": naive, "written_rows": written,
            **({"gt_s1_with_links": s1_with_links} if key == "gt" else {})}


def to_parquet_cache(data_dir: str | os.PathLike, work_dir: str | os.PathLike, splits: Iterable[str] = SPLITS,
                     *, force: bool = False, unquote: bool = True, verify: bool = True,
                     chunk_bytes: int = 32 << 20, row_group_size: int = 250_000, log=print) -> dict:
    """Convert every organiser TSV once to zstd parquet under <work_dir>/cache/ (gt -> LONG form s1, mid).

    Bounded memory: the TSV is parsed in ~32 MB line-aligned chunks with the same parser as read_source and
    appended to one parquet file (peak RSS a few hundred MB whatever the file size; polars' own
    scan_csv().sink_parquet() peaked at 1.3 GB on a 425 MB file on this laptop). A file is skipped when its
    .meta.json sidecar matches the TSV size + mtime (+ settings). Writes go to *.tmp then os.replace, so a crash
    never leaves a half-written file that looks valid. With verify=True the parsed row count must equal the
    naive newline count (the `wc -l` - 1 gate), else RuntimeError. Returns {"train/s1": {path, status, rows,
    seconds}, ...}.
    """
    report = {}
    for split in splits:
        src_map, dst_map = source_files(data_dir, split), cache_paths(work_dir, split)
        for key, src in src_map.items():
            dst = dst_map[key]
            name = f"{split}/{key}"
            if not src.exists():
                raise FileNotFoundError(src)
            if not force and _up_to_date(src, dst, unquote):
                m = json.loads(_meta_path(dst).read_text(encoding="utf-8"))
                report[name] = {"path": str(dst), "status": "up-to-date", "rows": m.get("rows"), "seconds": 0.0}
                continue
            t0 = time.time()
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            counts = _convert_tsv(src, tmp, key, unquote, chunk_bytes, row_group_size)
            rows = _parquet_rows(tmp)
            meta = {"source": str(src), "size": src.stat().st_size, "mtime_ns": src.stat().st_mtime_ns,
                    "unquote": unquote, "schema_version": CACHE_SCHEMA_VERSION, "rows": rows, **counts,
                    "polars": pl.__version__, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            if verify and (counts["parsed_rows"] != counts["naive_data_lines"] or rows != counts["written_rows"]):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{src}: parsed rows != naive line count ({meta}) -- check separator/quoting")
            os.replace(tmp, dst)
            _meta_path(dst).write_text(json.dumps(meta, indent=1), encoding="utf-8")
            report[name] = {"path": str(dst), "status": "written", "rows": rows, "seconds": round(time.time() - t0, 1)}
            if log:
                log(f"[io] {name}: {rows:,} rows -> {dst} ({report[name]['seconds']}s)")
    return report


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------
def _head_bytes(path: Path, n_lines: int) -> bytes:
    import itertools
    with open(path, "rb") as f:
        return b"".join(itertools.islice(f, n_lines))


def verify_files(data_dir, head: int | None = 100_000) -> dict:
    """Row-count / quote / country checklist for every organiser file (head=None -> full files, slow)."""
    out = {}
    for split in SPLITS:
        for key, p in source_files(data_dir, split).items():
            if not p.exists():
                continue
            info = sniff(p)
            if head:
                raw = _head_bytes(p, head)
                lines = raw.decode("utf-8").split("\n")
                naive = raw.count(b"\n") - 1 + (0 if raw.endswith(b"\n") else 1)
                src = raw
            else:
                lines, naive, src = None, count_data_lines(p), p
            if key == "gt":
                wide = pl.read_csv(src if head else str(p), **_csv_kwargs())
                gt = read_ground_truth(src if head else p)
                info.update(rows=wide.height, naive_rows=naive, rows_ok=wide.height == naive,
                            **verify_ground_truth(gt, s1_ids=wide[wide.columns[0]]))
            else:
                df = read_source(src)
                info.update(naive_rows=naive, rows_ok=df.height == naive, **verify_source(df, key))
                if lines is not None:     # field-level round trip against a naive split (+ quote decoding)
                    body = [ln.rstrip("\r").split("\t") for ln in lines[1:] if ln]
                    quoted = sum(1 for r in body for v in r[1:] if v.startswith('"'))
                    apos = sum(1 for r in body for v in r if "'" in v or "’" in v)
                    info.update(quote_encoded_fields=quoted, fields_with_apostrophe=apos,
                                apostrophes_survive=apos == int(df.select(pl.sum_horizontal(
                                    pl.all().str.contains("['’]"))).to_series().sum()))
            out[f"{split}/{key}"] = info
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="row counts vs naive line counts, quotes, ids, countries")
    v.add_argument("--data-dir", required=True)
    v.add_argument("--head", type=int, default=100_000, help="first N lines per file (0 = full files)")
    c = sub.add_parser("cache", help="build/refresh the parquet cache")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--work-dir", required=True)
    c.add_argument("--splits", default="train,test")
    c.add_argument("--force", action="store_true")
    f = sub.add_parser("folds", help="write <work>/cache/train_folds.parquet (s1, fold)")
    f.add_argument("--work-dir", required=True)
    f.add_argument("--n-folds", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        res = verify_files(a.data_dir, a.head or None)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 0 if all(r.get("rows_ok") for r in res.values()) else 1
    if a.cmd == "cache":
        rep = to_parquet_cache(a.data_dir, a.work_dir, splits=tuple(a.splits.split(",")), force=a.force)
        print(json.dumps(rep, indent=1))
        return 0
    if a.cmd == "folds":
        s1 = s1_ids_in_file_order(cache_paths(a.work_dir, "train")["s1"])
        folds = make_folds(s1, a.n_folds, a.seed)
        dst = Path(a.work_dir) / "cache" / "train_folds.parquet"
        folds.write_parquet(dst, compression="zstd")
        print(folds.group_by("fold").len().sort("fold"))
        print(f"-> {dst}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
