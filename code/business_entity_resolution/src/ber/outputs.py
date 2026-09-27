#!/usr/bin/env python
"""
outputs.py -- write matching_results.tsv / candidate_pairs.tsv exactly to the
Amazon ML Challenge 2026 (Business Entity Resolution) output spec.

File spec (student_resource/README.md + utils/validate_submission.py):
    header      "source1_entity_id<TAB>matched_entity_ids"   (or "...candidate_entity_ids")
    rows        one per S1 id, in the order given (pass test_source1.tsv FILE ORDER),
                every S1 exactly once, empty list field when there is no id
    list field  unique S2-/S3- ids, sorted, joined by "," with NO spaces
    bytes       TAB separator, LF newlines, UTF-8 without BOM, no quoting

Contract function (CONTRACT.md):
    write_id_lists(s1_ids, pairs, path, list_col) -> None
        s1_ids   : sequence / pl.Series of S1 ids (test_source1 order)
        pairs    : pl.DataFrame or pl.LazyFrame with column "s1" and an id column
                   "mid" (final links) or "cand" (scored candidate pairs); other
                   columns (p, features...) are ignored
        list_col : "matched_entity_ids" | "candidate_entity_ids"
Helpers:
    read_s1_order(test_source1_path) -> list[str]     S1 ids in file order (validator semantics)
    write_submission(s1_ids, links, scored, out_dir) -> dict
        writes BOTH files; candidate_pairs.tsv = exactly the pairs the model scored;
        raises if a link is not among the scored pairs (matches must be a subset).

Usage:
    from ber.outputs import read_s1_order, write_submission
    s1_ids = read_s1_order(f"{data_dir}/test/test_source1.tsv")
    write_submission(s1_ids, links, scored, out_dir)      # links: s1, mid ; scored: s1, cand, p
    # or one file at a time:
    write_id_lists(s1_ids, links, f"{out_dir}/matching_results.tsv", "matched_entity_ids")

Memory: S1 ids are processed in chunks (chunk_s1) so a 40M-pair candidate frame is
aggregated ~250k S1 at a time; pass pl.scan_parquet(...) to avoid holding the pairs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence, Union

import polars as pl

S1_HEADER = "source1_entity_id"
LIST_COLS = ("matched_entity_ids", "candidate_entity_ids")
# strict id shape: official validator only checks the prefix, but spaces/quotes/commas in
# an id always mean an upstream bug, so the writer refuses them
_ID_RE = r"^S[23]-[^\s,\"']+$"

__all__ = ["write_id_lists", "read_s1_order", "write_submission", "S1_HEADER", "LIST_COLS"]


def _refuse_protected(path: Union[str, os.PathLike]) -> Path:
    p = Path(path).resolve()
    if any(part.lower() == "student_resource" for part in p.parts):
        raise ValueError(f"refusing to write inside student_resource/ (read-only organiser files): {p}")
    return p


def _as_order(s1_ids) -> pl.Series:
    if isinstance(s1_ids, pl.Series):
        s = s1_ids.cast(pl.Utf8).rename("s1")
    else:
        s = pl.Series("s1", list(s1_ids), dtype=pl.Utf8)
    if s.null_count():
        raise ValueError("s1_ids contains nulls")
    if s.n_unique() != s.len():
        dups = s.filter(s.is_duplicated()).unique().sort().head(5).to_list()
        raise ValueError(f"s1_ids has duplicate ids (each S1 must be written once), e.g. {dups}")
    bad = s.filter(~s.str.contains(r"^[^\t\r\n]+$") | (s.str.strip_chars() != s)).head(5).to_list()
    if bad:
        raise ValueError(f"s1_ids contains empty ids / whitespace / tabs, e.g. {bad!r}")
    return s


def _id_column(names: Sequence[str], list_col: str) -> str:
    prefer, other = ("mid", "cand") if list_col == "matched_entity_ids" else ("cand", "mid")
    for c in (prefer, other):
        if c in names:
            return c
    raise ValueError(f"pairs needs column 's1' and 'mid' or 'cand'; got {list(names)}")


def write_id_lists(s1_ids, pairs, path: str, list_col: str, *, strict: bool = True,
                   chunk_s1: int = 250_000) -> None:
    """Write one submission-format TSV (see module docstring). Atomic: the file appears
    only when fully written (tmp + os.replace), so a crash never leaves a half file.

    strict=True raises when pairs reference an S1 id that is not in s1_ids (those rows
    would be silently dropped), when an id is not a clean S2-/S3- id, or has nulls.
    """
    if list_col not in LIST_COLS:
        raise ValueError(f"list_col must be one of {LIST_COLS}, got {list_col!r}")
    target = _refuse_protected(path)
    order = _as_order(s1_ids)
    lf = pairs.lazy() if isinstance(pairs, pl.DataFrame) else pairs
    if not isinstance(lf, pl.LazyFrame):
        raise TypeError("pairs must be a polars DataFrame or LazyFrame")
    names = lf.collect_schema().names()
    if "s1" not in names:
        raise ValueError(f"pairs needs column 's1'; got {names}")
    idc = _id_column(names, list_col)
    lf = lf.select(pl.col("s1").cast(pl.Utf8), pl.col(idc).cast(pl.Utf8).alias("id"))

    if strict:
        order_df = order.to_frame()
        orphans = (lf.select("s1").unique().join(order_df.lazy(), on="s1", how="anti")
                   .sort("s1").head(5).collect())
        if orphans.height:
            raise ValueError("pairs contain S1 ids that are not in s1_ids (they would be dropped), "
                             f"e.g. {orphans['s1'].to_list()}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    n = order.len()
    try:
        with open(tmp, "wb") as f:
            f.write(f"{S1_HEADER}\t{list_col}\n".encode("utf-8"))
            for start in range(0, n, chunk_s1):
                chunk = order.slice(start, chunk_s1).to_frame()
                agg = (lf.join(chunk.lazy(), on="s1", how="semi")
                         .group_by("s1")
                         .agg(pl.col("id").null_count().alias("_nnull"),
                              pl.col("id").drop_nulls().unique().sort().alias("ids"))
                         .collect())
                if strict and agg.height:
                    if agg["_nnull"].sum():
                        raise ValueError("pairs contain null ids")
                    bad = (agg.select(pl.col("ids").explode(empty_as_null=False)).drop_nulls()
                              .filter(~pl.col("ids").str.contains(_ID_RE)).head(5))
                    if bad.height:
                        raise ValueError("ids must be clean S2-/S3- ids (no S1-, spaces, commas, quotes), "
                                         f"e.g. {bad['ids'].to_list()!r}")
                out = (chunk.join(agg.select("s1", "ids"), on="s1", how="left", maintain_order="left")
                            .select(pl.col("s1").alias(S1_HEADER),
                                    pl.col("ids").list.join(",").fill_null("").alias(list_col)))
                out.write_csv(f, include_header=False, separator="\t", quote_style="never",
                              line_terminator="\n", include_bom=False)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_s1_order(test_source1_path: str) -> list:
    """S1 ids of test_source1.tsv in FILE ORDER, with the official validator's semantics
    (first column, stripped, header skipped, blank lines ignored). Streams the file."""
    out = []
    with open(test_source1_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if line.strip():
                out.append(line.split("\t", 1)[0].strip())
    return out


def write_submission(s1_ids, links, scored, out_dir: str, *, strict: bool = True) -> dict:
    """Write output/candidate_pairs.tsv (= every pair the model scored) and
    output/matching_results.tsv (= final links), after asserting links ⊆ scored pairs.

    links : (s1, mid) final links.  scored : (s1, cand[, p, ...]) scored candidate pairs.
    Returns {"matching": path, "candidate": path, "n_links": int, "n_pairs": int}.
    """
    out = _refuse_protected(out_dir)
    links_l = links.lazy() if isinstance(links, pl.DataFrame) else links
    scored_l = scored.lazy() if isinstance(scored, pl.DataFrame) else scored
    lk = links_l.select(pl.col("s1").cast(pl.Utf8), pl.col(_id_column(links_l.collect_schema().names(),
                                                                        "matched_entity_ids")).cast(pl.Utf8).alias("id"))
    sc = scored_l.select(pl.col("s1").cast(pl.Utf8), pl.col(_id_column(scored_l.collect_schema().names(),
                                                                        "candidate_entity_ids")).cast(pl.Utf8).alias("id"))
    missing = lk.join(sc, on=["s1", "id"], how="anti").head(5).collect()
    if missing.height:
        raise ValueError("final links not among the scored candidate pairs (matches must be a subset "
                         f"of candidates), e.g. {missing.rows()}")
    mpath = str(out / "matching_results.tsv")
    cpath = str(out / "candidate_pairs.tsv")
    write_id_lists(s1_ids, sc.rename({"id": "cand"}), cpath, "candidate_entity_ids", strict=strict)
    write_id_lists(s1_ids, lk.rename({"id": "mid"}), mpath, "matched_entity_ids", strict=strict)
    n_links = lk.unique().select(pl.len()).collect().item()
    n_pairs = sc.unique().select(pl.len()).collect().item()
    return {"matching": mpath, "candidate": cpath, "n_links": int(n_links), "n_pairs": int(n_pairs)}


if __name__ == "__main__":
    import sys
    import tempfile

    # tiny demo: python outputs.py  -> prints the two files written to a temp dir
    s1 = ["S1-3", "S1-1", "S1-2"]
    scored = pl.DataFrame({"s1": ["S1-3", "S1-3", "S1-1", "S1-1"],
                           "cand": ["S3-9", "S2-10", "S2-5", "S2-5"], "p": [0.9, 0.8, 0.2, 0.2]})
    links = scored.filter(pl.col("p") > 0.5).select("s1", pl.col("cand").alias("mid"))
    d = tempfile.mkdtemp()
    info = write_submission(s1, links, scored, d)
    for k in ("candidate", "matching"):
        sys.stdout.buffer.write(open(info[k], "rb").read())
