#!/usr/bin/env python
"""
check_submission.py -- streaming, low-memory re-implementation of every rule of the
official student_resource/utils/validate_submission.py, for BOTH output files, plus:

  * strict format gates the official validator lets through: CRLF line endings, header
    not byte-exact (case/spaces), extra TAB columns, whitespace/quotes inside ids or a
    whitespace-only list field, blank lines, matches not a subset of candidates
  * matches ⊆ candidates, by lockstep merge when both files share the row order
    (O(1) memory) and by an on-disk external sorted merge otherwise
  * exclusivity diagnostic: an S2/S3 id listed under more than one S1 (never happens in
    the train ground truth, so it signals a decision-stage bug; warning only)
  * stats: rows, empty rows, ids/row (mean, max, histogram), S2 vs S3, per country,
    whether rows follow test_source1 order and ids are sorted
  * --official: also runs the organisers' validator on the scored file with the literal
    `--candidate NONE` (omitting --candidate still loads output/candidate_pairs.tsv
    relative to the cwd) and fails with exit 3 if its verdict disagrees with ours

Levels: "official" = the official validator rejects it; "strict" = our extra gate
(--lenient turns these into warnings); "warning" = never fails.
Stdlib only (Python 3.8+), so it runs in any venv; psutil (optional) adds peak-RSS stats.

Memory: a dict of the required S1 ids (~0.2 GB for 1.73M S1) plus `chunk_lines`
strings for the external sorts (ID existence, exclusivity, subset fallback).

Usage (absolute paths):
  python check_submission.py --matching <abs>/output/matching_results.tsv \
      --candidate <abs>/output/candidate_pairs.tsv \
      --test-dir <abs>/student_resource/dataset/test [--check-ids] [--official] [--json rep.json]

Library:
  from check_submission import check_submission, run_official
  rep = check_submission(matching, candidate, test_dir, check_ids=False)
  rep["ok"], rep["official_ok"], rep["issues"], rep["stats"]

Exit codes: 0 PASS, 1 FAIL, 2 bad arguments, 3 disagreement with the official validator.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

DELIM = "\t"
HEADERS = {"matching": ["source1_entity_id", "matched_entity_ids"],
           "candidate": ["source1_entity_id", "candidate_entity_ids"]}
OFFICIAL, STRICT, WARNING = "official", "strict", "warning"
EX_CAP = 50           # distinct examples kept per issue
SHOW = 5              # examples printed per issue (same as the official validator)
HIST_MAX = 10         # ids-per-row histogram buckets 0..10, then "11+"
_BAD_ID_CHARS = re.compile(r"[\s\"']")
BOM = b"\xef\xbb\xbf"


# ----------------------------------------------------------------------------- helpers
class Issues:
    """Accumulates issues keyed by (file, rule) with a count and a few distinct examples."""

    def __init__(self):
        self.items = {}

    def add(self, file, rule, level, msg, example=None, n=1):
        it = self.items.get((file, rule))
        if it is None:
            it = self.items[(file, rule)] = {"file": file, "rule": rule, "level": level,
                                             "msg": msg, "count": 0, "examples": set()}
        it["count"] += n
        if example is not None and len(it["examples"]) < EX_CAP:
            it["examples"].add(example)

    def of(self, level=None, file=None):
        return [i for i in self.items.values()
                if (level is None or i["level"] == level) and (file is None or i["file"] == file)]


def _mirror_lines(fb, st):
    """Yield text lines as the official validator sees them (open(path, encoding="utf-8"):
    universal newlines, strict UTF-8) while recording raw facts in `st` (BOM, CR, final LF)."""
    first = True
    last = b""
    for raw in fb:                                  # binary iteration splits on b"\n" only
        if first:
            st["bom"] = raw.startswith(BOM)
            first = False
        last = raw
        if b"\r" in raw:
            st["cr_lines"] += 1
            text = raw.decode("utf-8").replace("\r\n", "\n")
            parts = text.split("\r")                # a lone CR is a line break in text mode
            for i, p in enumerate(parts):
                s = p + "\n" if i < len(parts) - 1 else p
                if s:
                    yield s
        else:
            yield raw.decode("utf-8")
    st["final_newline"] = (not last) or last.endswith(b"\n")


class ExtSorter:
    """External sort of text lines: sorted chunks of `chunk` lines spilled to tmp files,
    streamed back with heapq.merge. sorted_iter() may be called more than once."""

    def __init__(self, tmpdir, tag, chunk=1_000_000):
        self.tmpdir, self.tag, self.chunk = tmpdir, tag, chunk
        self.buf, self.files, self._open = [], [], []
        self.n = 0

    def add(self, line):
        self.buf.append(line)
        self.n += 1
        if len(self.buf) >= self.chunk:
            self._spill()

    def _spill(self):
        self.buf.sort()
        path = os.path.join(self.tmpdir, f"{self.tag}_{len(self.files):04d}.txt")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(self.buf)
        self.files.append(path)
        self.buf = []

    def sorted_iter(self):
        if not self.files:
            self.buf.sort()
            return iter(self.buf)
        if self.buf:
            self._spill()
        hs = [open(p, encoding="utf-8", newline="") for p in self.files]
        self._open.extend(hs)
        return heapq.merge(*hs)

    def close(self):
        for h in self._open:
            h.close()
        for p in self.files:
            try:
                os.remove(p)
            except OSError:
                pass
        self.buf, self.files, self._open = [], [], []


def _load_required(s1_path):
    """{s1_id: country_index << 2} with the official read_ids semantics (first column,
    stripped, header skipped, blank lines ignored). Bits 0/1 = seen in matching/candidate."""
    req, countries, cidx = {}, [], {}
    with open(s1_path, encoding="utf-8") as f:
        header = f.readline()
        hcols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        ci = hcols.index("country") if "country" in hcols else None
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(DELIM)
            c = parts[ci].strip() if ci is not None and len(parts) > ci else ""
            i = cidx.get(c)
            if i is None:
                i = cidx[c] = len(countries)
                countries.append(c)
            req[parts[0].strip()] = i << 2
    return req, countries


def _required_order(s1_path):
    with open(s1_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if line.strip():
                yield line.split(DELIM, 1)[0].strip()


def _light_rows(path):
    """(s1, ids list) for every row with a TAB, parsed exactly like the official validator."""
    st = {"bom": False, "cr_lines": 0, "final_newline": True}
    with open(path, "rb") as fb:
        lines = _mirror_lines(fb, st)
        next(lines, None)
        for line in lines:
            s1, tab, rest = line.partition(DELIM)
            if tab:
                yield s1, (rest.rstrip("\n").split(",") if rest.strip() else [])


def _new_stats(countries):
    return {"rows": 0, "empty": 0, "ids": 0, "max_ids": 0, "hist": [0] * (HIST_MAX + 2),
            "s2_ids": 0, "s3_ids": 0, "unsorted_rows": 0, "in_test_order": True,
            "bom": False, "cr_lines": 0, "final_newline": True, "blank_lines": 0,
            "_bc": [[0, 0, 0] for _ in range(len(countries) + 1)]}


# ----------------------------------------------------------------------------- one file
def _check_file(path, kind, req, countries, s1_path, bit, iss, sorter, check_ids, lock):
    """Validate one id-list file in a single streaming pass. Returns (parsed_ok, stats).
    sorter: ExtSorter receiving ids (matching: "id\\ts1\\n", candidate: "id\\n") or None.
    lock:   dict(active, it, offenders) for the lockstep subset check, or None."""
    name = os.path.basename(path)
    st = _new_stats(countries)
    expected = HEADERS[kind]
    if not os.path.isfile(path):
        iss.add(name, "missing", OFFICIAL, f"File not found: {path}")
        return False, st
    seen_extra = set()
    order = _required_order(s1_path)
    with open(path, "rb") as fb:
        lines = _mirror_lines(fb, st)
        try:
            header = next(lines, "")
            if not header:
                iss.add(name, "empty", OFFICIAL, f"{name} is empty.")
                return False, st
            if DELIM not in header and "," in header:
                iss.add(name, "csv", OFFICIAL, f"{name}: header has no TAB but contains commas -- "
                        "the file looks COMMA-separated; write TAB-separated.")
                return False, st
            cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
            if cols != expected:
                hint = (" The file starts with a UTF-8 BOM: write encoding='utf-8' (not utf-8-sig, "
                        "not PowerShell Out-File/>).") if st["bom"] else ""
                iss.add(name, "header", OFFICIAL, f"{name}: unexpected header {cols}. "
                        f"Expected exactly {expected} (tab-separated).{hint}")
                return False, st
            if header.rstrip("\n") != DELIM.join(expected):
                iss.add(name, "header_exact", STRICT, f"{name}: header is not byte-exact "
                        f"(got {header.rstrip(chr(10))!r}); the official validator lower-cases/strips it, "
                        "the scorer may not.")
            bc = st["_bc"]
            for line_num, line in enumerate(lines, start=2):
                s1, tab, rest = line.partition(DELIM)
                if not tab:
                    if s1.strip():
                        iss.add(name, "malformed", OFFICIAL, f"{name}: malformed row (no tab)",
                                example=f"line {line_num}: {line.rstrip()!r}")
                    else:
                        st["blank_lines"] += 1
                    continue
                st["rows"] += 1
                if st["in_test_order"] and next(order, None) != s1:
                    st["in_test_order"] = False
                v = req.get(s1)
                if v is None:
                    iss.add(name, "extra_s1", OFFICIAL,
                            f"{name}: row(s) using an S1 ID that is not in the test set", example=s1)
                    if s1 in seen_extra:
                        iss.add(name, "dup_row", OFFICIAL, f"{name}: duplicate source1_entity_id row(s). "
                                "Each S1 entity may appear on only one row.", example=s1)
                    elif len(seen_extra) < 1_000_000:
                        seen_extra.add(s1)
                    ci = len(countries)
                else:
                    if v & bit:
                        iss.add(name, "dup_row", OFFICIAL, f"{name}: duplicate source1_entity_id row(s). "
                                "Each S1 entity may appear on only one row.", example=s1)
                    else:
                        req[s1] = v | bit
                    ci = v >> 2
                body = rest.rstrip("\n")
                if DELIM in body:
                    iss.add(name, "extra_tab", STRICT, f"{name}: more than 2 TAB-separated columns",
                            example=f"line {line_num}")
                if rest.strip():
                    ids = body.split(",")
                else:
                    ids = []
                    if body:
                        iss.add(name, "blank_field", STRICT, f"{name}: whitespace-only list field "
                                "(write an empty field)", example=s1)
                k = len(ids)
                bc[ci][0] += 1
                bc[ci][2] += k
                st["hist"][min(k, HIST_MAX + 1)] += 1
                if k > st["max_ids"]:
                    st["max_ids"] = k
                if lock is not None and lock["active"]:
                    m = next(lock["it"], None)
                    if m is None or m[0] != s1:
                        lock["active"] = False
                    elif m[1] and not set(m[1]) <= set(ids):
                        lock["offenders"] += 1
                        if len(lock["examples"]) < EX_CAP:
                            lock["examples"].add(s1)
                if not k:
                    st["empty"] += 1
                    bc[ci][1] += 1
                    continue
                st["ids"] += k
                sset = set(ids)
                if len(sset) != k:
                    iss.add(name, "dup_id", OFFICIAL, f"{name}: repeated ID inside a {expected[1]} list. "
                            "No duplicate IDs are allowed within a list.", example=s1)
                if ids != sorted(ids):
                    st["unsorted_rows"] += 1
                has_bad = _BAD_ID_CHARS.search(body) is not None
                for mid in sset:
                    p = mid[:3]
                    if p == "S2-" or p == "S3-":
                        if p == "S2-":
                            st["s2_ids"] += 1
                        else:
                            st["s3_ids"] += 1
                        if has_bad and _BAD_ID_CHARS.search(mid):
                            iss.add(name, "id_chars", STRICT, f"{name}: ids with whitespace/quotes "
                                    "(e.g. a ', ' separator or trailing space)", example=repr(mid))
                            if check_ids:   # stripped pool ids can never contain whitespace
                                iss.add(name, "unknown_id", OFFICIAL, f"{name}: {expected[1]} references "
                                        "IDs not in the test Source-2/3 files", example=repr(mid))
                            continue
                        if sorter is not None:
                            sorter.add(f"{mid}\t{s1}\n" if kind == "matching" else f"{mid}\n")
                    elif p == "S1-":
                        iss.add(name, "self_match", OFFICIAL, f"{name}: {expected[1]} contains Source-1 IDs "
                                "(self-matches). Only S2-/S3- IDs are allowed.", example=mid)
                    else:
                        iss.add(name, "prefix", OFFICIAL, f"{name}: {expected[1]} contains IDs without "
                                "an S2-/S3- prefix", example=repr(mid))
        except UnicodeDecodeError as exc:
            iss.add(name, "utf8", OFFICIAL, f"{name} is not valid UTF-8 ({exc.reason} at byte {exc.start} "
                    "of a line); re-save as plain UTF-8.")
            return False, st
    # missing required S1 (official: required - seen)
    n_missing = sum(1 for v in req.values() if not v & bit)
    if n_missing:
        ex = heapq.nsmallest(SHOW, (k for k, v in req.items() if not v & bit))
        iss.add(name, "missing_s1", OFFICIAL, f"{name}: required S1 entity(ies) missing. Every entity in "
                "test_source1.tsv needs a row (empty = no match).", n=n_missing)
        iss.items[(name, "missing_s1")]["examples"].update(ex)
    if st["cr_lines"]:
        iss.add(name, "crlf", STRICT, f"{name}: {st['cr_lines']} line(s) contain CR (CRLF). The official "
                "validator passes this (text mode), the scorer might not: write LF (newline='').")
    if st["blank_lines"]:
        iss.add(name, "blank_lines", STRICT, f"{name}: {st['blank_lines']} blank line(s)")
    if not st["final_newline"]:
        iss.add(name, "no_final_lf", WARNING, f"{name}: last line has no trailing LF")
    return True, st


def _finish_stats(st, countries):
    rows, bc = st["rows"], st.pop("_bc")
    st["mean_ids_per_row"] = round(st["ids"] / rows, 4) if rows else 0.0
    ne = rows - st["empty"]
    st["mean_ids_per_nonempty_row"] = round(st["ids"] / ne, 4) if ne else 0.0
    st["empty_rate"] = round(st["empty"] / rows, 4) if rows else 0.0
    labels = list(countries) + ["<not in test_source1>"]
    st["by_country"] = {labels[i]: {"rows": r, "empty": e, "ids": n,
                                    "mean_ids_per_row": round(n / r, 4) if r else 0.0,
                                    "empty_rate": round(e / r, 4) if r else 0.0}
                        for i, (r, e, n) in enumerate(bc) if r}
    st["hist"] = {(str(i) if i <= HIST_MAX else f"{HIST_MAX + 1}+"): c for i, c in enumerate(st["hist"])}
    return st


def _scan_sorted_ids(ref_iter, pool_iter, excl, name, col, iss):
    """One pass over sorted referenced ids: unknown ids (vs sorted pool) and, for the
    matching file ("id\\ts1" lines), ids listed under more than one S1."""
    if pool_iter is not None:
        pool_iter = (x.rstrip("\n") for x in pool_iter)   # pool lines end with a newline
    pool = next(pool_iter, None) if pool_iter is not None else None
    prev_id = prev_s1 = None
    n_excl = 0
    for line in ref_iter:
        line = line.rstrip("\n")
        mid, _, s1 = line.partition(DELIM)
        if pool_iter is not None:
            while pool is not None and pool < mid:
                pool = next(pool_iter, None)
            if pool != mid:
                iss.add(name, "unknown_id", OFFICIAL, f"{name}: {col} references IDs not in the test "
                        "Source-2/3 files", example=mid)
        if excl and mid == prev_id and s1 != prev_s1:
            n_excl += 1
            iss.add(name, "exclusivity", WARNING, f"{name}: S2/S3 id(s) listed under more than one S1 "
                    "(train ground truth never does this -- check the exclusivity step)", example=mid)
        prev_id, prev_s1 = mid, s1
    return n_excl


def _pool_sorter(test_dir, tmpdir, chunk, iss):
    s = ExtSorter(tmpdir, "pool", chunk)
    for fn in ("test_source2.tsv", "test_source3.tsv"):
        p = os.path.join(test_dir, fn)
        if not os.path.isfile(p):
            iss.add("-", "no_pool", WARNING, f"{p} not found -- skipping the ID-existence check.")
            s.close()
            return None
        with open(p, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                if line.strip():
                    s.add(line.split(DELIM, 1)[0].strip() + "\n")
    return s


def _subset_external(matching, candidate, tmpdir, chunk):
    """matches ⊆ candidates by external sort of (s1, id) lines + merge. Returns (n, examples)."""
    ms, cs = ExtSorter(tmpdir, "msub", chunk), ExtSorter(tmpdir, "csub", chunk)
    try:
        for s1, ids in _light_rows(matching):
            for mid in set(ids):
                ms.add(f"{s1}\t{mid}\n")
        for s1, ids in _light_rows(candidate):
            for mid in set(ids):
                cs.add(f"{s1}\t{mid}\n")
        ci = cs.sorted_iter()
        c = next(ci, None)
        n, ex, last = 0, set(), None
        for line in ms.sorted_iter():
            while c is not None and c < line:
                c = next(ci, None)
            if c != line:
                s1 = line.split(DELIM, 1)[0]
                if s1 != last:
                    n += 1
                    last = s1
                    if len(ex) < EX_CAP:
                        ex.add(s1)
        return n, ex
    finally:
        ms.close()
        cs.close()


def _peak_rss_gb():
    """Peak RSS of this process in GB (Windows: psutil peak_wset; POSIX: ru_maxrss)."""
    try:
        import resource  # POSIX only
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 2**20 / (1024 if sys.platform == "darwin" else 1), 3)
    except ImportError:
        pass
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2**30, 3)
    except Exception:
        return None


# ----------------------------------------------------------------------------- main API
def check_submission(matching, candidate=None, test_dir=None, check_ids=False, lenient=False,
                     tmp_dir=None, chunk_lines=1_000_000, exclusivity=True):
    """Run every check; returns a report dict (see module docstring). Never raises on bad files."""
    t0 = time.time()
    iss = Issues()
    rep = {"matching": matching, "candidate": candidate, "test_dir": test_dir, "check_ids": check_ids,
           "lenient": lenient, "stats": {}}
    s1_path = os.path.join(test_dir or "", "test_source1.tsv")
    if not os.path.isfile(s1_path):
        iss.add("-", "no_source1", OFFICIAL, f"Test source1 file not found: {s1_path} (check --test-dir).")
        return _verdict(rep, iss, t0)
    req, countries = _load_required(s1_path)
    rep["required_s1"] = len(req)
    rep["countries"] = countries
    tmp = tempfile.mkdtemp(prefix="chk_", dir=tmp_dir)
    sorters, lock = [], None
    try:
        pool = _pool_sorter(test_dir, tmp, chunk_lines, iss) if check_ids else None
        if pool is not None:
            sorters.append(pool)
        use_ids = pool is not None
        msort = ExtSorter(tmp, "mids", chunk_lines) if (use_ids or exclusivity) else None
        if msort is not None:
            sorters.append(msort)
        m_ok, mst = _check_file(matching, "matching", req, countries, s1_path, 1, iss, msort, use_ids, None)
        rep["stats"]["matching"] = _finish_stats(mst, countries)
        mname = os.path.basename(matching)
        if m_ok and msort is not None:
            n_excl = _scan_sorted_ids(msort.sorted_iter(), pool.sorted_iter() if use_ids else None,
                                      exclusivity, mname, HEADERS["matching"][1], iss)
            rep["stats"]["matching"]["ids_under_multiple_s1"] = n_excl
        if msort is not None:
            msort.close()
        if candidate:
            csort = ExtSorter(tmp, "cids", chunk_lines) if use_ids else None
            if csort is not None:
                sorters.append(csort)
            lock = ({"active": True, "it": _light_rows(matching), "offenders": 0, "examples": set()}
                    if m_ok else None)
            c_ok, cst = _check_file(candidate, "candidate", req, countries, s1_path, 2, iss, csort,
                                    use_ids, lock)
            rep["stats"]["candidate"] = _finish_stats(cst, countries)
            cname = os.path.basename(candidate)
            if c_ok and csort is not None:
                _scan_sorted_ids(csort.sorted_iter(), pool.sorted_iter(), False, cname,
                                 HEADERS["candidate"][1], iss)
            if m_ok and c_ok:
                if lock["active"] and next(lock["it"], None) is None:
                    mode, n_off, ex = "lockstep", lock["offenders"], lock["examples"]
                else:
                    mode = "external-sort"
                    n_off, ex = _subset_external(matching, candidate, tmp, chunk_lines)
                rep["stats"]["subset"] = {"mode": mode, "s1_with_matches_outside_candidates": n_off}
                if n_off:
                    iss.add(mname, "subset", STRICT, f"{n_off} S1 entity(ies) have matched IDs not present in "
                            "candidate_pairs.tsv (official: warning; audit: pipeline bug -- write the "
                            "candidate file from the scored pairs)", n=0)
                    it = iss.items[(mname, "subset")]
                    it["count"] = n_off
                    it["examples"].update(list(ex)[:EX_CAP])
        else:
            iss.add("-", "no_candidate", WARNING, "candidate file not given -- skipping candidate_pairs.tsv "
                    "checks and the subset rule (it is still required in the zip).")
    finally:
        if lock is not None:
            lock["it"].close()
        for s in sorters:
            s.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return _verdict(rep, iss, t0)


def _verdict(rep, iss, t0):
    def ser(i):
        return {k: (sorted(v)[:SHOW] if k == "examples" else v) for k, v in i.items()}
    rep["issues"] = [ser(i) for i in iss.items.values()]
    mname = os.path.basename(rep["matching"] or "")
    off_all = iss.of(OFFICIAL)
    off_m = [i for i in off_all if i["file"] in (mname, "-")]
    rep["official_ok"] = not off_all
    rep["official_ok_matching"] = not off_m
    rep["strict_ok"] = not iss.of(STRICT)
    rep["ok"] = rep["official_ok"] and (rep["lenient"] or rep["strict_ok"])
    rep["elapsed_s"] = round(time.time() - t0, 2)
    rep["peak_rss_gb"] = _peak_rss_gb()
    return rep


# ----------------------------------------------------------------------------- official validator
def find_official_validator(test_dir=None, explicit=None):
    """Locate student_resource/utils/validate_submission.py (explicit path, then
    <test_dir>/../../utils, then upwards from the cwd and from this file)."""
    cands = [explicit] if explicit else []
    if test_dir:
        cands.append(os.path.join(test_dir, "..", "..", "utils", "validate_submission.py"))
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        d = os.path.abspath(base)
        while True:
            cands.append(os.path.join(d, "student_resource", "utils", "validate_submission.py"))
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def run_official(validator, matching, test_dir, candidate=None, check_ids=False, python=None):
    """Run the official validator with absolute paths from an EMPTY temp cwd, so the literal
    NONE (and its default output/candidate_pairs.tsv) can never resolve to a real file.
    candidate=None -> '--candidate NONE' (skips the candidate file). Returns a dict."""
    cmd = [python or sys.executable, os.path.abspath(validator), "--matching", os.path.abspath(matching),
           "--candidate", os.path.abspath(candidate) if candidate else "NONE",
           "--test-dir", os.path.abspath(test_dir)]
    if check_ids:
        cmd.append("--check-ids")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    peak = None
    with tempfile.TemporaryDirectory(prefix="offval_") as cwd:
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            import psutil
            ps = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:   # a venv python.exe on Windows is a launcher: the interpreter is its child
                    rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
                    peak = max(peak or 0, rss)
                except psutil.Error:
                    pass
                time.sleep(0.05)
        except ImportError:
            pass
        out, err = proc.communicate()
    return {"cmd": cmd, "returncode": proc.returncode, "pass": proc.returncode == 0,
            "stdout": out.decode("utf-8", "replace"), "stderr": err.decode("utf-8", "replace"),
            "elapsed_s": round(time.time() - t0, 2),
            "peak_rss_gb": round(peak / 2**30, 3) if peak else None}


# ----------------------------------------------------------------------------- CLI
def print_report(rep, out=sys.stdout):
    w = out.write
    w(f"check_submission -- required S1: {rep.get('required_s1', '?')}  countries: {rep.get('countries', [])}\n")
    for kind in ("matching", "candidate"):
        st = rep["stats"].get(kind)
        if not st:
            continue
        w(f"\n[{kind}] {rep[kind]}\n")
        w(f"  rows {st['rows']:,}  empty {st['empty']:,} ({st['empty_rate']:.2%})  ids {st['ids']:,}  "
          f"mean/row {st['mean_ids_per_row']}  mean/non-empty {st['mean_ids_per_nonempty_row']}  "
          f"max {st['max_ids']}  S2 {st['s2_ids']:,} / S3 {st['s3_ids']:,}\n")
        w(f"  format: BOM {'YES' if st['bom'] else 'no'} | CR lines {st['cr_lines']} | blank lines "
          f"{st['blank_lines']} | final LF {'yes' if st['final_newline'] else 'NO'} | rows in test_source1 "
          f"order {'yes' if st['in_test_order'] else 'no'} | unsorted rows {st['unsorted_rows']:,}\n")
        w("  ids/row histogram: " + " ".join(f"{k}:{v}" for k, v in st["hist"].items() if v) + "\n")
        for c, d in st["by_country"].items():
            w(f"  {c:<22} rows {d['rows']:>9,}  empty {d['empty_rate']:6.2%}  mean ids/row {d['mean_ids_per_row']}\n")
        if "ids_under_multiple_s1" in st:
            w(f"  ids listed under >1 S1: {st['ids_under_multiple_s1']:,}\n")
    if "subset" in rep["stats"]:
        s = rep["stats"]["subset"]
        w(f"\n[subset] mode {s['mode']}: S1 with matches outside candidates = {s['s1_with_matches_outside_candidates']:,}\n")
    for level, title in ((OFFICIAL, "ERRORS (official rules -> rejected)"),
                         (STRICT, "STRICT (extra gates%s)" % (" -> downgraded, --lenient" if rep["lenient"] else "")),
                         (WARNING, "WARNINGS")):
        items = [i for i in rep["issues"] if i["level"] == level]
        if items:
            w(f"\n{title}:\n")
            for n, i in enumerate(items, 1):
                ex = (f" [{i['count']:,} x, e.g. {', '.join(map(str, i['examples']))}]"
                      if i["examples"] else (f" [{i['count']:,} x]" if i["count"] > 1 else ""))
                w(f"  {n}. {i['msg']}{ex}\n")
    w(f"\nelapsed {rep['elapsed_s']} s, peak RSS {rep['peak_rss_gb']} GB\n")
    w(("PASS" if rep["ok"] else "FAIL") + f" (official rules {'ok' if rep['official_ok'] else 'FAIL'}, "
      f"strict gates {'ok' if rep['strict_ok'] else 'FAIL'})\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Streaming validator for matching_results.tsv / candidate_pairs.tsv")
    ap.add_argument("--matching", "-m", required=True)
    ap.add_argument("--candidate", "-c", default=None, help="candidate_pairs.tsv (omit to skip; never defaulted)")
    ap.add_argument("--test-dir", "-t", required=True, help="folder with test_source1.tsv (+ 2/3 for --check-ids)")
    ap.add_argument("--check-ids", action="store_true", help="IDs must exist in test_source2/3 (external sort)")
    ap.add_argument("--lenient", action="store_true", help="strict gates become warnings")
    ap.add_argument("--no-exclusivity", action="store_true", help="skip the ids-under->1-S1 diagnostic")
    ap.add_argument("--official", action="store_true",
                    help="also run the official validator on the matching file (--candidate NONE)")
    ap.add_argument("--official-candidate", action="store_true",
                    help="also run the official validator WITH the candidate file (RAM-heavy at full scale)")
    ap.add_argument("--validator", default=None, help="path to validate_submission.py (auto-detected)")
    ap.add_argument("--tmp-dir", default=None, help="where external-sort chunks go (default: system temp)")
    ap.add_argument("--chunk-lines", type=int, default=1_000_000)
    ap.add_argument("--json", default=None, help="write the report as JSON here")
    a = ap.parse_args(argv)
    if a.official_candidate and not a.candidate:
        ap.error("--official-candidate needs --candidate")
    rep = check_submission(a.matching, a.candidate, a.test_dir, check_ids=a.check_ids, lenient=a.lenient,
                           tmp_dir=a.tmp_dir, chunk_lines=a.chunk_lines, exclusivity=not a.no_exclusivity)
    print_report(rep)
    code = 0 if rep["ok"] else 1
    runs = []
    if a.official or a.official_candidate:
        val = find_official_validator(a.test_dir, a.validator)
        if not val:
            print("official validator not found (pass --validator)")
            return 2
        if a.official:
            runs.append(("matching only (--candidate NONE)", rep["official_ok_matching"],
                         run_official(val, a.matching, a.test_dir, None, a.check_ids)))
        if a.official_candidate:
            runs.append(("matching + candidate", rep["official_ok"],
                         run_official(val, a.matching, a.test_dir, a.candidate, a.check_ids)))
        for label, ours, r in runs:
            tail = "\n".join(r["stdout"].strip().splitlines()[-8:])
            print(f"\n[official validator: {label}] exit {r['returncode']} in {r['elapsed_s']} s, "
                  f"peak RSS {r['peak_rss_gb']} GB\n  $ {' '.join(r['cmd'])}\n{tail}")
            if r["pass"] != ours:
                print(f"DISAGREEMENT: official {'PASS' if r['pass'] else 'FAIL'} vs ours "
                      f"{'PASS' if ours else 'FAIL'} on official rules -- investigate before uploading")
                code = 3
        rep["official_runs"] = [{"label": lb, "ours_official_ok": o, **{k: r[k] for k in
                                ("cmd", "returncode", "pass", "elapsed_s", "peak_rss_gb", "stdout")}}
                                for lb, o, r in runs]
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
