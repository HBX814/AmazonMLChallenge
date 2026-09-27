"""pytest tests for outputs.py, check_submission.py and package_submission.py.

Run (Windows, project venv):
  set PYTHONDONTWRITEBYTECODE=1 & set PYTHONIOENCODING=utf-8
  C:/Users/harsh/Downloads/AmazonMLChallenge/.venv/Scripts/python -m pytest -q -p no:cacheprovider test_outputs.py

Needs the mini test split (MINI_TEST, env ER_MINI_TEST overrides) and the official validator
(OFFICIAL_VALIDATOR, env ER_OFFICIAL_VALIDATOR). Every broken-file case asserts that our
streaming checker and the official validator reach the SAME verdict on the official rules.
Temporary files go under ITEST (env ER_ITEST overrides) and are removed afterwards.
"""
import io
import os
import random
import shutil
import sys
import uuid
import zipfile

import polars as pl
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_submission as C  # noqa: E402
import outputs as O  # noqa: E402
import package_submission as P  # noqa: E402

SCRATCH = r"C:/Users/harsh/AppData/Local/Temp/claude/c--Users-harsh-Downloads-AmazonMLChallenge/da46a70e-4037-4864-96e9-d5a67558da5d/scratchpad"
MINI_TEST = os.environ.get("ER_MINI_TEST", SCRATCH + "/mini/test")
ITEST = os.environ.get("ER_ITEST", SCRATCH + "/itest/b-pack")
OFFICIAL_VALIDATOR = os.environ.get(
    "ER_OFFICIAL_VALIDATOR", r"C:/Users/harsh/Downloads/AmazonMLChallenge/student_resource/utils/validate_submission.py")
BLANK_TEMPLATE = os.path.join(os.path.dirname(OFFICIAL_VALIDATOR), "..", "Documentation_template.md")

needs_mini = pytest.mark.skipif(not os.path.isfile(os.path.join(MINI_TEST, "test_source1.tsv")),
                                reason="mini test split not found")
needs_official = pytest.mark.skipif(not os.path.isfile(OFFICIAL_VALIDATOR), reason="official validator not found")


@pytest.fixture
def work():
    d = os.path.join(ITEST, "t_" + uuid.uuid4().hex[:8])
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def rb(path):
    with open(path, "rb") as f:
        return f.read()


# ============================================================================ outputs.py
def test_write_exact_bytes(work):
    pairs = pl.DataFrame({"s1": ["S1-2", "S1-2", "S1-2", "S1-9"], "mid": ["S3-5", "S2-7", "S3-5", "S2-1"]})
    p = os.path.join(work, "m.tsv")
    O.write_id_lists(["S1-9", "S1-1", "S1-2"], pairs, p, "matched_entity_ids")
    assert rb(p) == (b"source1_entity_id\tmatched_entity_ids\n"
                     b"S1-9\tS2-1\nS1-1\t\nS1-2\tS2-7,S3-5\n")


def test_candidate_header_accepts_cand_or_mid_and_lazy(work):
    a, b = os.path.join(work, "a.tsv"), os.path.join(work, "b.tsv")
    df = pl.DataFrame({"s1": ["S1-1", "S1-1"], "cand": ["S3-2", "S2-9"], "p": [0.1, 0.9]})
    O.write_id_lists(pl.Series(["S1-1", "S1-0"]), df.lazy(), a, "candidate_entity_ids")
    O.write_id_lists(["S1-1", "S1-0"], df.rename({"cand": "mid"}), b, "candidate_entity_ids")
    assert rb(a) == rb(b) == b"source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-9,S3-2\nS1-0\t\n"


def test_chunking_gives_identical_bytes(work):
    rng = random.Random(1)
    s1 = [f"S1-{i}" for i in rng.sample(range(10**6), 50)]
    rows = [(s, f"S{rng.choice('23')}-{rng.randint(1, 10**6)}") for s in s1 for _ in range(rng.randint(0, 5))]
    df = pl.DataFrame({"s1": [r[0] for r in rows], "cand": [r[1] for r in rows]})
    a, b = os.path.join(work, "a.tsv"), os.path.join(work, "b.tsv")
    O.write_id_lists(s1, df, a, "candidate_entity_ids")
    O.write_id_lists(s1, df, b, "candidate_entity_ids", chunk_s1=7)
    assert rb(a) == rb(b)
    lines = rb(a).decode("utf-8").split("\n")
    assert lines[-1] == "" and len(lines) == 52 and [ln.split("\t")[0] for ln in lines[1:-1]] == s1
    for ln in lines[1:-1]:
        ids = ln.split("\t")[1].split(",") if ln.split("\t")[1] else []
        assert ids == sorted(set(ids))


@pytest.mark.parametrize("bad", ["S1-5", "S2-1 ", " S3-1", "X-1", "S2-1,S2-2", 'S2-"1'])
def test_writer_rejects_bad_ids(work, bad):
    df = pl.DataFrame({"s1": ["S1-1"], "mid": [bad]})
    with pytest.raises(ValueError):
        O.write_id_lists(["S1-1"], df, os.path.join(work, "x.tsv"), "matched_entity_ids")
    assert not os.path.exists(os.path.join(work, "x.tsv")) and not os.path.exists(os.path.join(work, "x.tsv.tmp"))


def test_writer_rejects_orphans_dups_and_bad_args(work):
    p = os.path.join(work, "x.tsv")
    df = pl.DataFrame({"s1": ["S1-1", "S1-404"], "mid": ["S2-1", "S2-2"]})
    with pytest.raises(ValueError, match="not in s1_ids"):
        O.write_id_lists(["S1-1"], df, p, "matched_entity_ids")
    with pytest.raises(ValueError, match="duplicate"):
        O.write_id_lists(["S1-1", "S1-1"], df.head(1), p, "matched_entity_ids")
    with pytest.raises(ValueError):
        O.write_id_lists(["S1-1"], df.head(1), p, "matches")
    with pytest.raises(ValueError, match="student_resource"):
        O.write_id_lists(["S1-1"], df.head(1), os.path.join(work, "student_resource", "o.tsv"), "matched_entity_ids")
    O.write_id_lists(["S1-1"], df, p, "matched_entity_ids", strict=False)   # orphan dropped when not strict
    assert rb(p) == b"source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n"


def test_write_submission_enforces_subset(work):
    scored = pl.DataFrame({"s1": ["S1-1", "S1-1"], "cand": ["S2-1", "S3-1"], "p": [0.9, 0.2]})
    links = pl.DataFrame({"s1": ["S1-1", "S1-1"], "mid": ["S2-1", "S2-9"]})
    with pytest.raises(ValueError, match="subset"):
        O.write_submission(["S1-1"], links, scored, work)
    info = O.write_submission(["S1-1", "S1-2"], links.head(1), scored, work)
    assert info["n_links"] == 1 and info["n_pairs"] == 2
    assert rb(info["candidate"]).endswith(b"S1-1\tS2-1,S3-1\nS1-2\t\n")


@needs_mini
def test_read_s1_order_matches_official_semantics():
    order = O.read_s1_order(os.path.join(MINI_TEST, "test_source1.tsv"))
    with open(os.path.join(MINI_TEST, "test_source1.tsv"), encoding="utf-8") as f:
        next(f)
        official = {ln.split("\t", 1)[0].strip() for ln in f if ln.strip()}
    assert len(order) == len(set(order)) == len(official) == 9000 and set(order) == official


# ============================================================================ fixtures for validator tests
def make_ok_outputs(d):
    """Valid matching/candidate files for the mini test split (random, exclusive matches)."""
    s1 = O.read_s1_order(os.path.join(MINI_TEST, "test_source1.tsv"))
    pool = O.read_s1_order(os.path.join(MINI_TEST, "test_source2.tsv")) + \
        O.read_s1_order(os.path.join(MINI_TEST, "test_source3.tsv"))
    rng = random.Random(0)
    rng.shuffle(pool)
    it = iter(pool)
    rows = []
    for s in s1:
        own = [next(it) for _ in range(rng.choice([0, 1, 2, 3, 3, 4, 5]))]
        rows += [(s, c, 0.9) for c in own]
        rows += [(s, c, 0.1) for c in rng.sample(pool, rng.randint(0, 6)) if c not in own]
    scored = pl.DataFrame({"s1": [r[0] for r in rows], "cand": [r[1] for r in rows], "p": [r[2] for r in rows]})
    links = scored.filter(pl.col("p") > 0.5).select("s1", pl.col("cand").alias("mid"))
    return O.write_submission(s1, links, scored, d)


def official(m, c=None, check_ids=False):
    return C.run_official(OFFICIAL_VALIDATOR, m, MINI_TEST, c, check_ids)


def mutate(src, dst, fn):
    data = rb(src)
    with open(dst, "wb") as f:
        f.write(fn(data))
    return dst


def _rows(data):
    return data.split(b"\n")


def _first_nonempty_row(lines):
    return next(i for i, ln in enumerate(lines) if i and ln.count(b"\t") == 1 and not ln.endswith(b"\t"))


def m_comma_space(d):
    return d.replace(b",", b", ")


def m_drop_row(d):
    ls = _rows(d)
    del ls[5]
    return b"\n".join(ls)


def m_dup_id(d):
    ls = _rows(d)
    i = _first_nonempty_row(ls)
    first = ls[i].split(b"\t")[1].split(b",")[0]
    ls[i] += b"," + first
    return b"\n".join(ls)


def m_s1_in_list(d):
    ls = _rows(d)
    i = _first_nonempty_row(ls)
    ls[i] += b",S1-241771779"
    return b"\n".join(ls)


def m_dup_row(d):
    ls = _rows(d)
    ls.insert(3, ls[2])
    return b"\n".join(ls)


def m_extra_s1(d):
    return d + b"S1-999999999999\t\n"


def m_trailing_comma(d):
    ls = _rows(d)
    ls[_first_nonempty_row(ls)] += b","
    return b"\n".join(ls)


def m_unknown_id(d):
    ls = _rows(d)
    ls[_first_nonempty_row(ls)] += b",S2-000"
    return b"\n".join(ls)


def m_trailing_space_id(d):
    ls = _rows(d)
    ls[_first_nonempty_row(ls)] += b" "
    return b"\n".join(ls)


def m_malformed(d):
    return d + b"S1-garbage-row-without-tab\n"


def m_bad_utf8(d):
    return d + b"S1-\xff\xfe\t\n"


def m_extra_tab(d):
    ls = _rows(d)
    ls[_first_nonempty_row(ls)] += b"\tjunk"
    return b"\n".join(ls)


def m_exclusivity(d):
    ls = _rows(d)
    i = _first_nonempty_row(ls)
    first = ls[i].split(b"\t")[1].split(b",")[0]
    j = next(k for k in range(i + 1, len(ls)) if ls[k].count(b"\t") == 1 and first not in ls[k])
    ls[j] = ls[j] + (b"," if not ls[j].endswith(b"\t") else b"") + first
    return b"\n".join(ls)


# name -> (mutation, official verdict on the official rules, our strict verdict, check_ids)
MATCHING_CASES = {
    "ok": (lambda d: d, True, True, False),
    "bom": (lambda d: b"\xef\xbb\xbf" + d, False, False, False),
    "comma_space": (m_comma_space, False, False, False),
    "missing_s1": (m_drop_row, False, False, False),
    "dup_id": (m_dup_id, False, False, False),
    "s1_in_list": (m_s1_in_list, False, False, False),
    "dup_row": (m_dup_row, False, False, False),
    "extra_s1": (m_extra_s1, False, False, False),
    "trailing_comma": (m_trailing_comma, False, False, False),
    "csv_header": (lambda d: d.replace(b"source1_entity_id\tmatched", b"source1_entity_id,matched", 1)
                   .replace(b"\t", b","), False, False, False),
    "wrong_header": (lambda d: d.replace(b"matched_entity_ids", b"matched_ids", 1), False, False, False),
    "empty_file": (lambda d: b"", False, False, False),
    "malformed_row": (m_malformed, False, False, False),
    "bad_utf8": (m_bad_utf8, False, False, False),
    "unknown_id_nocheck": (m_unknown_id, True, True, False),
    "unknown_id_check": (m_unknown_id, False, False, True),
    "crlf": (lambda d: d.replace(b"\n", b"\r\n"), True, False, False),
    "header_case": (lambda d: d.replace(b"source1_entity_id", b"Source1_Entity_ID", 1), True, False, False),
    "blank_line": (lambda d: d + b"\n", True, False, False),
    "trailing_space_nocheck": (m_trailing_space_id, True, False, False),
    "trailing_space_check": (m_trailing_space_id, False, False, True),
    "extra_tab": (m_extra_tab, True, False, False),
    "exclusivity_warning": (m_exclusivity, True, True, False),
}


@pytest.fixture(scope="module")
def ok_files():
    d = os.path.join(ITEST, "ok_" + uuid.uuid4().hex[:8])
    os.makedirs(d)
    info = make_ok_outputs(d)
    yield info
    shutil.rmtree(d, ignore_errors=True)


@needs_mini
@needs_official
@pytest.mark.parametrize("case", list(MATCHING_CASES))
def test_matching_file_agrees_with_official(case, ok_files, work):
    fn, off_expected, ours_strict, check_ids = MATCHING_CASES[case]
    m = mutate(ok_files["matching"], os.path.join(work, "matching_results.tsv"), fn)
    off = official(m, None, check_ids)
    assert "NONE not found" in off["stdout"] or not off["pass"]
    assert off["pass"] is off_expected, off["stdout"]
    rep = C.check_submission(m, None, MINI_TEST, check_ids=check_ids, tmp_dir=work, chunk_lines=5000)
    assert rep["official_ok_matching"] is off_expected, rep["issues"]
    assert rep["ok"] is ours_strict, rep["issues"]
    lenient = C.check_submission(m, None, MINI_TEST, check_ids=check_ids, lenient=True, tmp_dir=work)
    assert lenient["ok"] is off_expected
    if case == "exclusivity_warning":
        assert rep["stats"]["matching"]["ids_under_multiple_s1"] == 1
    if case == "bom":
        assert rep["stats"]["matching"]["bom"] and "BOM" in str(rep["issues"])


@needs_mini
@needs_official
def test_omitting_candidate_still_loads_default_path(ok_files, work):
    """Why --candidate NONE: without it the validator reads output/candidate_pairs.tsv from the cwd."""
    import subprocess
    os.makedirs(os.path.join(work, "output"))
    with open(os.path.join(work, "output", "candidate_pairs.tsv"), "wb") as f:
        f.write(b"source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-1\n")
    cmd = [sys.executable, OFFICIAL_VALIDATOR, "--matching", ok_files["matching"], "--test-dir", MINI_TEST]
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, encoding="utf-8",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode == 1 and "candidate_pairs.tsv" in r.stdout
    assert official(ok_files["matching"])["pass"]


def c_drop_matched_id(m_data, c_data):
    """Remove one matched id from its candidate row -> subset violation."""
    mrows = {ln.split(b"\t")[0]: ln.split(b"\t")[1] for ln in _rows(m_data)[1:] if b"\t" in ln}
    ls = _rows(c_data)
    for i, ln in enumerate(ls[1:], 1):
        if b"\t" not in ln:
            continue
        s1, ids = ln.split(b"\t")
        if mrows.get(s1):
            victim = mrows[s1].split(b",")[0]
            ls[i] = s1 + b"\t" + b",".join(x for x in ids.split(b",") if x != victim)
            break
    return b"\n".join(ls)


def c_shuffle(data):
    ls = _rows(data)
    body = [ln for ln in ls[1:] if ln]
    random.Random(3).shuffle(body)
    return b"\n".join([ls[0]] + body) + b"\n"


# name -> (candidate mutation(m_data, c_data), official verdict, our strict verdict, subset mode, offenders)
CANDIDATE_CASES = {
    "ok": (lambda m, c: c, True, True, "lockstep", 0),
    "shuffled": (lambda m, c: c_shuffle(c), True, True, "external-sort", 0),
    "subset_violation": (c_drop_matched_id, True, False, "lockstep", 1),
    "shuffled_subset_violation": (lambda m, c: c_shuffle(c_drop_matched_id(m, c)), True, False, "external-sort", 1),
    "missing_s1": (lambda m, c: m_drop_row(c), False, False, None, None),
    "bom": (lambda m, c: b"\xef\xbb\xbf" + c, False, False, None, None),
    "comma_space": (lambda m, c: m_comma_space(c), False, False, None, None),
    "dup_id": (lambda m, c: m_dup_id(c), False, False, None, None),
    "s1_in_list": (lambda m, c: m_s1_in_list(c), False, False, None, None),
    "matching_header": (lambda m, c: c.replace(b"candidate_entity_ids", b"matched_entity_ids", 1), False, False, None, None),
}


@needs_mini
@needs_official
@pytest.mark.parametrize("case", list(CANDIDATE_CASES))
def test_candidate_file_agrees_with_official(case, ok_files, work):
    fn, off_expected, ours_strict, mode, n_off = CANDIDATE_CASES[case]
    m = ok_files["matching"]
    c = os.path.join(work, "candidate_pairs.tsv")
    with open(c, "wb") as f:
        f.write(fn(rb(m), rb(ok_files["candidate"])))
    off = official(m, c, check_ids=True)
    assert off["pass"] is off_expected, off["stdout"]
    rep = C.check_submission(m, c, MINI_TEST, check_ids=True, tmp_dir=work, chunk_lines=5000)
    assert rep["official_ok"] is off_expected, rep["issues"]
    assert rep["ok"] is ours_strict, rep["issues"]
    if mode:
        assert rep["stats"]["subset"]["mode"] == mode
        assert rep["stats"]["subset"]["s1_with_matches_outside_candidates"] == n_off
        assert ("not present in candidate_pairs.tsv" in off["stdout"]) is bool(n_off)


@needs_mini
def test_stats_and_cli_json(ok_files, work):
    js = os.path.join(work, "rep.json")
    code = C.main(["--matching", ok_files["matching"], "--candidate", ok_files["candidate"],
                   "--test-dir", MINI_TEST, "--json", js, "--tmp-dir", work])
    assert code == 0 and os.path.isfile(js)
    rep = C.check_submission(ok_files["matching"], ok_files["candidate"], MINI_TEST)
    st = rep["stats"]["matching"]
    assert st["rows"] == 9000 and set(st["by_country"]) == {"India", "France", "US"}
    assert st["in_test_order"] and st["unsorted_rows"] == 0 and st["cr_lines"] == 0
    assert abs(st["mean_ids_per_row"] - st["ids"] / 9000) < 1e-3
    assert sum(st["hist"].values()) == 9000


@needs_mini
@needs_official
def test_cli_official_flag(ok_files, work):
    assert C.main(["--matching", ok_files["matching"], "--candidate", ok_files["candidate"], "--test-dir",
                   MINI_TEST, "--official", "--official-candidate", "--validator", OFFICIAL_VALIDATOR,
                   "--tmp-dir", work]) == 0
    bad = mutate(ok_files["matching"], os.path.join(work, "matching_results.tsv"), m_comma_space)
    assert C.main(["--matching", bad, "--test-dir", MINI_TEST, "--official", "--validator",
                   OFFICIAL_VALIDATOR]) == 1


def test_external_sorter_multi_chunk(work):
    s = C.ExtSorter(work, "t", chunk=3)
    vals = [f"S2-{i}\n" for i in random.Random(5).sample(range(1000), 50)]
    for v in vals:
        s.add(v)
    assert list(s.sorted_iter()) == sorted(vals) and len(s.files) > 5
    assert list(s.sorted_iter()) == sorted(vals)   # re-iterable
    s.close()
    assert not [f for f in os.listdir(work) if f.startswith("t_")]


# ============================================================================ package_submission.py
GOOD_README = ("# Master Bolt\n\npython -m pip install -r requirements.txt\n"
               "python src/run_pipeline.py --stage all\n")
GOOD_REQ = "polars==1.44.2\n"
GOOD_SRC = "import polars as pl\nimport os\n\n\ndef main():\n    return pl.DataFrame()\n"


def make_tree(root, ok_info=None):
    code = os.path.join(root, "code", "business_entity_resolution")
    os.makedirs(os.path.join(code, "src", "ber", "__pycache__"))
    files = {
        os.path.join(code, "src", "run_pipeline.py"): GOOD_SRC,
        os.path.join(code, "src", "ber", "__init__.py"): "",
        os.path.join(code, "src", "ber", "outputs.py"): open(os.path.join(HERE, "outputs.py"), encoding="utf-8").read(),
        os.path.join(code, "src", "check_submission.py"): open(os.path.join(HERE, "check_submission.py"),
                                                               encoding="utf-8").read(),
        os.path.join(code, "README.md"): GOOD_README,
        os.path.join(code, "requirements.txt"): GOOD_REQ,
        os.path.join(root, "Documentation_template.md"): "# Master Bolt\n**Team Name:** Master Bolt\nfilled.\n",
    }
    for p, t in files.items():
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(t)
    for junk in ("src/ber/__pycache__/outputs.cpython-310.pyc", "src/ber/cache.parquet", "run.log", "model.lgb"):
        with open(os.path.join(code, *junk.split("/")), "wb") as f:
            f.write(b"\0" * 64)
    os.makedirs(os.path.join(root, "student_resource"))
    shutil.copy(BLANK_TEMPLATE if os.path.isfile(BLANK_TEMPLATE) else files[os.path.join(root, "Documentation_template.md")],
                os.path.join(root, "student_resource", "Documentation_template.md"))
    os.makedirs(os.path.join(root, "output"))
    if ok_info:
        shutil.copy(ok_info["matching"], os.path.join(root, "output", "matching_results.tsv"))
        shutil.copy(ok_info["candidate"], os.path.join(root, "output", "candidate_pairs.tsv"))
        future = os.path.getmtime(os.path.join(code, "src", "run_pipeline.py")) + 10
        for n in ("matching_results.tsv", "candidate_pairs.tsv"):
            os.utime(os.path.join(root, "output", n), (future, future))
    return code


def pkg(root, **kw):
    kw.setdefault("check_test_dir", MINI_TEST)
    buf = io.StringIO()
    rep = P.build_package(root, out=buf, tmp_dir=root, **kw)
    rep["text"] = buf.getvalue()
    return rep


@needs_mini
def test_package_happy_path(ok_files, work):
    make_tree(work, ok_files)
    rep = pkg(work)
    assert rep["ok"], rep["text"]
    with zipfile.ZipFile(rep["zip"]) as zf:
        names = sorted(zf.namelist())
        assert zf.testzip() is None
    assert names == sorted([
        "Documentation_template.md", "output/candidate_pairs.tsv", "output/matching_results.tsv",
        "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
        "code/business_entity_resolution/src/check_submission.py", "code/business_entity_resolution/src/run_pipeline.py",
        "code/business_entity_resolution/src/ber/__init__.py", "code/business_entity_resolution/src/ber/outputs.py"])
    assert "EXCLUDED (cache/binary)" in rep["text"] and "cache.parquet" in rep["text"]
    assert rep["zip"].endswith(os.path.join("submission", "Master_Bolt_submission.zip"))


@needs_mini
@pytest.mark.parametrize("case", ["openai_import", "url", "unidecode_req", "unpinned", "missing_pin",
                                  "blank_doc", "placeholder_doc", "stale", "missing_candidate",
                                  "bad_output", "binary_extra", "undocumented_extra", "in_student_resource"])
def test_package_gates_fail(case, ok_files, work):
    code = make_tree(work, ok_files)
    kw = {}
    src = os.path.join(code, "src", "ber")
    if case == "openai_import":
        open(os.path.join(src, "llm.py"), "w").write("import openai\n")
    elif case == "url":
        open(os.path.join(src, "geo.py"), "w").write("URL = 'https://nominatim.example/search'\n")
    elif case == "unidecode_req":
        open(os.path.join(code, "requirements.txt"), "a").write("Unidecode==1.4.0\n")
    elif case == "unpinned":
        open(os.path.join(code, "requirements.txt"), "a").write("numpy>=2\n")
    elif case == "missing_pin":
        open(os.path.join(src, "feat.py"), "w").write("import rapidfuzz\n")
    elif case == "blank_doc":
        shutil.copy(os.path.join(work, "student_resource", "Documentation_template.md"),
                    os.path.join(work, "Documentation_template.md"))
    elif case == "placeholder_doc":
        open(os.path.join(work, "Documentation_template.md"), "a").write("**Team Name:** [Your Team Name]\n")
    elif case == "stale":
        os.utime(os.path.join(code, "src", "run_pipeline.py"), None)
        os.utime(os.path.join(work, "output", "matching_results.tsv"), (1, 1))
    elif case == "missing_candidate":
        os.remove(os.path.join(work, "output", "candidate_pairs.tsv"))
    elif case == "bad_output":
        mutate(ok_files["matching"], os.path.join(work, "output", "matching_results.tsv"), lambda d: b"\xef\xbb\xbf" + d)
    elif case == "binary_extra":
        open(os.path.join(code, "README.md"), "a").write("model.lgb is the trained booster\n")
        kw["extras"] = ["code/business_entity_resolution/model.lgb"]
    elif case == "undocumented_extra":
        open(os.path.join(src, "abbrev.tsv"), "w").write("st\tstreet\n")
        kw["extras"] = ["code/business_entity_resolution/src/ber/abbrev.tsv"]
    elif case == "in_student_resource":
        kw["zip_path"] = os.path.join(work, "student_resource", "x.zip")
    rep = pkg(work, **kw)
    assert not rep["ok"], rep["text"]
    assert not os.path.exists(os.path.join(work, "submission", "Master_Bolt_submission.zip"))


@needs_mini
def test_package_overrides_and_documented_extra(ok_files, work):
    code = make_tree(work, ok_files)
    open(os.path.join(code, "README.md"), "a").write("model.lgb: trained booster shipped for audit\n")
    os.utime(os.path.join(work, "output", "matching_results.tsv"), (1, 1))
    rep = pkg(work, extras=["code/business_entity_resolution/model.lgb"], allow_binary=True, allow_stale=True)
    assert rep["ok"], rep["text"]
    assert any("stale" in w for w in rep["warnings"])
    with zipfile.ZipFile(rep["zip"]) as zf:
        assert "code/business_entity_resolution/model.lgb" in zf.namelist()


@needs_mini
def test_package_dry_run_and_self_skip(ok_files, work):
    code = make_tree(work, ok_files)
    shutil.copy(os.path.join(HERE, "package_submission.py"), os.path.join(code, "src", "package_submission.py"))
    rep = pkg(work, dry_run=True)
    assert rep["ok"], rep["text"]
    assert any("it is this packager" in n for n in rep["notes"])
    assert not os.path.exists(os.path.join(work, "submission"))
