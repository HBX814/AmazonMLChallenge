#!/usr/bin/env python
"""
package_submission.py -- build <team>_submission.zip (default Master_Bolt_submission.zip)
from an explicit ALLOWLIST, after gates that catch what gets a package rejected in review.

Zip layout (student_resource/README.md, "Final Submission Package"):
    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/src/**/*.py
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    Documentation_template.md                          (the FILLED copy, zip root)
    + files given with --extra: must live under code/business_entity_resolution/ and be named
      in README.md; caches/model binaries additionally need --allow-binary

Gates (any error -> exit 1 and no zip is written):
    G1 required files exist; src/ has .py files (warning if src/run_pipeline.py is absent)
    G2 forbidden strings in src/*.py, requirements.txt and text extras (FORBIDDEN_RE below)
    G3 requirements.txt: every line pinned name==version; no GPL/AGPL/API-client packages;
       every third-party import in src/ is pinned (inside try/except -> warning); pins that
       src/ never imports -> warning (pip-freeze smell); pin != installed version -> warning
    G4 Documentation_template.md filled: differs from the blank student_resource copy and has
       no template placeholders ("[Your Team Name]", ...)
    G5 outputs not stale: no src/*.py newer than the output files (--allow-stale overrides)
    G6 outputs pass check_submission.py (streaming, strict) against the test dir
       (auto: <root>/student_resource/dataset/test; --check-test-dir; --skip-check)
    G7 nothing is written inside student_resource/
Non-allowlisted files under code/business_entity_resolution/ are listed as EXCLUDED
(caches such as .npy/.parquet/.pkl/.log/.duckdb/__pycache__ are flagged).
After writing: the zip is re-opened, testzip() CRC-checked, names == allowlist, sizes ==
sources, and the listing with sizes is printed.

Usage:
    python package_submission.py --project-root C:/Users/harsh/Downloads/AmazonMLChallenge --dry-run
    python package_submission.py --project-root C:/Users/harsh/Downloads/AmazonMLChallenge
    python package_submission.py --project-root <root> --extra code/business_entity_resolution/src/ber/data/x.tsv
Library:
    from package_submission import build_package
    rep = build_package(root, dry_run=True)   # rep["ok"], rep["errors"], rep["warnings"], rep["entries"]
Exit codes: 0 ok, 1 gate failed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import sys
import time
import zipfile
from pathlib import Path

CODE = "code/business_entity_resolution"
REQUIRED = ["output/matching_results.tsv", "output/candidate_pairs.tsv",
            f"{CODE}/README.md", f"{CODE}/requirements.txt", "Documentation_template.md"]
OUTPUTS = REQUIRED[:2]
FORBIDDEN_RE = re.compile(r"groq|openai|anthropic|unidecode|aksharamukha|requests\.(get|post)|"
                          r"urllib\.request|https?://", re.IGNORECASE)
DENY_PKGS = {"unidecode", "text-unidecode", "aksharamukha", "groq", "openai", "anthropic",
             "levenshtein", "python-levenshtein", "igraph", "python-igraph", "leidenalg", "zingg",
             "postal", "pypostal"}
CACHE_EXT = {".npy", ".npz", ".parquet", ".pkl", ".pickle", ".joblib", ".log", ".duckdb", ".wal",
             ".pyc", ".pyo", ".feather", ".arrow", ".ipc", ".h5", ".hdf5", ".pt", ".pth", ".bin",
             ".onnx", ".safetensors", ".ckpt", ".model", ".lgb", ".cbm", ".gz", ".zip", ".7z", ".tar",
             ".db", ".sqlite", ".faiss", ".index", ".mmap"}
TEXT_EXT = {".py", ".md", ".txt", ".tsv", ".csv", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
PLACEHOLDERS = ["[Your Team Name]", "[List all team members]", "[Date]", "[your best validation score]",
                "[total]", "[brief description]", "[Brief description of your main technical contribution]"]
IMPORT_TO_DIST = {"sklearn": "scikit-learn", "sparse_dot_topn": "sparse-dot-topn", "yaml": "pyyaml",
                  "PIL": "pillow", "faiss": "faiss-cpu", "indic_transliteration": "indic-transliteration",
                  "sentence_transformers": "sentence-transformers", "cv2": "opencv-python",
                  "dateutil": "python-dateutil", "rapidfuzz": "rapidfuzz"}


def _norm_dist(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _inside(path: Path, folder_name: str) -> bool:
    return any(p.lower() == folder_name for p in path.resolve().parts)


# ----------------------------------------------------------------------------- gates
def _forbidden_hits(path, rel, errors, notes, self_sha):
    if self_sha and _sha256(path) == self_sha:
        notes.append(f"G2 skipped {rel}: it is this packager (it defines the patterns)")
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            m = FORBIDDEN_RE.search(line)
            if m:
                errors.append(f"G2 forbidden string {m.group(0)!r} in {rel}:{n}: {line.strip()[:120]}")


class _Imports(ast.NodeVisitor):
    """Top-level module names imported; `optional` when inside a try: body."""

    def __init__(self):
        self.req, self.opt, self._try = set(), set(), 0

    def visit_Try(self, node):
        self._try += 1
        for n in node.body:
            self.visit(n)
        self._try -= 1
        for n in node.handlers + node.orelse + node.finalbody:
            self.visit(n)

    visit_TryStar = visit_Try

    def _add(self, name):
        (self.opt if self._try else self.req).add(name.split(".")[0])

    def visit_Import(self, node):
        for a in node.names:
            self._add(a.name)

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            self._add(node.module)


def _stdlib_names():
    names = set(getattr(sys, "stdlib_module_names", ()))
    return names | {"__future__", "typing_extensions"} if names else {
        "os", "sys", "re", "io", "json", "csv", "math", "time", "argparse", "pathlib", "typing", "collections",
        "itertools", "functools", "dataclasses", "hashlib", "heapq", "random", "shutil", "subprocess",
        "tempfile", "unicodedata", "zipfile", "multiprocessing", "concurrent", "logging", "gc", "__future__"}


def _check_requirements(req_path, src_dir, py_files, errors, warnings):
    pins = {}
    for n, raw in enumerate(Path(req_path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]+\])?==([^\s;]+)\s*(;.*)?$", line)
        if not m:
            errors.append(f"G3 requirements.txt:{n} not pinned as name==version: {raw.strip()!r}")
            continue
        pins[_norm_dist(m.group(1))] = m.group(3)
    for d in sorted(set(pins) & DENY_PKGS):
        errors.append(f"G3 requirements.txt pins a forbidden package (GPL/AGPL or API client): {d}")
    if len(pins) > 25:
        warnings.append(f"G3 requirements.txt has {len(pins)} pins -- looks like `pip freeze`; "
                        "hand-write pins for what src/ imports")
    # every module shipped under src/ (incl. src/ber/*.py) is local: flat-layout fallback imports such as
    # `except ImportError: from blocking import ...` refer to these siblings, not to third-party packages
    local = ({p.stem for p in Path(src_dir).rglob("*.py") if "__pycache__" not in p.parts}
             | {p.name for p in Path(src_dir).rglob("*") if p.is_dir() and p.name != "__pycache__"})
    stdlib = _stdlib_names()
    req_mods, opt_mods = set(), set()
    for f in py_files:
        try:
            tree = ast.parse(Path(f).read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as exc:
            errors.append(f"G3 cannot parse {f}: {exc}")
            continue
        v = _Imports()
        v.visit(tree)
        req_mods |= v.req
        opt_mods |= v.opt
    third = lambda mods: {m for m in mods if m not in stdlib and m not in local}  # noqa: E731
    needed = {_norm_dist(IMPORT_TO_DIST.get(m, m)): m for m in third(req_mods)}
    optional = {_norm_dist(IMPORT_TO_DIST.get(m, m)): m for m in third(opt_mods) - third(req_mods)}
    for d, m in sorted(needed.items()):
        if d not in pins:
            errors.append(f"G3 src/ imports '{m}' but requirements.txt does not pin '{d}'")
    for d, m in sorted(optional.items()):
        if d not in pins:
            warnings.append(f"G3 src/ optionally imports '{m}' (inside try:) -- not pinned")
    for d in sorted(set(pins) - set(needed) - set(optional)):
        warnings.append(f"G3 requirements.txt pins '{d}' but src/ never imports it (ok only if it is a "
                        "runtime backend, e.g. pyarrow for parquet)")
    try:
        from importlib import metadata
        for d, v in sorted(pins.items()):
            try:
                inst = metadata.version(d)
            except metadata.PackageNotFoundError:
                continue
            if inst != v:
                warnings.append(f"G3 pin {d}=={v} but this interpreter has {inst}")
    except ImportError:
        pass
    return pins


def _check_doc(doc, blank, errors, allow_placeholders):
    text = Path(doc).read_text(encoding="utf-8", errors="replace")
    if blank and Path(blank).is_file():
        if text.replace("\r\n", "\n").strip() == Path(blank).read_text(encoding="utf-8").replace("\r\n", "\n").strip():
            errors.append("G4 Documentation_template.md is the blank template -- fill it (see documentation_guide.md)")
            return
    left = [p for p in PLACEHOLDERS if p in text]
    if left and not allow_placeholders:
        errors.append(f"G4 Documentation_template.md still has template placeholders: {left}")


# ----------------------------------------------------------------------------- main API
def build_package(project_root, zip_path=None, team="Master Bolt", extras=(), allow_binary=False,
                  allow_stale=False, allow_placeholders=False, check_test_dir=None, skip_check=False,
                  check_ids=False, dry_run=False, max_extra_mb=20.0, tmp_dir=None, out=sys.stdout):
    root = Path(project_root).resolve()
    errors, warnings, notes = [], [], []
    rep = {"root": str(root), "errors": errors, "warnings": warnings, "notes": notes, "entries": []}
    zip_path = Path(zip_path) if zip_path else root / "submission" / f"{team.replace(' ', '_')}_submission.zip"
    zip_path = zip_path.resolve()
    rep["zip"] = str(zip_path)
    if _inside(root, "student_resource") or _inside(zip_path, "student_resource"):
        errors.append("G7 project root / zip path is inside student_resource/ (read-only organiser files)")
        return _done(rep, out)

    # G1 allowlist
    entries = []
    for rel in REQUIRED:
        p = root / rel
        if p.is_file():
            entries.append((rel, p))
        else:
            errors.append(f"G1 missing required file: {rel}")
    src = root / CODE / "src"
    py_files = sorted(p for p in src.rglob("*.py") if "__pycache__" not in p.parts) if src.is_dir() else []
    if not py_files:
        errors.append(f"G1 no .py files under {CODE}/src/")
    elif not (src / "run_pipeline.py").is_file():
        warnings.append(f"G1 {CODE}/src/run_pipeline.py not found (the README must name the entry point)")
    entries += [(p.relative_to(root).as_posix(), p) for p in py_files]

    readme = root / CODE / "README.md"
    readme_text = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
    for rel in extras:
        p = (root / rel).resolve()
        relp = p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)
        if not p.is_file():
            errors.append(f"extra not found: {rel}")
            continue
        if not relp.startswith(CODE + "/"):
            errors.append(f"extra must live under {CODE}/: {relp}")
            continue
        if p.suffix.lower() in CACHE_EXT and not allow_binary:
            errors.append(f"refused extra {relp}: cache/binary type {p.suffix} (pass --allow-binary if the "
                          "pipeline truly needs it and README.md documents it)")
            continue
        if p.name not in readme_text:
            errors.append(f"refused extra {relp}: not documented in README.md (name the file and why it ships)")
            continue
        if p.stat().st_size > max_extra_mb * 2**20:
            errors.append(f"refused extra {relp}: {p.stat().st_size / 2**20:.1f} MB > --max-extra-mb {max_extra_mb}")
            continue
        entries.append((relp, p))
    allowed = {rel for rel, _ in entries}
    code_dir = root / CODE
    if code_dir.is_dir():
        for p in sorted(code_dir.rglob("*")):
            if p.is_file() and p.relative_to(root).as_posix() not in allowed:
                rel = p.relative_to(root).as_posix()
                kind = "cache/binary" if (p.suffix.lower() in CACHE_EXT or "__pycache__" in p.parts) else "not allowlisted"
                notes.append(f"EXCLUDED ({kind}): {rel} ({p.stat().st_size:,} B)")

    # G2 forbidden strings
    self_sha = _sha256(__file__)
    scan = [p for p in py_files] + [root / CODE / "requirements.txt"] + \
           [p for rel, p in entries if rel not in REQUIRED and p.suffix.lower() in TEXT_EXT and p not in py_files]
    for p in scan:
        if p.is_file():
            _forbidden_hits(p, p.relative_to(root).as_posix(), errors, notes, self_sha)

    # G3 requirements
    reqf = root / CODE / "requirements.txt"
    if reqf.is_file() and src.is_dir():
        rep["pins"] = _check_requirements(reqf, src, py_files, errors, warnings)
    if readme_text:
        for word in ("requirements.txt", "run_pipeline"):
            if word not in readme_text:
                warnings.append(f"README.md never mentions {word!r}")

    # G4 documentation
    doc = root / "Documentation_template.md"
    if doc.is_file():
        _check_doc(doc, root / "student_resource" / "Documentation_template.md", errors, allow_placeholders)

    # G5 stale outputs
    outs = [root / r for r in OUTPUTS if (root / r).is_file()]
    if outs and py_files:
        newest_py = max(py_files, key=lambda p: p.stat().st_mtime)
        oldest_out = min(outs, key=lambda p: p.stat().st_mtime)
        if newest_py.stat().st_mtime > oldest_out.stat().st_mtime:
            msg = (f"G5 {newest_py.relative_to(root).as_posix()} is newer than {oldest_out.relative_to(root).as_posix()} "
                   "-- outputs may be stale; re-run the pipeline (or --allow-stale if the change cannot affect them)")
            (warnings if allow_stale else errors).append(msg)

    # G6 output validation
    if len(outs) == 2 and not skip_check:
        tdir = Path(check_test_dir) if check_test_dir else root / "student_resource" / "dataset" / "test"
        if not (tdir / "test_source1.tsv").is_file():
            errors.append(f"G6 test dir not found for output validation: {tdir} (pass --check-test-dir or --skip-check)")
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from check_submission import check_submission
            chk = check_submission(str(outs[0]), str(outs[1]), str(tdir), check_ids=check_ids, tmp_dir=tmp_dir)
            rep["check"] = {k: chk[k] for k in ("ok", "official_ok", "strict_ok", "elapsed_s")}
            for i in chk["issues"]:
                line = f"G6 {i['level']}: {i['msg']} {i['examples'][:3] if i['examples'] else ''}".rstrip()
                (errors if i["level"] in ("official", "strict") else warnings).append(line)
            ms, cs = chk["stats"].get("matching", {}), chk["stats"].get("candidate", {})
            if ms and cs and ms.get("ids") == cs.get("ids"):
                warnings.append("G6 candidate_pairs.tsv has exactly as many ids as matching_results.tsv -- it must be "
                                "every pair the model SCORED, not the final matches")
    elif skip_check:
        warnings.append("G6 output validation skipped (--skip-check)")

    rep["entries"] = [(rel, p.stat().st_size) for rel, p in sorted(entries)]
    if errors or dry_run:
        return _done(rep, out)

    # write + verify
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    t0 = time.time()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True,
                         strict_timestamps=False) as zf:   # clamps pre-1980 mtimes instead of crashing
        for rel, p in sorted(entries):
            zf.write(p, rel)
    os.replace(tmp, zip_path)
    rep["zip_seconds"] = round(time.time() - t0, 1)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            errors.append(f"zip CRC check failed at {bad}")
        infos = zf.infolist()
        names = [i.filename for i in infos]
        if sorted(names) != sorted(allowed) or len(names) != len(set(names)):
            errors.append(f"zip names differ from the allowlist: {sorted(set(names) ^ allowed)}")
        src_of = dict(entries)
        for i in infos:
            if i.filename.startswith("/") or ".." in i.filename.split("/") or "__pycache__" in i.filename:
                errors.append(f"unsafe zip entry {i.filename}")
            elif i.filename in src_of and i.file_size != src_of[i.filename].stat().st_size:
                errors.append(f"size mismatch in zip for {i.filename}")
        rep["listing"] = [(i.filename, i.file_size, i.compress_size) for i in infos]
    rep["zip_bytes"] = zip_path.stat().st_size
    return _done(rep, out)


def _done(rep, out):
    rep["ok"] = not rep["errors"]
    w = out.write
    for n in rep["notes"]:
        w(f"  note: {n}\n")
    if rep.get("listing"):
        w(f"\n{'size':>15} {'compressed':>15}  name   ({rep['zip']})\n")
        for name, size, csize in rep["listing"]:
            w(f"{size:>15,} {csize:>15,}  {name}\n")
        w(f"{sum(s for _, s, _ in rep['listing']):>15,} {rep['zip_bytes']:>15,}  TOTAL ({len(rep['listing'])} files, "
          f"{rep.get('zip_seconds', 0)} s)\n")
    elif rep["entries"]:
        w("\nwould zip (dry run or gate failure):\n")
        for rel, size in rep["entries"]:
            w(f"{size:>15,}  {rel}\n")
    for x in rep["warnings"]:
        w(f"WARNING: {x}\n")
    for x in rep["errors"]:
        w(f"ERROR: {x}\n")
    w(("OK -- " + (f"wrote {rep['zip']}" if rep.get("listing") else "gates passed (dry run)")) if rep["ok"]
      else f"FAILED -- {len(rep['errors'])} error(s); no zip written\n")
    w("\n")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the challenge submission zip from an allowlist")
    ap.add_argument("--project-root", default=os.getcwd())
    ap.add_argument("--zip", default=None, help="zip path (default <root>/submission/<Team>_submission.zip)")
    ap.add_argument("--team", default="Master Bolt")
    ap.add_argument("--extra", action="append", default=[], help="extra file under code/business_entity_resolution/")
    ap.add_argument("--allow-binary", action="store_true")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--allow-placeholders", action="store_true")
    ap.add_argument("--check-test-dir", default=None)
    ap.add_argument("--check-ids", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--max-extra-mb", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rep = build_package(a.project_root, a.zip, a.team, a.extra, a.allow_binary, a.allow_stale,
                        a.allow_placeholders, a.check_test_dir, a.skip_check, a.check_ids, a.dry_run,
                        a.max_extra_mb, a.tmp_dir)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
