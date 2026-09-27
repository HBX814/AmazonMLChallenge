# -*- coding: utf-8 -*-
"""smoke_test.py -- assemble the reference modules into a temp `ber` package and run the WHOLE pipeline on a
mini dataset, then both validators. Run after any change to a reference module (and before copying them).

    python .claude/skills/er-challenge-playbook/smoke_test.py                       # builds work/dev/mini if missing
    python .claude/skills/er-challenge-playbook/smoke_test.py --data-dir <mini dir with train/ and test/>
Prints the stage log, OOF macro F0.5 (per country), blocking recall, validator verdicts and a final PASS/FAIL line.
The mini set (default 3,000 S1 per country; train with GT + realistic-density random distractors; test incl.
France) is built from student_resource/dataset with streaming polars scans (low RAM).
Note: random distractors are easier than real neighbours, so mini F0.5 is optimistic -- use geo slices
(er-data-loading make_dev_slice.py) for real numbers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent
PROJECT = SKILLS.parent.parent
MODULES = {                      # module -> skill folder
    "io": "er-data-loading", "normalize": "er-text-normalization", "normalize_frame": "er-text-normalization",
    "blocking": "er-blocking-candidates", "features": "er-pair-features", "model": "er-matcher-training",
    "decide": "er-f05-decisions", "metric": "er-f05-decisions", "outputs": "er-submission-packaging",
    "config": "er-challenge-playbook", "stage2": "er-matcher-training", "adapt": "er-f05-decisions",
}


def _load_io():
    spec = importlib.util.spec_from_file_location("ber_io_ref", SKILLS / "er-data-loading" / "io.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_mini(real: Path, out: Path, n_s1: int = 3000, seed: int = 7) -> None:
    import polars as pl
    bio = _load_io()
    t = time.time()
    for split in ("train", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
        src = {k: bio.scan_source(str(real / split / f"{split}_source{i}.tsv")) for i, k in ((1, "s1"), (2, "s2"), (3, "s3"))}
        s1 = (src["s1"].with_columns(pl.col("entity_id").hash(seed=seed).alias("_h"))
                       .sort("_h").group_by("country", maintain_order=True).head(n_s1).drop("_h").collect())
        bio.write_source_tsv(s1, out / split / f"{split}_source1.tsv")
        if split == "train":
            gt = bio.scan_ground_truth(str(real / "train" / "train_ground_truth.tsv")).collect()
            gt = gt.join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
            bio.write_ground_truth_tsv(gt, s1["entity_id"], out / "train" / "train_ground_truth.tsv")
            linked = set(gt["mid"].to_list())
        for k, i in (("s2", 2), ("s3", 3)):
            lf = src[k].with_columns(pl.col("entity_id").hash(seed=seed + i).alias("_h"))
            if split == "train":
                lk = lf.filter(pl.col("entity_id").is_in(list(linked))).drop("_h").collect()
                n_extra = int(lk.height * 1.22 / 3.46) // max(1, s1["country"].n_unique())
                ex = (lf.filter(~pl.col("entity_id").is_in(list(linked))).sort("_h")
                        .group_by("country", maintain_order=True).head(n_extra).drop("_h").collect())
                df = pl.concat([lk, ex.select(lk.columns)])      # group_by().head() moves the key column first
            else:
                df = lf.sort("_h").group_by("country", maintain_order=True).head(int(n_s1 * 2.9)).drop("_h").collect()
            bio.write_source_tsv(df, out / split / f"{split}_source{i}.tsv")
    print(f"mini dataset built at {out} in {time.time() - t:.0f}s", flush=True)


def assemble(tmp: Path) -> Path:
    pkg = tmp / "src" / "ber"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text('"""Master Bolt business entity resolution package."""\n', encoding="utf-8")
    for mod, folder in MODULES.items():
        shutil.copy2(SKILLS / folder / f"{mod}.py", pkg / f"{mod}.py")
    shutil.copy2(HERE / "run_pipeline.py", tmp / "src" / "run_pipeline.py")
    return tmp / "src" / "run_pipeline.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="mini dataset dir (train/, test/); default work/dev/mini")
    ap.add_argument("--real-data-dir", default=str(PROJECT / "student_resource" / "dataset"))
    ap.add_argument("--n-s1", type=int, default=3000)
    ap.add_argument("--tmp", default=None, help="keep the assembled package + outputs here")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--config", default=None, help="PipelineConfig JSON overrides passed to run_pipeline.py")
    a = ap.parse_args(argv)
    data = Path(a.data_dir) if a.data_dir else PROJECT / "work" / "dev" / "mini"
    if not (data / "train" / "train_ground_truth.tsv").exists():
        build_mini(Path(a.real_data_dir), data, a.n_s1)
    tmp = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix="ber_smoke_"))
    runner = assemble(tmp)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    t = time.time()
    cmd = [sys.executable, str(runner), "--data-dir", str(data), "--work-dir", str(tmp / "work"),
           "--out-dir", str(tmp / "output"), "--stage", "all", "--n-jobs", str(a.n_jobs)]
    if a.config:
        cmd += ["--config", os.path.abspath(a.config)]
    r = subprocess.run(cmd, env=env)
    ok = r.returncode == 0
    print(f"pipeline exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    if ok:
        dec = json.load(open(tmp / "work" / "model" / "decision.json", encoding="utf-8"))
        print("OOF AUC", dec["oof_auc"], "best decision", dec["best"])
        print(dec["breakdown"])
        chk = SKILLS / "er-submission-packaging" / "check_submission.py"
        val = PROJECT / "student_resource" / "utils" / "validate_submission.py"
        cmd = [sys.executable, str(chk), "--matching", str(tmp / "output" / "matching_results.tsv"),
               "--candidate", str(tmp / "output" / "candidate_pairs.tsv"), "--test-dir", str(data / "test"), "--check-ids"]
        if val.exists():
            cmd += ["--official", "--validator", str(val)]
        r2 = subprocess.run(cmd, env=env)
        ok = r2.returncode == 0
    print(f"SMOKE TEST {'PASS' if ok else 'FAIL'}  (package + outputs in {tmp})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
