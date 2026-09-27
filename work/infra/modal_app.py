# -*- coding: utf-8 -*-
"""modal_app.py -- run Master Bolt pipeline commands on a Modal CPU container (infra only; not part of the zip).

The container mirrors the project tree at /root/proj:
    code/business_entity_resolution/   (baked into the image from the laptop copy)
    .claude/skills/                    (dev tools: make_dev_slice, measure_blocking, eval_slice, ...)
    student_resource/utils/validate_submission.py
    student_resource/dataset -> /vol/dataset   (Modal Volume "mbolt-data", uploaded once)
    work   -> /vol/work                         (checkpoints persist on the volume)
    output -> /vol/output
Each shell command runs from /root/proj; the volume is committed after every command, so a failure loses at most
the running command.

    modal run --detach work/infra/modal_app.py --cmds "cmd1 ;; cmd2" --cpu 16 --mem-gb 128 --tag prepare
Logs: `modal app logs <app-id>` (live) and /vol/work/logs/modal_<tag>.log (after each command).
"""
import os
import subprocess
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
REMOTE = "/root/proj"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1")
    .pip_install("polars==1.44.2", "pyarrow==25.0.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.2",
                 "rapidfuzz==3.14.5", "lightgbm==4.7.0", "psutil==7.2.2", "sparse-dot-topn==1.2.0", "pytest==9.1.1")
    .add_local_dir(os.path.join(PROJECT, "code", "business_entity_resolution"),
                   f"{REMOTE}/code/business_entity_resolution", ignore=["**/__pycache__/**"])
    .add_local_dir(os.path.join(PROJECT, ".claude", "skills"), f"{REMOTE}/.claude/skills", ignore=["**/__pycache__/**"])
    .add_local_file(os.path.join(PROJECT, "student_resource", "utils", "validate_submission.py"),
                    f"{REMOTE}/student_resource/utils/validate_submission.py")
    .add_local_dir(HERE, f"{REMOTE}/infra", ignore=["**/__pycache__/**", "jobs/**"])
)
vol = modal.Volume.from_name("mbolt-data")
app = modal.App("mbolt-er", image=image)


def _link(src: str, dst: str) -> None:
    os.makedirs(src, exist_ok=True)
    if not os.path.lexists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)


@app.function(volumes={"/vol": vol}, cpu=8, memory=32768, timeout=24 * 3600)
def run_cmds(cmds: list, tag: str, threads: int) -> list:
    _link("/vol/dataset", f"{REMOTE}/student_resource/dataset")
    _link("/vol/work", f"{REMOTE}/work")
    _link("/vol/output", f"{REMOTE}/output")
    os.makedirs("/vol/work/logs", exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=str(threads), POLARS_MAX_THREADS=str(threads), BER_MIN_FREE_GB="2")
    results = []
    log_path = f"/vol/work/logs/modal_{tag}.log"
    for c in cmds:
        t = time.time()
        head = f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {c}\n"
        print(head, flush=True)
        p = subprocess.Popen(c, shell=True, cwd=REMOTE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(head)
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
            rc = p.wait()
            tail = f"===== exit {rc} after {time.time() - t:.0f}s\n"
            f.write(tail)
        print(tail, flush=True)
        vol.commit()
        results.append({"cmd": c, "rc": rc, "seconds": round(time.time() - t)})
        if rc != 0:
            break
    return results


@app.local_entrypoint()
def main(cmds: str, tag: str = "job", cpu: float = 8, mem_gb: int = 32):
    """--cmds "a ;; b" or --cmds @file (one command per line, '#' comments) -- the file form avoids PowerShell
    quoting problems."""
    if cmds.startswith("@"):
        with open(cmds[1:], encoding="utf-8-sig") as f:
            lst = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        lst = [c.strip() for c in cmds.split(";;") if c.strip()]
    fn = run_cmds.with_options(cpu=cpu, memory=int(mem_gb * 1024))
    res = fn.remote(lst, tag, max(1, int(cpu)))
    for r in res:
        print(r)
