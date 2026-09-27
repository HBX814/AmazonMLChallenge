# -*- coding: utf-8 -*-
"""modal_gpu.py -- GPU variant of modal_app.py (same volume / layout / job-file protocol) for the cross-encoder.
    modal run --detach work/infra/modal_gpu.py --cmds @work/infra/jobs/<job>.txt --tag <tag> [--gpu L4]
Hugging Face weights are cached on the volume (/vol/hf) so a model is downloaded once.
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
    .pip_install("torch==2.5.1", "transformers==4.46.3", "tokenizers==0.20.3", "safetensors==0.4.5",
                 "polars==1.44.2", "pyarrow==25.0.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.2",
                 "rapidfuzz==3.14.5", "lightgbm==4.7.0", "psutil==7.2.2", "sparse-dot-topn==1.2.0")
    .add_local_dir(os.path.join(PROJECT, "code", "business_entity_resolution"),
                   f"{REMOTE}/code/business_entity_resolution", ignore=["**/__pycache__/**"])
    .add_local_dir(HERE, f"{REMOTE}/infra", ignore=["**/__pycache__/**", "jobs/**"])
)
vol = modal.Volume.from_name("mbolt-data")
app = modal.App("mbolt-gpu", image=image)


def _link(src: str, dst: str) -> None:
    os.makedirs(src, exist_ok=True)
    if not os.path.lexists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)


@app.function(volumes={"/vol": vol}, gpu="L4", cpu=8, memory=65536, timeout=12 * 3600)
def run_gpu(cmds: list, tag: str) -> list:
    _link("/vol/dataset", f"{REMOTE}/student_resource/dataset")
    _link("/vol/work", f"{REMOTE}/work")
    os.makedirs("/vol/work/logs", exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="8", POLARS_MAX_THREADS="8", HF_HOME="/vol/hf", TOKENIZERS_PARALLELISM="true")
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
def main(cmds: str, tag: str = "gpu", gpu: str = "L4"):
    if cmds.startswith("@"):
        with open(cmds[1:], encoding="utf-8-sig") as f:
            lst = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        lst = [c.strip() for c in cmds.split(";;") if c.strip()]
    fn = run_gpu.with_options(gpu=gpu) if gpu != "L4" else run_gpu
    for r in fn.remote(lst, tag):
        print(r)
