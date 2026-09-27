# -*- coding: utf-8 -*-
"""watch_logs.py -- print new lines matching a regex from several local modal logs (UTF-8 or UTF-16), exit when
every log shows the Modal app finished (or on first failure if --stop-on-fail).
    python work/infra/watch_logs.py --pattern "block (train|test)/" work/logs/a.log work/logs/b.log
"""
import argparse
import re
import sys
import time

DONE = re.compile(r"App completed|Stopping app|App stopped|Error: |Traceback|===== exit [1-9]")
FAIL = re.compile(r"Traceback|===== exit [1-9]|Killed|MemoryError|OOM|Error: ")


def read(path):
    try:
        raw = open(path, "rb").read()
    except OSError:
        return []
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        txt = raw.decode("utf-16", errors="replace")
    else:
        txt = raw.decode("utf-8", errors="replace")
    return txt.splitlines()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--poll", type=float, default=10)
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    pat = re.compile(a.pattern)
    seen = {p: len(read(p)) for p in a.logs}
    done = set()
    while len(done) < len(a.logs):
        for p in a.logs:
            lines = read(p)
            for ln in lines[seen[p]:]:
                if pat.search(ln) or FAIL.search(ln):
                    tag = p.replace("\\", "/").split("/")[-1].replace("modal_", "").replace("_local.log", "")
                    print(f"[{tag}] {ln.strip()}", flush=True)
                if DONE.search(ln):
                    done.add(p)
            seen[p] = len(lines)
        time.sleep(a.poll)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
