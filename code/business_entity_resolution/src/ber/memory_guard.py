"""memory_guard.py - RAM budget guard and chunk-size helper for the Master Bolt ER pipeline.

Purpose
    The team laptop has 15.7 GiB RAM but usually only 1-4 GiB free. Once Windows starts paging,
    a stage slows down about 10x (measured 2026-09-25: hash-vectorizing the 6.19M-row US train pool
    did its first 4M rows in 13 s, then needed 264 s in total). This module
      * logs RSS / available RAM / pagefile growth per pipeline stage,
      * raises a clear MemoryBudgetError *before* the machine thrashes, and
      * sizes chunks and worker counts from the RAM that is actually free right now.
    The same code runs unchanged on a 128 GiB AWS box: limits come from env vars, not constants.

    Copy to: code/business_entity_resolution/src/ber/memory_guard.py (needs only psutil; polars optional).
    "GiB" below = 2**30 bytes (what Task Manager calls GB).

Public API
    MemoryBudgetError(MemoryError)
    MemorySnapshot                  dataclass: rss_gb, held_gb, avail_gb, total_gb, sys_used_pct, swap_used_gb
    snapshot(include_children=True) -> MemorySnapshot
    MemoryGuard(budget_gb=None, min_free_gb=None, log_path=None, name="ber", include_children=True, sample_s=0.5)
        .check(stage="", need_gb=0.0) -> MemorySnapshot   raises MemoryBudgetError when over budget
        .usable_gb() -> float                              GiB this process may still allocate
        .log(stage, event="mark", **extra) -> MemorySnapshot
        .stage(name)                                       context manager: start/end lines, elapsed, peak, min free
    chunk_rows(bytes_per_row, *, guard=None, fraction=0.5, min_rows=10_000, max_rows=5_000_000, multiple=1_000) -> int
    bytes_per_row(df, overhead=3.0) -> float               polars estimated_size()/height x overhead
    iter_slices(n_rows, chunk) -> Iterator[tuple[int, int]]
    max_workers(per_worker_gb, *, guard=None, cap=None) -> int
    log_memory(stage, **extra) -> MemorySnapshot           uses a process-wide default guard

Environment overrides (read when a MemoryGuard is created)
    BER_MEM_BUDGET_GB   cap on the RSS held by this process tree (unset = no cap; only the free-RAM floor applies)
    BER_MIN_FREE_GB     floor for system available RAM (default 0.75)
    BER_MEM_LOG         file to append log lines to (default: stderr only)

Usage
    from ber.memory_guard import MemoryGuard, bytes_per_row, chunk_rows, iter_slices
    guard = MemoryGuard(log_path="../../work/logs/memory.log")
    with guard.stage("features"):
        step = chunk_rows(bytes_per_row(cands.head(50_000), overhead=6.0), guard=guard)
        for lo, hi in iter_slices(cands.height, step):
            guard.check("features")            # raises before the laptop starts paging
            part = compute_chunk(cands.slice(lo, hi - lo))
            part.write_parquet(f"../../work/features/part_{lo:09d}.parquet")

    CLI (prints a snapshot and suggested sizes; exit code 0):
        python memory_guard.py
        python memory_guard.py --bytes-per-row 400 --per-worker-gb 1.5
        python memory_guard.py --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import os
import sys
import tempfile
import threading
import time
from typing import Iterator, Optional

import psutil

GIB = float(1024 ** 3)
DEFAULT_MIN_FREE_GB = 0.75


class MemoryBudgetError(MemoryError):
    """Raised when continuing would exceed the RSS budget or push free RAM below the floor."""


@dataclasses.dataclass(frozen=True)
class MemorySnapshot:
    rss_gb: float        # resident set / working set of this process tree
    held_gb: float       # max(rss, private bytes): what the tree really holds even if Windows trimmed it
    avail_gb: float      # system-wide available RAM
    total_gb: float
    sys_used_pct: float
    swap_used_gb: float  # pagefile/swap in use (growth during a stage = paging)
    t: float             # time.time()

    def short(self) -> str:
        return (f"rss={self.rss_gb:.2f}GiB held={self.held_gb:.2f}GiB avail={self.avail_gb:.2f}GiB "
                f"of {self.total_gb:.1f}GiB used={self.sys_used_pct:.0f}% swap={self.swap_used_gb:.2f}GiB")


def _proc_mem(p: psutil.Process) -> tuple[int, int]:
    mi = p.memory_info()
    # Windows exposes private bytes; under pressure the working set (rss) is trimmed but private stays.
    return mi.rss, max(mi.rss, getattr(mi, "private", 0))


def snapshot(include_children: bool = True) -> MemorySnapshot:
    """Current memory state of this process (plus spawn/joblib children) and of the system."""
    me = psutil.Process()
    rss, held = _proc_mem(me)
    if include_children:
        for ch in me.children(recursive=True):
            try:
                r, h = _proc_mem(ch)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            rss += r
            held += h
    vm = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory().used
    except Exception:  # some containers/VMs do not expose swap counters
        swap = 0
    return MemorySnapshot(rss / GIB, held / GIB, vm.available / GIB, vm.total / GIB, float(vm.percent),
                          swap / GIB, time.time())


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number (GiB)") from exc


class MemoryGuard:
    """Per-run memory policy: an optional RSS budget plus a system free-RAM floor."""

    def __init__(self, budget_gb: Optional[float] = None, min_free_gb: Optional[float] = None,
                 log_path: Optional[str] = None, name: str = "ber", include_children: bool = True,
                 sample_s: float = 0.5, echo: bool = True) -> None:
        self.budget_gb = budget_gb if budget_gb is not None else _env_float("BER_MEM_BUDGET_GB")
        floor = min_free_gb if min_free_gb is not None else _env_float("BER_MIN_FREE_GB")
        self.min_free_gb = DEFAULT_MIN_FREE_GB if floor is None else floor
        self.log_path = log_path if log_path is not None else (os.environ.get("BER_MEM_LOG") or None)
        self.name = name
        self.include_children = include_children
        self.sample_s = sample_s
        self.echo = echo
        self.last: Optional[MemorySnapshot] = None

    # ------------------------------------------------------------------ checks
    def usable_gb(self, snap: Optional[MemorySnapshot] = None) -> float:
        """GiB this process may still allocate before hitting the floor or the budget (can be <= 0)."""
        s = snap or snapshot(self.include_children)
        room = s.avail_gb - self.min_free_gb
        if self.budget_gb is not None:
            room = min(room, self.budget_gb - s.held_gb)
        return room

    def check(self, stage: str = "", need_gb: float = 0.0) -> MemorySnapshot:
        """Raise MemoryBudgetError if allocating `need_gb` more would breach the budget or the floor."""
        s = snapshot(self.include_children)
        self.last = s
        where = f"stage '{stage}'" if stage else "memory check"
        hint = ("Fix: close other apps (browser, IDE, other python jobs), lower the chunk size, shard by "
                "country x state, or run this stage on AWS (skill er-compute-and-aws).")
        if self.budget_gb is not None and s.held_gb + need_gb > self.budget_gb:
            raise MemoryBudgetError(
                f"[memory_guard] {where}: this process tree holds {s.held_gb:.2f} GiB and needs {need_gb:.2f} GiB "
                f"more, over the budget BER_MEM_BUDGET_GB={self.budget_gb:.2f} GiB. {hint}")
        if s.avail_gb - need_gb < self.min_free_gb:
            raise MemoryBudgetError(
                f"[memory_guard] {where}: system available RAM {s.avail_gb:.2f} GiB (minus {need_gb:.2f} GiB needed) "
                f"would drop below the floor BER_MIN_FREE_GB={self.min_free_gb:.2f} GiB; Windows would start paging "
                f"(~10x slower). This process tree holds {s.held_gb:.2f} GiB. {hint}")
        return s

    # ----------------------------------------------------------------- logging
    def _emit(self, line: str) -> None:
        if self.echo:
            print(line, file=sys.stderr, flush=True)
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")

    def log(self, stage: str, event: str = "mark", **extra: object) -> MemorySnapshot:
        s = snapshot(self.include_children)
        self.last = s
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        tail = "".join(f" {k}={v}" for k, v in extra.items())
        self._emit(f"{ts} {self.name} stage={stage} event={event} {s.short()}{tail}")
        return s

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator["MemoryGuard"]:
        """Log start/end of a stage with elapsed time, peak held RSS, minimum free RAM and pagefile growth."""
        start = self.log(name, "start")
        peak = [start.held_gb]
        low = [start.avail_gb]
        stop = threading.Event()

        def _sample() -> None:
            while not stop.wait(self.sample_s):
                try:
                    s = snapshot(self.include_children)
                except Exception:
                    continue
                peak[0] = max(peak[0], s.held_gb)
                low[0] = min(low[0], s.avail_gb)

        th = threading.Thread(target=_sample, name=f"memguard-{name}", daemon=True)
        th.start()
        t0 = time.perf_counter()
        failed: Optional[BaseException] = None
        try:
            yield self
        except BaseException as exc:
            failed = exc
            raise
        finally:
            stop.set()
            th.join(timeout=5)
            end = snapshot(self.include_children)
            peak[0] = max(peak[0], end.held_gb)
            low[0] = min(low[0], end.avail_gb)
            extra = dict(elapsed_s=f"{time.perf_counter() - t0:.1f}", peak_held=f"{peak[0]:.2f}GiB",
                         min_avail=f"{low[0]:.2f}GiB", swap_delta=f"{end.swap_used_gb - start.swap_used_gb:+.2f}GiB")
            if failed is not None:
                extra["error"] = type(failed).__name__
            if low[0] < self.min_free_gb:
                extra["WARNING"] = "free-RAM-floor-breached(paging-likely)"
            self.log(name, "error" if failed is not None else "end", **extra)


# --------------------------------------------------------------------- sizing
def bytes_per_row(df: object, overhead: float = 3.0) -> float:
    """Bytes per row of a polars frame times `overhead` (temporaries: joins ~2-3x, string ops/features ~4-6x)."""
    height = getattr(df, "height", None)
    if height is None:
        height = len(df)  # type: ignore[arg-type]
    size = df.estimated_size() if hasattr(df, "estimated_size") else sys.getsizeof(df)  # type: ignore[attr-defined]
    return max(1.0, float(size) / max(1, int(height)) * overhead)


def chunk_rows(bytes_per_row: float, *, guard: Optional[MemoryGuard] = None, fraction: float = 0.5,
               min_rows: int = 10_000, max_rows: int = 5_000_000, multiple: int = 1_000) -> int:
    """Rows per chunk so one chunk uses at most `fraction` of the RAM that is usable right now.

    Raises MemoryBudgetError if not even `min_rows` rows fit (better than a 10x-slower paging run).
    """
    if bytes_per_row <= 0:
        raise ValueError("bytes_per_row must be > 0")
    g = guard or MemoryGuard(echo=False)
    usable = g.usable_gb()
    fit = int(max(0.0, usable) * GIB * fraction / bytes_per_row)
    if fit < min_rows:
        raise MemoryBudgetError(
            f"[memory_guard] only {usable:.2f} GiB usable (floor {g.min_free_gb:.2f} GiB): {min_rows:,} rows x "
            f"{bytes_per_row:,.0f} B/row do not fit in {fraction:.0%} of it. Free RAM, lower the per-row cost, "
            f"or run on AWS (er-compute-and-aws).")
    rows = min(fit, max_rows)
    if multiple > 1 and rows >= multiple:
        rows -= rows % multiple
    return max(min_rows, rows)


def iter_slices(n_rows: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Yield (lo, hi) half-open slices covering range(n_rows)."""
    if chunk <= 0:
        raise ValueError("chunk must be > 0")
    for lo in range(0, max(0, n_rows), chunk):
        yield lo, min(n_rows, lo + chunk)


def max_workers(per_worker_gb: float, *, guard: Optional[MemoryGuard] = None, cap: Optional[int] = None) -> int:
    """Worker processes that fit in usable RAM. On Windows every spawn worker re-imports polars/numpy
    (~0.15-0.3 GiB) before it holds any data, so include that in `per_worker_gb`."""
    if per_worker_gb <= 0:
        raise ValueError("per_worker_gb must be > 0")
    g = guard or MemoryGuard(echo=False)
    limit = cap if cap is not None else (os.cpu_count() or 1)
    return max(1, min(limit, int(max(0.0, g.usable_gb()) // per_worker_gb)))


_DEFAULT: Optional[MemoryGuard] = None


def log_memory(stage: str, **extra: object) -> MemorySnapshot:
    """One-liner for ad-hoc logging with a process-wide default guard (env-configured)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MemoryGuard()
    return _DEFAULT.log(stage, **extra)


# ------------------------------------------------------------------------ CLI
def _selftest() -> int:
    s = snapshot()
    assert s.total_gb > 0 and 0 <= s.avail_gb <= s.total_gb and s.held_gb >= s.rss_gb * 0.5
    roomy = MemoryGuard(budget_gb=10_000, min_free_gb=0.0, echo=False)
    roomy.check("selftest")
    try:
        MemoryGuard(min_free_gb=s.total_gb + 1, echo=False).check("selftest", need_gb=0.0)
        raise AssertionError("floor above total RAM must raise")
    except MemoryBudgetError as exc:
        assert "BER_MIN_FREE_GB" in str(exc)
    try:
        MemoryGuard(budget_gb=0.001, min_free_gb=0.0, echo=False).check("selftest")
        raise AssertionError("tiny budget must raise")
    except MemoryBudgetError as exc:
        assert "BER_MEM_BUDGET_GB" in str(exc)
    small = chunk_rows(4_000, guard=roomy, min_rows=1, max_rows=10**12, multiple=1)
    big = chunk_rows(400, guard=roomy, min_rows=1, max_rows=10**12, multiple=1)
    assert big >= small, (small, big)
    assert [x for x in iter_slices(10, 4)] == [(0, 4), (4, 8), (8, 10)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "logs", "memory.log")
        g = MemoryGuard(budget_gb=10_000, min_free_gb=0.0, log_path=path, echo=False, sample_s=0.05)
        with g.stage("selftest"):
            blob = bytearray(64 * 1024 * 1024)
            time.sleep(0.2)
            del blob
        lines = open(path, encoding="utf-8").read().splitlines()
        assert len(lines) == 2 and "event=start" in lines[0] and "event=end" in lines[1] and "peak_held=" in lines[1]
    print("memory_guard selftest OK:", s.short())
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Memory snapshot + chunk/worker sizing for the ER pipeline.")
    ap.add_argument("--bytes-per-row", type=float, default=None, help="per-row working-set estimate (bytes)")
    ap.add_argument("--per-worker-gb", type=float, default=None, help="RAM per spawned worker (GiB)")
    ap.add_argument("--fraction", type=float, default=0.5, help="share of usable RAM one chunk may take")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    g = MemoryGuard(echo=False)
    s = snapshot()
    print(f"snapshot : {s.short()}")
    print(f"policy   : budget={g.budget_gb if g.budget_gb is not None else 'none'} GiB, "
          f"floor={g.min_free_gb:.2f} GiB -> usable now {g.usable_gb(s):.2f} GiB, cpus={os.cpu_count()}")
    if s.avail_gb < 3.0:
        print("advice   : < 3 GiB free - close apps; only dev slices / one country x state shard fit; "
              "full-scale stages belong on AWS (er-compute-and-aws).")
    if a.bytes_per_row:
        try:
            print(f"chunk    : {chunk_rows(a.bytes_per_row, guard=g, fraction=a.fraction):,} rows "
                  f"at {a.bytes_per_row:,.0f} B/row")
        except MemoryBudgetError as exc:
            print(f"chunk    : {exc}")
    if a.per_worker_gb:
        print(f"workers  : {max_workers(a.per_worker_gb, guard=g)} at {a.per_worker_gb} GiB each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
