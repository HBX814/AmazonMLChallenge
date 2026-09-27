# -*- coding: utf-8 -*-
"""run_pipeline.py -- single entry point of the Master Bolt ER pipeline. Copy to code/business_entity_resolution/src/.

    python src/run_pipeline.py --data-dir ../../student_resource/dataset --work-dir ../../work --out-dir ../../output --stage all

Stages (each writes parquet under --work-dir and is skipped when its outputs exist, unless --force):
  prepare   TSV -> parquet cache; native-token dictionary learned from TRAIN links only; normalized frames per split/country
  block     candidates per split/country (same BlockingConfig for train and test); train candidates get `label`
  features  pairwise/context/competition/group features per split/country
  train     LightGBM GroupKFold-by-S1 OOF + calibration; decision parameters chosen on OOF macro F0.5 (all train S1)
  predict   test scoring + exclusivity-aware per-S1 decisions
  write     output/candidate_pairs.tsv (exact scored pairs) + output/matching_results.tsv (subset), test_source1 order
Countries are whatever labels exist in the data (France appears only in test) -- never a hard-coded list.
Validation uses the FULL same-country pool for every held-out S1. Logs: <work>/logs/pipeline.log (timings, RSS).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import polars as pl

from ber import io as bio
from ber import normalize_frame as NF
from ber import blocking as BL
from ber import features as FE
from ber import model as MD
from ber import decide as D
from ber import metric as M
from ber import outputs as O
from ber import stage2 as S2
from ber import adapt as AD
from ber.config import PipelineConfig

STAGES = ["prepare", "prio", "block", "features", "train", "predict", "write"]


# ------------------------------------------------------------------------------------------------ utilities
class Log:
    def __init__(self, work: Path):
        (work / "logs").mkdir(parents=True, exist_ok=True)
        self.path = work / "logs" / "pipeline.log"

    def __call__(self, *a):
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / 2**30
        except Exception:
            rss = float("nan")
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [rss {rss:.2f}G] " + " ".join(str(x) for x in a)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _safe(c: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (c or "EMPTY"))


_ONLY = {"splits": None, "countries": None}      # --splits / --countries: run a shard of a stage (parallel machines)


def _countries(work: Path, split: str, only: bool = True) -> list:
    cs = sorted(pl.scan_parquet(str(bio.cache_paths(work, split)["s1"])).select("country").unique().collect()["country"].to_list())
    if only and _ONLY["countries"]:
        cs = [c for c in cs if c in _ONLY["countries"]]
    return cs


def _splits() -> list:
    return [s for s in ("train", "test") if not _ONLY["splits"] or s in _ONLY["splits"]]


def _train_sample(s1_ids: pl.Series, frac, seed: int) -> pl.Series:
    """Boolean mask: deterministic, version-stable hash sample of S1 groups for model training (all pairs of an
    S1 stay together). frac None/>=1 -> everything."""
    if not frac or frac >= 1:
        return pl.Series([True] * len(s1_ids))
    buckets = M.fold_of(s1_ids, n_folds=1000, seed=seed + 7919)
    return pl.Series(buckets < int(round(frac * 1000)))


def _train_drop(s1_ids: pl.Series, cfg) -> pl.Series:
    """Boolean mask of train S1 REMOVED before blocking (cfg.train_drop_s1_frac): their S2/S3 copies stay in the pool
    as ownerless distractors, so the train pool density can be raised to the (label-free) test density. Uses the top
    hash buckets, disjoint from the matcher sample (low buckets) and the prio exclusion range."""
    frac = cfg.train_drop_s1_frac or 0.0
    if frac <= 0:
        return pl.Series([False] * len(s1_ids))
    lo = 1000 - int(round(frac * 1000))
    used = max(cfg.train_s1_frac or 0.0, cfg.prio_exclude_frac or 0.0) * 1000
    if lo < used:
        raise ValueError(f"train_drop_s1_frac={frac} overlaps the training / prio-exclusion hash buckets (< {used:.0f})")
    return pl.Series(M.fold_of(s1_ids, n_folds=1000, seed=cfg.seed + 7919) >= lo)


def _read_s1(work: Path, split: str, c: str, cfg) -> pl.DataFrame:
    s1 = pl.read_parquet(work / "norm" / f"{split}_{_safe(c)}_s1.parquet")
    if split == "train" and cfg.train_drop_s1_frac:
        s1 = s1.filter(~_train_drop(s1["entity_id"], cfg))
    return s1


def _sample_s1(s1: pl.DataFrame, n) -> pl.DataFrame:
    if not n or s1.height <= n:
        return s1
    return s1.with_columns(pl.col("entity_id").hash(seed=13).alias("_h")).sort("_h").head(n).drop("_h")


def _pool(lz: dict, country: str) -> pl.DataFrame:
    return pl.concat([lz["s2"].filter(pl.col("country") == country).collect(),
                      lz["s3"].filter(pl.col("country") == country).collect()])


# ------------------------------------------------------------------------------------------------ stages
def stage_prepare(a, cfg, log):
    work = Path(a.work_dir)
    bio.to_parquet_cache(a.data_dir, work)
    dict_path = work / "native_token_dict.tsv"
    lz = bio.scan_split(work, "train")
    if not dict_path.exists() or a.force:
        gt = lz["gt"].collect()
        # learning uses only native-script names; pass the pools lazily (the function filters)
        n = NF.learn_dict_from_train(lz["s1"], pl.concat([lz["s2"], lz["s3"]]), gt, str(dict_path))
        log("native dict entries:", n)
    NF.use_native_dict(str(dict_path))
    for split in _splits():
        lzs = bio.scan_split(work, split)
        for c in _countries(work, split):
            out_s1 = work / "norm" / f"{split}_{_safe(c)}_s1.parquet"
            out_pool = work / "norm" / f"{split}_{_safe(c)}_pool.parquet"
            if out_s1.exists() and out_pool.exists() and not a.force:
                continue
            out_s1.parent.mkdir(parents=True, exist_ok=True)
            s1 = _sample_s1(lzs["s1"].filter(pl.col("country") == c).collect(), cfg.max_s1_per_country)
            pool = _pool(lzs, c)
            pool_n = NF.add_normalized_columns(pool, n_jobs=cfg.n_jobs)
            s1_n = NF.add_normalized_columns(s1, n_jobs=cfg.n_jobs)
            s1_n.write_parquet(out_s1)
            pool_n.write_parquet(out_pool)
            log(f"prepare {split}/{c}: s1 {s1_n.height:,} pool {pool_n.height:,}")


def _prio_path(work: Path) -> Path:
    return work / "model" / "prio_model.txt"


def stage_prio(a, cfg, log):
    """Learned candidate priority (cfg.learned_prio): uncapped blocking for the train S1 of a stable hash sample of
    whole geo blocks (realistic in-block density), labelled with TRAIN links; rows of S1 that belong to the main
    model's training sample are excluded, so the OOF stays honest. Fits one pooled model (all train countries)."""
    if not cfg.learned_prio:
        return
    import copy
    import zlib
    work = Path(a.work_dir)
    out = _prio_path(work)
    if out.exists() and not a.force:
        return
    bcfg = copy.deepcopy(cfg.blocking)
    bcfg.max_cands_per_s1, bcfg.prio_model = None, None
    gt = bio.scan_split(work, "train")["gt"].collect()
    parts = []
    for c in _countries(work, "train", only=False):
        s1 = _read_s1(work, "train", c, cfg)
        pool = pl.read_parquet(work / "norm" / f"train_{_safe(c)}_pool.parquet")
        cnt = s1.group_by(bcfg.block_col).len().filter(pl.col(bcfg.block_col) != "")
        order = sorted(cnt.rows(), key=lambda r: zlib.crc32(f"{cfg.seed}|{r[0]}".encode()))
        chosen, tot = [], 0
        for st, n in order:
            if tot >= cfg.prio_block_frac * s1.height:
                break
            chosen.append(st)
            tot += n
        s1s = s1.filter(pl.col(bcfg.block_col).is_in(chosen))
        t = time.time()
        u = BL.generate_candidates(s1s, pool, bcfg)
        u = u.join(gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1, pl.Int8).alias("label")),
                   on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
        excl = cfg.prio_exclude_frac if cfg.prio_exclude_frac is not None else cfg.train_s1_frac
        main = s1s["entity_id"].filter(_train_sample(s1s["entity_id"], excl, cfg.seed))
        u = u.filter(~pl.col("s1").is_in(main.implode()))
        log(f"prio {c}: blocks {chosen} -> {s1s.height:,} S1, {u.height:,} uncapped rows (after excluding "
            f"main-train S1), {int(u['label'].sum()):,} positives in {time.time() - t:.0f}s")
        parts.append(u)
        del s1, pool
    rep = BL.train_prio_model(pl.concat(parts, how="diagonal_relaxed"), bcfg, str(out))
    log(f"prio model: {rep}")


def stage_block(a, cfg, log):
    work = Path(a.work_dir)
    if cfg.learned_prio:
        if not _prio_path(work).exists():
            raise FileNotFoundError(f"learned_prio=True but {_prio_path(work)} is missing: run --stage prio first")
        cfg.blocking.prio_model = str(_prio_path(work))
    for split in _splits():
        gt = bio.scan_split(work, "train")["gt"].collect() if split == "train" else None
        for c in _countries(work, split):
            out = work / "cands" / f"{split}_{_safe(c)}.parquet"
            if out.exists() and not a.force:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            s1 = _read_s1(work, split, c, cfg)
            pool = pl.read_parquet(work / "norm" / f"{split}_{_safe(c)}_pool.parquet")
            t = time.time()
            if cfg.shard_blocking:
                parts = [BL.generate_candidates(sp, pp, cfg.blocking)
                         for _, _, sp, pp in BL.geo_shards(s1, pool, cfg.blocking, cfg.shard_max_pool_rows)]
                cands = pl.concat(parts, how="diagonal_relaxed").unique(["s1", "cand"], keep="first")
            else:
                cands = BL.generate_candidates(s1, pool, cfg.blocking)
            if gt is not None:
                cands = cands.join(gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1, pl.Int8).alias("label")),
                                   on=["s1", "cand"], how="left", maintain_order="left").with_columns(pl.col("label").fill_null(0))
                rec = BL.candidate_recall(cands, gt.join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi"),
                                          s1.select("entity_id", "country"))
                log(f"block {split}/{c}: pair recall {rec['pair_recall']:.4f} cands/S1 {rec['cands_per_s1_mean']:.1f}")
            cands.write_parquet(out)
            log(f"block {split}/{c}: {cands.height:,} pairs in {time.time() - t:.0f}s")


def stage_features(a, cfg, log):
    work = Path(a.work_dir)
    for split in _splits():
        for c in _countries(work, split):
            out = work / "feats" / f"{split}_{_safe(c)}.parquet"
            if out.exists() and not a.force:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cands = pl.read_parquet(work / "cands" / f"{split}_{_safe(c)}.parquet")
            s1 = _read_s1(work, split, c, cfg)          # dropped train S1 absent (name-ambiguity counts too)
            pool = pl.read_parquet(work / "norm" / f"{split}_{_safe(c)}_pool.parquet")
            FE.compute_features(cands, s1, pool, cfg.features).write_parquet(out)
            log(f"features {split}/{c}: done")


def _train_s1_meta(work: Path, cfg) -> pl.DataFrame:
    parts = [pl.read_parquet(work / "norm" / f"train_{_safe(c)}_s1.parquet", columns=["entity_id", "country"])
             for c in _countries(work, "train", only=False)]
    meta = pl.concat(parts).rename({"entity_id": "s1"})
    meta = meta.filter(~_train_drop(meta["s1"], cfg))  # removed S1 are not part of the evaluation population
    return meta.filter(_train_sample(meta["s1"], cfg.train_s1_frac, cfg.seed))


def _decision_grid(scored, gt, ids, cfg, biases, lam, log, tag):
    """expected-F x exclusivity x empty_bias + thresholds on OOF (s1, cand, p) -> (best, rows, best_links)."""
    rows = []
    for ex in cfg.decision.exclusivity_grid:
        for b in biases:
            links = D.select_links(scored, "expected_f", exclusivity=ex, lam_missing=lam, empty_bias=b)
            rows.append({"method": "expected_f", "exclusivity": ex, "empty_bias": b, "lam_missing": lam,
                         "f05": M.macro_f05(links, gt, ids)})
            log(f"{tag}: decision {rows[-1]}")
    for t in cfg.decision.threshold_grid:
        links = D.select_links(scored, "threshold", exclusivity="none", t=t)
        rows.append({"method": "threshold", "exclusivity": "none", "t": t, "f05": M.macro_f05(links, gt, ids)})
    best = max(rows, key=lambda r: r["f05"])
    kw = {k: v for k, v in best.items() if k in ("empty_bias", "lam_missing", "t")}
    best_links = D.select_links(scored, best["method"], exclusivity=best["exclusivity"], **kw)
    return best, rows, best_links


def _adapt_cfg(cfg):
    m = cfg.adapt.method
    classes = (list(AD.HARD_CLASSES) if m.endswith("_hard")
               else list(AD.HARD_CLASSES) + list(AD.SMALL_CLASSES) if m.endswith("_hs") else None)
    return AD.AdaptConfig(method=m.replace("_hard", "").replace("_hs", ""), classes=classes, p_min=cfg.adapt.p_min,
                          max_factor=cfg.adapt.max_factor)


def _tag_by_country(work: Path, scored: pl.DataFrame, meta: pl.DataFrame, split: str, p_min: float) -> pl.DataFrame:
    """scored (s1, cand, p, ...) + country + hn_rel (house-number relation class, ber.adapt)."""
    parts = []
    for c in sorted(meta["country"].unique().to_list()):
        x = scored.join(meta.filter(pl.col("country") == c).select("s1"), on="s1", how="semi").with_columns(
            pl.lit(c).alias("country"))
        parts.append(AD.tag_pairs(x, str(work / "norm" / f"{split}_{_safe(c)}_s1.parquet"),
                                  str(work / "norm" / f"{split}_{_safe(c)}_pool.parquet"), p_min=p_min))
    return pl.concat(parts, how="vertical_relaxed")


def stage_train(a, cfg, log):
    """Stage-1 matcher (+ decision); then, when enabled, the stage-2 collective model (+ decision) and the
    label-free shift-adaptation source priors. Each part is skipped when its outputs exist (unless --force)."""
    work = Path(a.work_dir)
    mdir = work / "model"
    if not (mdir / "decision.json").exists() or a.force:
        _train_stage1(a, cfg, log)
    if cfg.stage2.enabled and (not (mdir / "stage2" / "decision_s2.json").exists() or a.force):
        _train_stage2(a, cfg, log)
    if cfg.adapt.method != "none" and (not (mdir / "adapt_source.json").exists() or a.force):
        _fit_adapt_source(a, cfg, log)


def _train_stage2(a, cfg, log):
    work = Path(a.work_dir)
    mdir, sdir = work / "model", work / "model" / "stage2"
    bundle = MD.load_bundle(str(mdir))
    oof = pl.read_parquet(mdir / "oof.parquet")                         # s1, cand, label, p_raw, p (stage-1 OOF)
    dec1 = json.load(open(mdir / "decision.json", encoding="utf-8"))["best"]
    meta = _train_s1_meta(work, cfg)
    gt = bio.scan_split(work, "train")["gt"].collect().join(meta.select("s1"), on="s1", how="semi")
    ids = meta["s1"]
    lam = float(dec1.get("lam_missing", max(0.0, (gt.height - int(oof["label"].sum())) / max(len(ids), 1))))
    cfg2 = S2.Stage2Config(orig_features=list(bundle["feature_columns"]), floor=cfg.stage2.floor, lam_missing=lam,
                           empty_bias=float(dec1.get("empty_bias", 2.0)), guard=cfg.stage2.guard)
    frames = []
    for c in _countries(work, "train", only=False):
        t = time.time()
        fpath = str(work / "feats" / f"train_{_safe(c)}.parquet")
        pop = meta.filter(pl.col("country") == c).select("s1")
        own = oof.join(pop, on="s1", how="semi").select("s1", "cand", "p", "p_raw", "label")
        # competition needs EVERY claimant of a record: final-model stage-1 p for the train S1 outside the sample
        rest = S2.stage1_predict_stream(bundle, fpath, skip_s1=pop, keep_label=True)
        scored = pl.concat([own, rest.select(own.columns)], how="vertical_relaxed")
        del rest
        fr = S2.build_frame(scored, fpath, str(work / "norm" / f"train_{_safe(c)}_pool.parquet"), cfg2, population=pop)
        log(f"stage2 frame {c}: {scored.height:,} scored pairs (all S1) -> {fr.height:,} re-scored rows in {time.time() - t:.0f}s")
        frames.append(fr)
        del scored
    frame = pl.concat(frames, how="vertical_relaxed")
    del frames
    b2, oof2 = S2.fit(frame, cfg2)
    del frame
    S2.save_bundle(b2, str(sdir))
    merged = S2.merge(oof.select("s1", "cand", "label", "p"), oof2.select("s1", "cand", "p"))
    merged.write_parquet(sdir / "oof_s2.parquet")
    best, rows, best_links = _decision_grid(merged, gt, ids, cfg, cfg.stage2.empty_bias_grid, lam, log, "stage2")
    br = M.f05_breakdown(best_links, gt, meta)
    with open(sdir / "decision_s2.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, "grid": rows, "stage1_best": dec1, "report": b2.get("report"),
                   "n_train_s1": meta.height, "pred_links_per_s1": best_links.height / max(1, len(ids)),
                   "breakdown": str(br)}, f, indent=1, default=str)
    log(f"stage2: best decision {best} (stage-1 {dec1['f05']:.5f})")
    log("stage2 OOF breakdown:\n" + str(br))


def _fit_adapt_source(a, cfg, log):
    """Source priors per (country, house-number relation class) from the final OOF probabilities."""
    work = Path(a.work_dir)
    mdir = work / "model"
    s2 = mdir / "stage2" / "oof_s2.parquet"
    oof = pl.read_parquet(s2 if cfg.stage2.enabled and s2.exists() else mdir / "oof.parquet").select("s1", "cand", "label", "p")
    acfg = _adapt_cfg(cfg)
    tagged = _tag_by_country(work, oof, _train_s1_meta(work, cfg), "train", acfg.p_min)
    src = AD.fit_source(tagged, acfg, country_col="country")
    AD.save_source(src, str(mdir / "adapt_source.json"))
    log(f"adapt: source priors from {tagged.height:,} OOF pairs -> {mdir / 'adapt_source.json'}")


def _train_stage1(a, cfg, log, mdir=None):
    """mdir: output folder (default <work>/model; experiment variants write elsewhere)."""
    work = Path(a.work_dir)
    mdir = Path(mdir) if mdir else work / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    meta = _train_s1_meta(work, cfg)                   # (sampled) train S1 = the OOF / decision-tuning population
    keep = meta["s1"].implode()
    feats = pl.concat([pl.scan_parquet(work / "feats" / f"train_{_safe(c)}.parquet").filter(pl.col("s1").is_in(keep)).collect()
                       for c in _countries(work, "train", only=False)]).sort(["s1", "cand"])   # canonical order
    log(f"train: {meta.height:,} S1 (train_s1_frac={cfg.train_s1_frac}), {feats.height:,} pairs, "
        f"{int(feats['label'].sum()):,} positives")
    cfg.model.n_folds = cfg.n_folds
    bundle, oof = MD.train_matcher(feats, cfg.model)
    del feats
    MD.save_bundle(bundle, str(mdir))
    oof.write_parquet(mdir / "oof.parquet")
    gt = bio.scan_split(work, "train")["gt"].collect().join(meta.select("s1"), on="s1", how="semi")
    ids = meta["s1"]
    lam = max(0.0, (gt.height - int(oof["label"].sum())) / max(len(ids), 1))
    best, rows, best_links = _decision_grid(oof, gt, ids, cfg, cfg.decision.empty_bias_grid, lam, log, "train")
    br = M.f05_breakdown(best_links, gt, meta)
    oracle = M.macro_f05(oof.filter(pl.col("label") == 1).select("s1", pl.col("cand").alias("mid")), gt, ids)
    with open(mdir / "decision.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, "grid": rows, "oof_auc": bundle["report"]["oof_auc"], "oracle_f05": oracle,
                   "n_train_s1": meta.height, "pred_links_per_s1": best_links.height / max(1, len(ids)),
                   "true_links_per_s1": gt.height / max(1, len(ids)), "breakdown": str(br)}, f, indent=1, default=str)
    with open(mdir / "pipeline_config.json", "w", encoding="utf-8") as f:
        f.write(cfg.to_json())
    log(f"train: OOF AUC {bundle['report']['oof_auc']:.5f}; best decision {best}")
    log("OOF breakdown:\n" + str(br))


def stage_predict(a, cfg, log):
    """Test scoring per country: stage-1 matcher -> [stage-2 re-scoring over the whole country frame] ->
    [label-free shift adaptation] -> exclusivity-aware per-S1 decision (parameters chosen on train OOF)."""
    work = Path(a.work_dir)
    mdir = work / "model"
    bundle = MD.load_bundle(str(mdir))
    best = json.load(open(mdir / "decision.json", encoding="utf-8"))["best"]
    b2 = cfg2 = src = None
    if cfg.stage2.enabled:
        b2 = S2.load_bundle(str(mdir / "stage2"))
        cfg2 = S2.stage2_config_from_bundle(b2)
        best = json.load(open(mdir / "stage2" / "decision_s2.json", encoding="utf-8"))["best"]
    if cfg.adapt.method != "none":
        src, acfg = AD.load_source(str(mdir / "adapt_source.json")), _adapt_cfg(cfg)
    kw = {k: v for k, v in best.items() if k in ("empty_bias", "lam_missing", "t")}
    for c in _countries(work, "test"):            # France gets the pooled parameters
        out = work / "pred" / f"test_{_safe(c)}_links.parquet"
        if out.exists() and not a.force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        fpath = work / "feats" / f"test_{_safe(c)}.parquet"
        s1p, poolp = work / "norm" / f"test_{_safe(c)}_s1.parquet", work / "norm" / f"test_{_safe(c)}_pool.parquet"
        n_s1 = pl.scan_parquet(s1p).select(pl.len()).collect().item()
        scored = MD.predict_matcher(bundle, pl.read_parquet(fpath))
        msg = f"stage-1 links-ready ({time.time() - t:.0f}s)"
        if b2 is not None:
            fr = S2.build_frame(scored, str(fpath), str(poolp), cfg2, stage1_bundle=bundle)
            scored = S2.predict(b2, fr, scored=scored).select("s1", "cand", "p")
            msg += f"; stage-2 re-scored {fr.height:,} rows ({time.time() - t:.0f}s)"
            del fr
        tagged = None
        if src is not None:
            tagged = AD.tag_pairs(scored, str(s1p), str(poolp), p_min=acfg.p_min)
            tagged, est = AD.adapt(tagged, src, acfg, country=c, n_s1=n_s1)
            est.write_parquet(work / "pred" / f"test_{_safe(c)}_adapt.parquet")
            scored = tagged.select("s1", "cand", "p")
            msg += f"; adapted {int((est['factor'] != 1.0).sum())} classes"
        links = D.select_links(scored, best["method"], exclusivity=best.get("exclusivity", cfg.decision.exclusivity), **kw)
        if cfg.adapt.rule:
            if tagged is None:
                tagged = AD.tag_pairs(scored, str(s1p), str(poolp), p_min=cfg.adapt.p_min)
            n0 = links.height
            links = AD.targeted_rule(links, tagged.select("s1", "cand", "p", "hn_rel"))
            msg += f"; rule dropped {n0 - links.height:,} HARD links"
        scored.write_parquet(work / "pred" / f"test_{_safe(c)}_scored.parquet")
        links.write_parquet(out)
        log(f"predict test/{c}: {scored.height:,} scored, {links.height:,} links, {links.height / max(n_s1, 1):.2f} "
            f"links/S1, sum p/S1 {float(scored['p'].sum()) / max(n_s1, 1):.3f}; {msg}")


def stage_write(a, cfg, log):
    work = Path(a.work_dir)
    cs = _countries(work, "test", only=False)          # the submission always covers every test country
    scored = pl.concat([pl.read_parquet(work / "pred" / f"test_{_safe(c)}_scored.parquet") for c in cs])
    links = pl.concat([pl.read_parquet(work / "pred" / f"test_{_safe(c)}_links.parquet") for c in cs])
    s1_ids = O.read_s1_order(os.path.join(a.data_dir, "test", "test_source1.tsv"))
    rep = O.write_submission(s1_ids, links, scored, a.out_dir)
    log("write:", rep)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Master Bolt business entity resolution pipeline")
    ap.add_argument("--data-dir", required=True, help="organiser dataset dir with train/ and test/")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--config", default=None, help="JSON overrides for PipelineConfig")
    ap.add_argument("--max-s1", type=int, default=None, help="smoke runs: S1 per country (pools stay full)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute the selected stage(s)")
    ap.add_argument("--splits", default=None, help="comma list (train,test): shard prepare/block/features")
    ap.add_argument("--countries", default=None, help="comma list: shard prepare/block/features/predict by country")
    a = ap.parse_args(argv)
    _ONLY["splits"] = [s for s in a.splits.split(",") if s] if a.splits else None
    _ONLY["countries"] = [c for c in a.countries.split(",") if c] if a.countries else None
    cfg = PipelineConfig.from_json(a.config)
    if a.max_s1:
        cfg.max_s1_per_country = a.max_s1
    if a.n_jobs is not None:
        cfg.n_jobs = a.n_jobs
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    log = Log(work)
    todo = STAGES if a.stage == "all" else [a.stage]
    fns = {"prepare": stage_prepare, "prio": stage_prio, "block": stage_block, "features": stage_features,
           "train": stage_train, "predict": stage_predict, "write": stage_write}
    for s in todo:
        t = time.time()
        log(f"=== stage {s} start")
        fns[s](a, cfg, log)
        log(f"=== stage {s} done in {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
