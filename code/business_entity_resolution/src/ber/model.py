# -*- coding: utf-8 -*-
"""model.py -- LightGBM pairwise matcher with GroupKFold-by-S1 OOF, calibration, save/load and LOCO evaluation
(Amazon ML Challenge 2026, Business Entity Resolution, team Master Bolt).
Copy into code/business_entity_resolution/src/ber/model.py.   LightGBM is MIT-licensed (the final model).

Contract
--------
train_matcher(feats, cfg=None) -> (bundle, oof)
    feats : s1, cand, label + FEATURE_COLUMNS (from features.compute_features on TRAIN candidates)
    bundle: dict(boosters=[final booster], feature_columns, params, best_iters, calibrator (x, y) knots, cfg, report)
    oof   : s1, cand, label, p_raw, p   (p = calibrated OOF probability; every pair predicted by a model that
            never saw its S1 -> honest input for decide.py tuning)
predict_matcher(bundle, feats) -> pl.DataFrame  s1, cand, p
save_bundle(bundle, out_dir) / load_bundle(out_dir)     model_final.txt + bundle.json (human-readable)
loco_eval(feats_by_country, cfg) -> dict                 train on one country, evaluate on another (France proxy)

Protocol
--------
* Folds = metric.fold_of(s1 id) (deterministic hash, same as io.make_folds) -> all pairs of an S1 share a fold,
  so no S1 is seen in training and validation (a random PAIR split leaks through the other candidates of the S1
  and through the competition/group features).
* Negatives = every non-linked candidate from blocking (hard negatives). Optional negative subsampling
  (cfg.neg_frac) with weight 1/neg_frac keeps the probabilities calibrated in expectation.
* Early stopping on an inner 10% S1-group split of each training fold; the final test model is trained on ALL
  folds with n_rounds = 1.1 x mean(best_iter).
* Isotonic calibration fitted on the OOF raw scores (monotone -> ranking unchanged), applied to test.
* The country value is never a feature; LOCO measures how well the model transfers to an unseen country.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

try:
    import lightgbm as lgb
except ImportError as e:            # pragma: no cover
    raise ImportError("pip install lightgbm==4.7.0 (MIT)") from e

try:
    from . import metric as _M                        # type: ignore
    from .features import FEATURE_COLUMNS as _FC      # type: ignore
except ImportError:
    import importlib.util as _ilu
    _here = os.path.dirname(os.path.abspath(__file__))

    def _load(name, rel):
        for p in (os.path.join(_here, f"{name}.py"), os.path.join(_here, "..", rel, f"{name}.py")):
            if os.path.exists(p):
                import sys as _sys
                if name in _sys.modules:
                    return _sys.modules[name]
                spec = _ilu.spec_from_file_location(name, p)
                mod = _ilu.module_from_spec(spec)
                _sys.modules[name] = mod          # dataclasses look the module up while executing it
                spec.loader.exec_module(mod)  # type: ignore
                return mod
        raise ImportError(name)
    _M = _load("metric", "er-f05-decisions")
    _FC = _load("features", "er-pair-features").FEATURE_COLUMNS


DEFAULT_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 255, "verbose": -1, "num_threads": 0, "seed": 0, "deterministic": True, "force_row_wise": True,
}


@dataclass
class ModelConfig:
    n_folds: int = 5
    fold_seed: int = 0
    params: Dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    max_rounds: int = 3000
    early_stopping: int = 100
    inner_frac: float = 0.10          # share of training-fold S1 used for early stopping
    neg_frac: Optional[float] = None  # e.g. 0.3 -> keep 30% of negatives (weight 1/0.3); None = all
    calibrate: bool = True
    feature_columns: Optional[List[str]] = None
    verbose: bool = True


def _log(cfg, *a):
    if cfg.verbose:
        print(time.strftime("%H:%M:%S"), "[model]", *a, flush=True)


def _cols(cfg: ModelConfig) -> List[str]:
    return list(cfg.feature_columns or _FC)


def _xy(df: pl.DataFrame, cols: List[str]):
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    y = df["label"].cast(pl.Int8).to_numpy() if "label" in df.columns else None
    return X, y


def _subsample(df: pl.DataFrame, frac: Optional[float], seed: int) -> Tuple[pl.DataFrame, np.ndarray]:
    if not frac or frac >= 1:
        return df, np.ones(df.height, dtype=np.float32)
    rng = np.random.default_rng(seed)
    keep = (df["label"].to_numpy() == 1) | (rng.random(df.height) < frac)
    sub = df.filter(pl.Series(keep))
    w = np.where(sub["label"].to_numpy() == 1, 1.0, 1.0 / frac).astype(np.float32)
    return sub, w


def _fit(train: pl.DataFrame, cols, cfg: ModelConfig, seed: int, rounds: Optional[int] = None):
    """Fit one booster. rounds=None -> early stopping on an inner S1-group split."""
    params = dict(cfg.params, seed=seed)
    if rounds is not None:
        sub, w = _subsample(train, cfg.neg_frac, seed)
        X, y = _xy(sub, cols)
        b = lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=cols, free_raw_data=True), num_boost_round=rounds)
        return b, rounds
    inner = _M.fold_of(train["s1"], n_folds=max(2, int(round(1 / cfg.inner_frac))), seed=seed + 101) == 0
    tr, va = train.filter(pl.Series(~inner)), train.filter(pl.Series(inner))
    tr, w = _subsample(tr, cfg.neg_frac, seed)
    Xtr, ytr = _xy(tr, cols)
    Xva, yva = _xy(va, cols)
    dtr = lgb.Dataset(Xtr, ytr, weight=w, feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=True)
    b = lgb.train(params, dtr, num_boost_round=cfg.max_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
    return b, int(b.best_iteration or cfg.max_rounds)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def train_matcher(feats: pl.DataFrame, cfg: Optional[ModelConfig] = None):
    cfg = cfg or ModelConfig()
    cols = _cols(cfg)
    t0 = time.time()
    fold = _M.fold_of(feats["s1"], n_folds=cfg.n_folds, seed=cfg.fold_seed)
    p_raw = np.zeros(feats.height, dtype=np.float64)
    best_iters = []
    for k in range(cfg.n_folds):
        m = fold == k
        tr, va = feats.filter(pl.Series(~m)), feats.filter(pl.Series(m))
        b, it = _fit(tr, cols, cfg, seed=cfg.params.get("seed", 0) + k)
        p_raw[m] = b.predict(_xy(va, cols)[0], num_iteration=it)
        best_iters.append(it)
        _log(cfg, f"fold {k}: best_iter {it}, AUC {_auc(va['label'].to_numpy(), p_raw[m]):.5f} ({time.time() - t0:.0f}s)")
    y = feats["label"].to_numpy()
    calib = None
    p = p_raw
    if cfg.calibrate:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
        calib = (iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())
        p = np.interp(p_raw, calib[0], calib[1])
    n_final = int(round(1.1 * float(np.mean(best_iters))))
    final, _ = _fit(feats, cols, cfg, seed=cfg.params.get("seed", 0) + 99, rounds=n_final)
    gain = dict(zip(cols, final.feature_importance("gain").tolist()))
    report = {"oof_auc": _auc(y, p_raw), "n_pairs": feats.height, "pos_rate": float(y.mean()),
              "best_iters": best_iters, "final_rounds": n_final, "seconds": round(time.time() - t0, 1),
              "top_gain": sorted(gain.items(), key=lambda kv: -kv[1])[:25]}
    _log(cfg, f"OOF AUC {report['oof_auc']:.5f}; final model {n_final} rounds ({report['seconds']}s)")
    bundle = {"boosters": [final], "feature_columns": cols, "params": cfg.params, "best_iters": best_iters,
              "calibrator": calib, "cfg": {k: v for k, v in asdict(cfg).items() if k != "params"}, "report": report}
    oof = feats.select("s1", "cand", "label").with_columns(pl.Series("p_raw", p_raw.astype(np.float32)),
                                                           pl.Series("p", np.asarray(p, dtype=np.float32)))
    return bundle, oof


def predict_matcher(bundle, feats: pl.DataFrame) -> pl.DataFrame:
    cols = bundle["feature_columns"]
    X = _xy(feats, cols)[0]
    raw = np.mean([b.predict(X) for b in bundle["boosters"]], axis=0)
    cal = bundle.get("calibrator")
    p = np.interp(raw, cal[0], cal[1]) if cal else raw
    return feats.select("s1", "cand").with_columns(pl.Series("p", np.asarray(p, dtype=np.float32)))


def save_bundle(bundle, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, b in enumerate(bundle["boosters"]):
        b.save_model(os.path.join(out_dir, f"model_final_{i}.txt"))
    meta = {k: v for k, v in bundle.items() if k != "boosters"}
    meta["n_boosters"] = len(bundle["boosters"])
    with open(os.path.join(out_dir, "bundle.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)


def load_bundle(out_dir: str):
    with open(os.path.join(out_dir, "bundle.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["boosters"] = [lgb.Booster(model_file=os.path.join(out_dir, f"model_final_{i}.txt"))
                        for i in range(meta.pop("n_boosters"))]
    return meta


def loco_eval(feats_by_country: Dict[str, pl.DataFrame], cfg: Optional[ModelConfig] = None) -> Dict:
    """Leave-one-country-out: for every ordered pair (A -> B) train on A, predict B. Returns AUC per direction
    plus the predictions (s1, cand, label, p) so decide.py / metric.py can score macro F0.5 on B."""
    cfg = cfg or ModelConfig()
    cols = _cols(cfg)
    out = {}
    names = list(feats_by_country)
    for a in names:
        b_model, it = _fit(feats_by_country[a], cols, cfg, seed=cfg.params.get("seed", 0))
        for b in names:
            if a == b:
                continue
            fb = feats_by_country[b]
            p = b_model.predict(_xy(fb, cols)[0], num_iteration=it)
            out[f"{a}->{b}"] = {"auc": _auc(fb["label"].to_numpy(), p),
                                "pred": fb.select("s1", "cand", "label").with_columns(pl.Series("p", p.astype(np.float32)))}
            _log(cfg, f"LOCO {a}->{b}: AUC {out[f'{a}->{b}']['auc']:.5f}")
    return out
