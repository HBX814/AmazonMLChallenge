# -*- coding: utf-8 -*-
"""config.py -- one place for every pipeline setting (Master Bolt ER). Copy to src/ber/config.py.

PipelineConfig bundles the stage configs so train and test ALWAYS use identical blocking / features / model
settings, and it is saved next to the model (work/model/pipeline_config.json) for the audit trail.
Override from JSON:  python src/run_pipeline.py ... --config my_config.json   (only the keys you set change)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional

try:
    from .blocking import BlockingConfig
    from .features import FeatureConfig
    from .model import ModelConfig
except ImportError:                                    # flat layout (smoke test copies everything into ber/)
    from blocking import BlockingConfig                # type: ignore
    from features import FeatureConfig                 # type: ignore
    from model import ModelConfig                      # type: ignore


@dataclass
class DecisionConfig:
    methods: List[str] = field(default_factory=lambda: ["expected_f", "threshold"])
    empty_bias_grid: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    threshold_grid: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8])
    exclusivity_grid: List[str] = field(default_factory=lambda: ["none", "soft"])   # tuned on OOF with the bias
    exclusivity: str = "none"          # fallback only (the tuned choice is stored in decision.json); repair still
                                       # guarantees <=1 S1 per record


@dataclass
class Stage2Settings:
    enabled: bool = False              # stage-2 collective re-scoring (ber/stage2.py) after the stage-1 matcher
    guard: str = "G1"                  # house-number-mismatch pairs may only go DOWN (test US hard negatives)
    floor: float = 1e-3                # pairs with stage-1 p below this keep p1
    empty_bias_grid: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0])


@dataclass
class AdaptSettings:
    method: str = "none"               # none | em | em_hard | density | density_hard | density_hs (HARD + SMALL
                                       # classes)  (ber/adapt.py, label-free)
    p_min: float = 1e-3
    max_factor: float = 4.0            # cap on the per-class odds factor
    rule: bool = False                 # after the decision: drop HARD-relation links of an S1 that also has an
                                       # exact-house-number link (p < 0.99) -- adapt.targeted_rule


@dataclass
class CrossEncSettings:
    enabled: bool = False              # transformer cross-encoder logits as stage-2 features (ber/crossenc.py)
    band: List[float] = field(default_factory=lambda: [0.01, 0.99])   # stage-1 p band that is scored
    models: List[dict] = field(default_factory=list)   # crossenc.CrossEncModel fields per model (name, model, ...)


@dataclass
class PipelineConfig:
    seed: int = 0
    n_jobs: int = -1
    n_folds: int = 5
    max_s1_per_country: Optional[int] = None     # smoke runs only; pools always stay full
    train_s1_frac: Optional[float] = None        # train stage only: hash sample of train S1 groups used for the
                                                 # model + OOF decision tuning (features keep full competition context)
    learned_prio: bool = False                   # stage `prio`: learned candidate priority before the per-S1 cap
    prio_block_frac: float = 0.2                 # share of each train country's S1 (whole geo blocks) used to fit it
    prio_exclude_frac: Optional[float] = None    # hash sample of S1 excluded from prio training (None = train_s1_frac)
    train_drop_s1_frac: float = 0.0              # train S1 removed before blocking (their copies become ownerless
                                                 # distractors) to match the higher test pool density; top hash buckets
    candidate_prune_p1: Optional[float] = None   # last blocking filter: keep only pairs with stage-1 p >= this for
                                                 # stage 2 / decision / candidate_pairs.tsv (None = keep all)
    shard_blocking: bool = False                 # True on the laptop: geo_shards() one block group at a time
    shard_max_pool_rows: int = 1_500_000
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    stage2: Stage2Settings = field(default_factory=Stage2Settings)
    adapt: AdaptSettings = field(default_factory=AdaptSettings)
    crossenc: CrossEncSettings = field(default_factory=CrossEncSettings)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, default=str)

    @classmethod
    def from_json(cls, path: Optional[str]) -> "PipelineConfig":
        cfg = cls()
        if not path:
            return cfg
        with open(path, encoding="utf-8") as f:
            _update(cfg, json.load(f))
        return cfg


def _update(obj, d: dict) -> None:
    names = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"unknown config key {k!r} for {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        else:
            setattr(obj, k, tuple(v) if isinstance(cur, tuple) else v)
