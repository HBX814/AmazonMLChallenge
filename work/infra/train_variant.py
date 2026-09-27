# -*- coding: utf-8 -*-
"""train_variant.py -- stage-1 training of a feature-set variant on an existing work dir (same S1 sample, same frozen
folds, same decision grid as run_pipeline), written to its own model folder. Used for honest A/B attribution.
    python infra/train_variant.py --work /vol/work_v4 --config code/business_entity_resolution/configs/submission_v4.json \
        --features v3 --out /vol/exp/v4a/model
--features: v3 (the 72 v3 columns) | all (FEATURE_COLUMNS)
"""
import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import run_pipeline as R                      # noqa: E402
from ber import features as F                 # noqa: E402
from ber.config import PipelineConfig         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--features", choices=["v3", "all"], required=True)
ap.add_argument("--out", required=True)
x = ap.parse_args()
cfg = PipelineConfig.from_json(x.config)
cfg.model.feature_columns = list(F.FEATURE_COLUMNS_V3) if x.features == "v3" else list(F.FEATURE_COLUMNS)
Path(x.out).mkdir(parents=True, exist_ok=True)
log = R.Log(Path(x.out).parent)
a = types.SimpleNamespace(work_dir=x.work, force=True)
R._train_stage1(a, cfg, log, mdir=x.out)
