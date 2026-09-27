# -*- coding: utf-8 -*-
"""diag_stage2_g_write.py [suffix] -- write stage-2 test links (test_<c>_<suffix>_links.parquet from
diag_stage2_d_test.py / diag_stage2_i_guard.py) as a submission-format matching TSV under /vol/diag/stage2/
(NOT the official output/): matching_results_stage2.tsv for suffix s2 (default), matching_results_<suffix>.tsv otherwise.
The candidate_pairs.tsv of v3 stays valid: stage 2 only re-scores v3's candidate pairs."""
import glob
import sys

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402
from ber import outputs as O        # noqa: E402

OUT = "/vol/diag/stage2"
suffix = sys.argv[1] if len(sys.argv) > 1 else "s2"
name = "matching_results_stage2.tsv" if suffix == "s2" else f"matching_results_{suffix}.tsv"
ids = O.read_s1_order("/root/proj/student_resource/dataset/test/test_source1.tsv")
paths = sorted(glob.glob(f"{OUT}/test_*_{suffix}_links.parquet"))
assert len(paths) >= 1, paths
links = pl.concat([pl.read_parquet(p) for p in paths])
assert links.select("mid").n_unique() == links.height, "a record is linked to two S1"
O.write_id_lists(ids, links, f"{OUT}/{name}", "matched_entity_ids")
print(f"{name}: {len(paths)} country files, {len(ids):,} S1 rows, {links.height:,} links ({links.height / len(ids):.4f}/S1)")
