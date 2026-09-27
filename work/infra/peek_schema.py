import polars as pl
for p in ["/vol/work_v4/norm/test_France_pool.parquet", "/vol/work_v4/norm/test_France_s1.parquet", "/vol/work_v5/pred/test_France_scored.parquet", "/vol/work_v5/pred/test_France_links.parquet"]:
    s = pl.read_parquet_schema(p)
    print(p, dict(s))
print(pl.read_parquet("/vol/work_v4/norm/test_France_pool.parquet", n_rows=3).to_dicts())
