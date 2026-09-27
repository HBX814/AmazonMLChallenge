# -*- coding: utf-8 -*-
"""normalize_frame.py -- polars front-end for normalize.py (Business Entity Resolution, team Master Bolt).

Maps the pure-python normalizers over the DISTINCT (business_name, country) and (business_address, country)
values of a frame only (S2/S3 repeat identical strings), in a spawn-safe process pool, and joins the results
back. Also wraps the native-token dictionary: learned from TRAIN true links only, shipped to every worker.

Contract functions
------------------
add_normalized_columns(df, n_jobs=-1, *, extras=False, city_counts=None, chunk_size=20000) -> pl.DataFrame
    adds  name_norm, name_core, name_compact, name_key, skel, script, name_is_domain      (per distinct name)
          addr_norm, house_nums (List[Utf8]), city_key, state_key                         (per distinct address)
    extras=True also adds  name_distinct, city_cands (List), all_nums (List), street_key, addr_other
learn_dict_from_train(train_s1, train_pool, gt_long, out_tsv, exclude_s1=None, ...) -> int (#entries)
use_native_dict(path_or_None) -> int           reset, then load a dictionary TSV into this process
native_dict_coverage(names) -> dict            share of Indic token occurrences the loaded dictionary covers
evaluate_native_heldout(train_s1, train_pool, gt_long, holdout_frac=0.2) -> dict   step-2 exit gate (held-out S1)
city_counts(df_with_city_cands) -> pl.DataFrame (country, city, n) to make S1 and pool city_key consistent

Usage
-----
    import normalize_frame as NF                      # inside the package: from ber import normalize_frame as NF
    NF.learn_dict_from_train(s1, pool, gt, "work/native_token_dict.tsv")      # TRAIN links only; loads it
    pool_n = NF.add_normalized_columns(pool, extras=True)
    s1_n = NF.add_normalized_columns(s1, city_counts=NF.city_counts(pool_n))
CLI (measurements):
    python normalize_frame.py bench --input mini/train/train_source2.tsv --native-dict native_token_dict.example.tsv
    python normalize_frame.py learn --s1 train_source1.tsv --pool train_source2.tsv train_source3.tsv \
        --gt train_ground_truth.tsv --out work/native_token_dict.tsv
    python normalize_frame.py gate --s1 train_source1.tsv --pool train_source2.tsv train_source3.tsv --gt train_ground_truth.tsv
    python normalize_frame.py coverage --input test_source2.tsv --country India --native-dict work/native_token_dict.tsv
Windows: call add_normalized_columns only from code under `if __name__ == "__main__":` (spawn re-imports __main__).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from collections import Counter, deque

try:
    from . import normalize as N          # inside the ber package
except ImportError:
    import normalize as N                 # flat layout (skill folder, tests)

__all__ = ['add_normalized_columns', 'learn_dict_from_train', 'use_native_dict', 'native_dict_coverage', 'evaluate_native_heldout',
           'city_counts', 'NAME_COLUMNS', 'ADDR_COLUMNS', 'EXTRA_COLUMNS']

NAME_COLUMNS = ['name_norm', 'name_core', 'name_compact', 'name_key', 'skel', 'script', 'name_is_domain']
ADDR_COLUMNS = ['addr_norm', 'house_nums', 'city_key', 'state_key']
EXTRA_COLUMNS = ['name_distinct', 'city_cands', 'all_nums', 'street_key', 'addr_other']
# house/door/plot numbers: designated numbers that name the premises (unit/floor/sector numbers do not).
# Measured on mini-train India true links: house_numbers only -> base-number agreement 54.0%;
# + hno/dno/plot/no -> 76.9% (random same-country pairs 1.4% -> 2.6%). US unchanged (80.7%).
HOUSE_DESIGNATORS = ('hno', 'dno', 'plot', 'no')
_INDIC_CLASS = '[ऀ-ൿ]'
_INDIC_TOKEN_RE = re.compile('[0-9a-zऀ-ൿ‌‍]+')


# =====================================================================================================
# workers (module-level so spawn can pickle them; polars is NOT imported here to keep workers light)
# =====================================================================================================
def _worker_init(state):
    """Replicate the parent's learned dictionary / snapping state: spawn children start with empty globals."""
    N._NATIVE_DICT.clear()
    N._NATIVE_DICT.update(state['native'])
    N._SNAP = state['snap']
    N._TL_CACHE.clear()
    N._register_native_states()


def _nlp_state():
    return {'native': dict(N._NATIVE_DICT), 'snap': N._SNAP}


def _script(s):
    if not s or s.isascii():
        return 'latin'
    if N._HAS_INDIC.search(s):
        return 'indic'
    s = unicodedata.normalize('NFKC', s)
    for ch in s:
        o = ord(ch)
        if o > 0x24F and ch.isalpha() and not (0x1E00 <= o <= 0x1EFF):
            return 'other'
    return 'latin'


def _names_chunk(names, countries, extras):
    out = {k: [] for k in NAME_COLUMNS + (['name_distinct'] if extras else [])}
    for name, country in zip(names, countries):
        v = N.name_variants(name, country)
        out['name_norm'].append(v['norm'])
        out['name_core'].append(v['core'])
        out['name_compact'].append(v['glued'])
        out['name_key'].append(v['key'])
        out['skel'].append(' '.join(N.skeleton_key(t) for t in v['core'].split()))
        out['script'].append(_script(name))
        # domain (.com/.in/...) or @handle: the glued forms where name_compact is the comparable field
        out['name_is_domain'].append(bool(name) and (N._URL_TOKEN_RE.search(name) is not None
                                                     or re.match(r"^\W*[@#]\s*\w", name) is not None))
        if extras:
            out['name_distinct'].append(v['distinct'])
    return out


def _addrs_chunk(addrs, countries, extras):
    out = {k: [] for k in ['addr_norm', 'house_nums', 'state_key', 'city_cands']
           + (['all_nums', 'street_key', 'addr_other'] if extras else [])}
    for addr, country in zip(addrs, countries):
        norm, p = N.address_all(addr, country)
        hn = list(p['house_numbers'])
        for d in HOUSE_DESIGNATORS:
            for x in p['designated_numbers'].get(d, ()):
                if x not in hn:
                    hn.append(x)
        out['addr_norm'].append(norm)
        out['house_nums'].append(hn)
        out['state_key'].append(p['state_canonical'] or '')
        out['city_cands'].append(p['city_candidates'])
        if extras:
            out['all_nums'].append(p['all_numbers'])
            out['street_key'].append(' '.join(p['street_tokens']))
            out['addr_other'].append(' '.join(p['other_tokens']))
    return out


def _work(kind, texts, countries, extras):
    return (_names_chunk if kind == 'name' else _addrs_chunk)(texts, countries, extras)


# =====================================================================================================
# driver
# =====================================================================================================
def _resolve_jobs(n_jobs, n_chunks):
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    if n_jobs < 0:
        n_jobs = max(1, min(8, (os.cpu_count() or 2) - 2))
        try:                                  # ~150 MB per worker; the laptop often has only 1-3 GB free
            import psutil
            n_jobs = max(1, min(n_jobs, int(psutil.virtual_memory().available / 150e6)))
        except Exception:
            pass
    return max(1, min(n_jobs, n_chunks))


def _schema(kind, extras):
    import polars as pl
    if kind == 'name':
        s = {k: pl.Utf8 for k in NAME_COLUMNS}
        s['name_is_domain'] = pl.Boolean
        if extras:
            s['name_distinct'] = pl.Utf8
        return s
    s = {'addr_norm': pl.Utf8, 'house_nums': pl.List(pl.Utf8), 'state_key': pl.Utf8, 'city_cands': pl.List(pl.Utf8)}
    if extras:
        s.update({'all_nums': pl.List(pl.Utf8), 'street_key': pl.Utf8, 'addr_other': pl.Utf8})
    return s


def _chunks(uniq, text_col, kind, chunk_size):
    for i in range(0, uniq.height, chunk_size):     # one chunk of python strings alive at a time
        part = uniq.slice(i, chunk_size)
        yield kind, part[text_col].to_list(), part['country'].to_list()


def _normalize_distinct(names, addrs, n_jobs, chunk_size, extras, min_parallel):
    """Distinct (business_name, country) / (business_address, country) frames -> same rows + output columns.
    Name and address chunks share ONE spawn pool (start-up paid once) and a bounded in-flight window
    (Executor.map would materialise every chunk up front)."""
    import itertools
    import polars as pl
    sch = {'name': _schema('name', extras), 'addr': _schema('addr', extras)}
    n_tasks = -(-names.height // chunk_size) + -(-addrs.height // chunk_size)
    jobs = _resolve_jobs(n_jobs, max(1, n_tasks)) if max(names.height, addrs.height) >= min_parallel else 1
    tasks = itertools.chain(_chunks(names, 'business_name', 'name', chunk_size),
                            _chunks(addrs, 'business_address', 'addr', chunk_size))
    frames = {'name': [], 'addr': []}
    if jobs == 1:
        for kind, t, c in tasks:
            frames[kind].append(pl.DataFrame(_work(kind, t, c, extras), schema=sch[kind]))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context('spawn'),
                                 initializer=_worker_init, initargs=(_nlp_state(),)) as ex:
            inflight = deque()
            for kind, t, c in tasks:
                if len(inflight) >= 2 * jobs:
                    k, fut = inflight.popleft()
                    frames[k].append(pl.DataFrame(fut.result(), schema=sch[k]))
                inflight.append((kind, ex.submit(_work, kind, t, c, extras)))
            while inflight:
                k, fut = inflight.popleft()
                frames[k].append(pl.DataFrame(fut.result(), schema=sch[k]))

    def glue(uniq, kind):
        res = pl.concat(frames[kind], how='vertical') if frames[kind] else pl.DataFrame(schema=sch[kind])
        return uniq.hstack(res.get_columns())
    return glue(names, 'name'), glue(addrs, 'addr')


def city_counts(df):
    """(country, city, n) from a frame that has `city_cands` (add_normalized_columns(..., extras=True)).
    n = number of DISTINCT addresses naming the city. Pass it as `city_counts=` when normalizing another frame of
    the same split (e.g. S1 after the pool) so both sides pick the same canonical city."""
    import polars as pl
    return (df.select('country', 'business_address', 'city_cands').unique(['country', 'business_address'])
            .explode('city_cands', empty_as_null=False).drop_nulls('city_cands')
            .group_by('country', 'city_cands').agg(pl.len().cast(pl.Int64).alias('n'))
            .rename({'city_cands': 'city'}).sort('country', 'city'))


def _pick_city(amap, prior):
    """city_key = the candidate named by the most distinct addresses of the same country (ties -> later
    component, S1 writes the city last). Measured on mini-train true links (city_key equal on both sides):
    India 80.8% (last candidate: 55.3%), US 80.1% (79.9%); random pairs 3.2% / 0.1%."""
    import polars as pl
    ex = (amap.select('_aid', 'country', 'city_cands').explode('city_cands', empty_as_null=False).drop_nulls('city_cands')
          .with_columns(pl.int_range(pl.len()).over('_aid').alias('_pos')))
    cnt = ex.group_by('country', 'city_cands').agg(pl.len().cast(pl.Int64).alias('n'))
    if prior is not None and prior.height:
        pr = prior.select(pl.col('country').cast(pl.Utf8), pl.col('city').cast(pl.Utf8).alias('city_cands'),
                          pl.col('n').cast(pl.Int64))
        cnt = pl.concat([cnt, pr]).group_by('country', 'city_cands').agg(pl.col('n').sum())
    best = (ex.join(cnt, on=['country', 'city_cands'], how='left')
            .group_by('_aid').agg(pl.col('city_cands').sort_by(['n', '_pos'], descending=[True, True]).first()
                                  .alias('city_key')))
    return amap.join(best, on='_aid', how='left').with_columns(pl.col('city_key').fill_null(''))


def add_normalized_columns(df, n_jobs=-1, *, extras=False, city_counts=None, chunk_size=20000, _min_parallel=20000):
    """Add the CONTRACT normalized columns (+ name_key, skel) to a source frame with
    entity_id, business_name, business_address, country. Works per distinct value; row order is preserved.
    n_jobs=-1 -> min(8, cpu-2) spawn workers capped by free RAM; 1 = in-process. Frames with fewer than
    _min_parallel distinct values run in-process (spawn start-up costs ~1-2 s)."""
    import polars as pl
    out_cols = NAME_COLUMNS + ADDR_COLUMNS + EXTRA_COLUMNS
    base = df.drop([c for c in out_cols if c in df.columns])
    base = base.with_columns([pl.col(c).cast(pl.Utf8).fill_null('') for c in ('business_name', 'business_address', 'country')])

    names = base.select('business_name', 'country').unique(maintain_order=True)
    addrs = base.select('business_address', 'country').unique(maintain_order=True)
    nmap, amap = _normalize_distinct(names, addrs, n_jobs, chunk_size, extras, _min_parallel)
    amap = _pick_city(amap.with_columns(pl.int_range(pl.len()).alias('_aid')), city_counts).drop('_aid')
    if not extras:
        amap = amap.drop('city_cands')

    out = (base.join(nmap, on=['business_name', 'country'], how='left', maintain_order='left')
               .join(amap, on=['business_address', 'country'], how='left', maintain_order='left'))
    order = [c for c in out_cols if c in out.columns]
    return out.select([c for c in base.columns] + order)


# =====================================================================================================
# native-token dictionary (a learned artifact: regenerate from TRAIN links inside the pipeline)
# =====================================================================================================
def use_native_dict(path=None, min_support=2):
    """Replace the dictionary of THIS process (normalize.load_native_dict only adds). None = rules only.
    add_normalized_columns ships whatever is loaded here to its workers."""
    N._NATIVE_DICT.clear()
    N._TL_CACHE.clear()
    n = N.load_native_dict(path, min_support=min_support) if path else 0
    N._register_native_states()
    return n


def learn_dict_from_train(train_s1, train_pool, gt_long, out_tsv, exclude_s1=None, min_support=2, min_share=0.4,
                          sim_thr=0.55, load=True):
    """Learn native->latin tokens from TRAIN true links (S1 Latin name, S2/S3 native-script name) and write
    'native<TAB>latin<TAB>support<TAB>share'. Frames or LazyFrames: train_s1/train_pool need entity_id,
    business_name; gt_long needs s1, mid. exclude_s1 = S1 ids to hold out (validation fold) when measuring.
    NEVER pass test data. Returns #entries; load=True also makes it the active dictionary (use_native_dict)."""
    import polars as pl
    t0 = time.time()
    lz = lambda x: x.lazy() if isinstance(x, pl.DataFrame) else x   # noqa: E731
    gt = lz(gt_long).select(pl.col('s1').cast(pl.Utf8), pl.col('mid').cast(pl.Utf8))
    if exclude_s1 is not None:
        ex = exclude_s1 if isinstance(exclude_s1, pl.Series) else pl.Series('s1', list(exclude_s1), dtype=pl.Utf8)
        gt = gt.filter(~pl.col('s1').is_in(ex.cast(pl.Utf8).implode()))
    native = (lz(train_pool).select(pl.col('entity_id').alias('mid'), pl.col('business_name').alias('native'))
              .filter(pl.col('native').str.contains(_INDIC_CLASS)))
    latin = lz(train_s1).select(pl.col('entity_id').alias('s1'), pl.col('business_name').alias('latin'))
    pairs = (native.join(gt, on='mid', how='inner').join(latin, on='s1', how='inner')
             .select('latin', 'native').unique().sort('latin', 'native').collect(engine='streaming'))
    if pairs.height == 0:
        raise ValueError('no (S1 name, native-script S2/S3 name) training pairs found -- wrong inputs?')
    d = N.learn_native_dict(pairs.iter_rows(), min_support=min_support, min_share=min_share, sim_thr=sim_thr)
    os.makedirs(os.path.dirname(os.path.abspath(out_tsv)), exist_ok=True)
    with open(out_tsv, 'w', encoding='utf-8', newline='\n') as f:
        f.write('native\tlatin\tsupport\tshare\n')
        for nat, (lat, sup, sh) in sorted(d.items(), key=lambda kv: (-kv[1][1], kv[0])):
            f.write(f'{nat}\t{lat}\t{sup}\t{sh}\n')
    if load:
        use_native_dict(out_tsv, min_support=min_support)
    print(f'[normalize_frame] native dict: {pairs.height} pairs -> {len(d)} entries in {time.time() - t0:.1f}s -> {out_tsv}',
          file=sys.stderr)
    return len(d)


def evaluate_native_heldout(train_s1, train_pool, gt_long, holdout_frac=0.2, seed=11, out_tsv=None):
    """Playbook step-2 gate: learn the dictionary WITHOUT a random holdout_frac of train S1, then score name_core
    on the held-out S1's native-script true links, with and without the dictionary. The previously loaded
    dictionary is restored afterwards. Returns {'pairs', 'entries', 'dict': {tsr, jw, exact}, 'rules': {...}}."""
    import tempfile
    import polars as pl
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    lz = lambda x: x.lazy() if isinstance(x, pl.DataFrame) else x   # noqa: E731
    s1 = lz(train_s1).select(pl.col('entity_id').cast(pl.Utf8), pl.col('business_name').alias('latin'),
                             pl.col('country').cast(pl.Utf8)).collect()
    held = s1['entity_id'].sort().sample(fraction=holdout_frac, seed=seed)
    prev = _nlp_state()
    out_tsv = out_tsv or os.path.join(tempfile.gettempdir(), 'native_token_dict.heldout.tsv')
    try:
        n = learn_dict_from_train(s1.lazy().select('entity_id', pl.col('latin').alias('business_name')), train_pool,
                                  gt_long, out_tsv, exclude_s1=held, load=True)
        pairs = (lz(train_pool).select(pl.col('entity_id').alias('mid'), pl.col('business_name').alias('native'))
                 .filter(pl.col('native').str.contains(_INDIC_CLASS))
                 .join(lz(gt_long).select(pl.col('s1').cast(pl.Utf8), pl.col('mid').cast(pl.Utf8))
                       .filter(pl.col('s1').is_in(held.implode())), on='mid', how='inner')
                 .join(s1.lazy().rename({'entity_id': 's1'}), on='s1', how='inner')
                 .select('latin', 'native', 'country').unique().sort('latin', 'native').collect(engine='streaming').rows())

        def score():
            a = [(N.name_core(la, c), N.name_core(na, c)) for la, na, c in pairs]
            k = max(1, len(a))
            return {'tsr': round(sum(fuzz.token_set_ratio(x, y) for x, y in a) / k, 2),
                    'jw': round(sum(JaroWinkler.normalized_similarity(x, y) for x, y in a) / k, 4),
                    'exact': round(sum(x == y for x, y in a) / k, 4)}
        with_dict = score()
        use_native_dict(None)
        rules = score()
    finally:
        _worker_init(prev)
    return {'pairs': len(pairs), 'entries': n, 'dict': with_dict, 'rules': rules}


def native_dict_coverage(names):
    """Share of Indic-script token OCCURRENCES (and types) found in the loaded dictionary. Unlabeled data is fine
    (test S2/S3): nothing is learned. Measured 2026-09-25: full test India names 96.4% of occurrences."""
    cnt = Counter()
    for s in names:
        if s and N._HAS_INDIC.search(s):
            for t in _INDIC_TOKEN_RE.findall(unicodedata.normalize('NFC', s).lower()):
                if N._HAS_INDIC.search(t):
                    cnt[''.join(ch for ch in t if ch not in N._ZW)] += 1
    tot = sum(cnt.values())
    cov = sum(v for t, v in cnt.items() if t in N._NATIVE_DICT)
    missing = [t for t, _ in cnt.most_common() if t not in N._NATIVE_DICT][:30]
    return {'tokens': tot, 'covered': cov, 'token_coverage': cov / tot if tot else 1.0, 'types': len(cnt),
            'type_coverage': (sum(1 for t in cnt if t in N._NATIVE_DICT) / len(cnt)) if cnt else 1.0,
            'top_missing': missing}


# =====================================================================================================
# CLI: measurements / dictionary regeneration
# =====================================================================================================
def _scan(path, limit=None, country=None):
    import polars as pl
    if path.endswith('.parquet'):
        lf = pl.scan_parquet(path)
    else:
        lf = pl.scan_csv(path, separator='\t', quote_char=None, infer_schema=False).with_columns(pl.all().fill_null(''))
    if country:
        lf = lf.filter(pl.col('country') == country)
    return lf.head(limit) if limit else lf


def _gt_long(path):
    import polars as pl
    lf = _scan(path)
    cols = lf.collect_schema().names()
    if 'matched_entity_ids' in cols:
        lf = (lf.select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('mid'))
              .explode('mid', empty_as_null=False).filter(pl.col('mid').is_not_null() & (pl.col('mid') != '')))
    return lf.select('s1', 'mid')


def _main(argv=None):
    import polars as pl
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('bench', help='normalize a file, report distinct ratios and us/row')
    b.add_argument('--input', required=True)
    b.add_argument('--country')
    b.add_argument('--limit', type=int)
    b.add_argument('--native-dict')
    b.add_argument('--n-jobs', type=int, default=-1)
    b.add_argument('--out', help='optional parquet output')
    le = sub.add_parser('learn', help='learn the native-token dictionary from TRAIN links')
    le.add_argument('--s1', required=True)
    le.add_argument('--pool', required=True, nargs='+')
    le.add_argument('--gt', required=True)
    le.add_argument('--out', required=True)
    le.add_argument('--exclude-s1', help='text file, one S1 id per line (held-out fold)')
    g = sub.add_parser('gate', help='held-out native-name uplift of the learned dictionary (TRAIN only)')
    g.add_argument('--s1', required=True)
    g.add_argument('--pool', required=True, nargs='+')
    g.add_argument('--gt', required=True)
    g.add_argument('--holdout-frac', type=float, default=0.2)
    g.add_argument('--out', help='where to write the held-out dictionary TSV')
    cv = sub.add_parser('coverage', help='dictionary coverage of Indic tokens in a file (unlabeled ok)')
    cv.add_argument('--input', required=True)
    cv.add_argument('--country')
    cv.add_argument('--limit', type=int)
    cv.add_argument('--native-dict', required=True)
    a = ap.parse_args(argv)

    if a.cmd == 'bench':
        n_d = use_native_dict(a.native_dict) if a.native_dict else 0
        df = _scan(a.input, a.limit, a.country).collect()
        t0 = time.time()
        out = add_normalized_columns(df, n_jobs=a.n_jobs)
        dt = time.time() - t0
        nn = df.select('business_name', 'country').n_unique()
        na = df.select('business_address', 'country').n_unique()
        print(f'rows={df.height} dict_entries={n_d} distinct_names={nn} ({nn / max(1, df.height):.3f}) '
              f'distinct_addrs={na} ({na / max(1, df.height):.3f}) n_jobs={a.n_jobs} seconds={dt:.1f} '
              f'us_per_row={1e6 * dt / max(1, df.height):.0f} us_per_distinct={1e6 * dt / max(1, nn + na):.0f}')
        print(out.head(5))
        if a.out:
            out.write_parquet(a.out)
    elif a.cmd == 'learn':
        pool = pl.concat([_scan(p).select('entity_id', 'business_name') for p in a.pool], how='vertical')
        ex = None
        if a.exclude_s1:
            with open(a.exclude_s1, encoding='utf-8') as f:
                ex = [x.strip() for x in f if x.strip()]
        learn_dict_from_train(_scan(a.s1), pool, _gt_long(a.gt), a.out, exclude_s1=ex)
    elif a.cmd == 'gate':
        pool = pl.concat([_scan(p).select('entity_id', 'business_name') for p in a.pool], how='vertical')
        print(evaluate_native_heldout(_scan(a.s1), pool, _gt_long(a.gt), holdout_frac=a.holdout_frac, out_tsv=a.out))
    else:
        use_native_dict(a.native_dict)
        names = _scan(a.input, a.limit, a.country).select('business_name').collect()['business_name']
        print(native_dict_coverage(names))


if __name__ == '__main__':
    _main()
