# -*- coding: utf-8 -*-
"""Tests for normalize_frame.py (+ regression tests for the city_candidates fix in normalize.py).
Run:  python -m pytest -q -p no:cacheprovider test_normalize_frame.py
Mini-data tests read ER_MINI_DIR (default: the session scratchpad mini/train) and skip when it is absent."""
import os
import sys

import polars as pl
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import normalize as N  # noqa: E402
import normalize_frame as NF  # noqa: E402

EXAMPLE_DICT = os.path.join(HERE, 'native_token_dict.example.tsv')
MINI = os.environ.get('ER_MINI_DIR', r'C:/Users/harsh/AppData/Local/Temp/claude/c--Users-harsh-Downloads-AmazonMLChallenge/'
                      r'da46a70e-4037-4864-96e9-d5a67558da5d/scratchpad/mini/train')
HAS_MINI = os.path.exists(os.path.join(MINI, 'train_source1.tsv'))
TMP = os.environ.get('ER_TEST_TMP', r'C:/Users/harsh/AppData/Local/Temp/claude/c--Users-harsh-Downloads-AmazonMLChallenge/'
                     r'da46a70e-4037-4864-96e9-d5a67558da5d/scratchpad/itest/b-norm')

ROWS = [
    ('S1-1', 'Fortune Finance Private Limited', '12 MG Road, Bangalore, Karnataka', 'India'),
    ('S2-1', 'ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', '12 MG ROAD, BANGALORE, ಕರ್ನಾಟಕ', 'India'),
    ('S1-2', 'Unified Pinnacle Esports L.L.C.', '221 Seneca Street, Corning, NY', 'US'),
    ('S2-2', 'unifiedpinnacleesports.com', '221 Seneca Saint, Corning, New York', 'US'),
    ('S3-2', 'Unified Pinnacle Ésports L.L.C.', '221 SENECA STREET, CRNING, NY', 'US'),
    ('S3-3', 'Unified Pinnacle Ésports L.L.C.', '221 SENECA STREET, CRNING, NY', 'US'),     # duplicate strings
    ('S1-3', 'Girwa Projects Private Limited', 'Niwasa, Near Sbi Atm, Punjawati Bhuwana, Girwa, Udaipur, Rajasthan', 'India'),
    ('S3-4', 'Girwa Projects  Private Limited', 'Hn 352 Niwasa, Udaipur, RJ, Girwa', 'India'),
    ('S1-4', 'Chretien Groupe SASU', '121 Rue Neuve, Calais, Hauts-de-France', 'France'),
    ('S2-5', '@COMITCERCLE', "8 R. DE LA PORTE D’EAU, DUNKERQUE, Nord", 'France'),
    ('S2-6', 'Acme Widgets GmbH', '12 bis Hauptstrasse, Berlin', 'Germany'),
    ('S2-7', '١٢٣ شارع', '', 'Brazil'),
    ('S2-8', 'Door Traders', 'Door No 4-5-6, Plot 12, Opp Bus Stand, Udaipur, Rajasthan', 'India'),
    ('S2-9', 'Katz Seafood', '1804A Main St, Unit 5, Austin, Texas', 'US'),
    ('S2-10', 'Box Co', 'PO Box 123, Nashville, TN', 'US'),
    ('S2-11', 'Suite Co', '12 Main St, Suite C, Waltham, MA', 'US'),
]


def frame(rows=ROWS):
    return pl.DataFrame(rows, schema=['entity_id', 'business_name', 'business_address', 'country'], orient='row')


def read_tsv(path):
    return pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False).with_columns(pl.all().fill_null(''))


@pytest.fixture(autouse=True)
def _example_dict():
    NF.use_native_dict(EXAMPLE_DICT)
    yield
    NF.use_native_dict(EXAMPLE_DICT)


# ------------------------------------------------------------------ contract shape
def test_contract_columns_types_order_no_nulls():
    df = frame()
    out = NF.add_normalized_columns(df, n_jobs=1)
    assert out.height == df.height and out['entity_id'].to_list() == df['entity_id'].to_list()
    for c in ['name_norm', 'name_core', 'name_compact', 'name_key', 'skel', 'script', 'addr_norm', 'city_key', 'state_key']:
        assert out.schema[c] == pl.Utf8, c
    assert out.schema['house_nums'] == pl.List(pl.Utf8) and out.schema['name_is_domain'] == pl.Boolean
    assert out.null_count().sum_horizontal()[0] == 0
    assert out.columns[:4] == ['entity_id', 'business_name', 'business_address', 'country']
    assert 'city_cands' not in out.columns


def test_values_equal_scalar_functions():
    out = NF.add_normalized_columns(frame(), n_jobs=1)
    for r in out.iter_rows(named=True):
        n, a, c = r['business_name'], r['business_address'], r['country']
        assert r['name_norm'] == N.normalize_name(n, c)
        assert r['name_core'] == N.name_core(n, c)
        assert r['name_compact'] == N.name_glued(n, c)
        assert r['name_key'] == N.name_key(n, c)
        assert r['skel'] == ' '.join(N.skeleton_key(t) for t in N.name_core(n, c).split())
        assert r['addr_norm'] == N.normalize_address(a, c)
        assert r['state_key'] == (N.address_parts(a, c)['state_canonical'] or '')


def test_expected_values_native_domain_france():
    out = {r['entity_id']: r for r in NF.add_normalized_columns(frame(), n_jobs=1).iter_rows(named=True)}
    assert out['S2-1']['name_norm'] == 'fortune finance pvt ltd' and out['S2-1']['name_core'] == out['S1-1']['name_core']
    assert out['S2-1']['script'] == 'indic' and out['S1-1']['script'] == 'latin' and out['S2-7']['script'] == 'other'
    assert out['S2-1']['state_key'] == out['S1-1']['state_key'] == 'karnataka'
    assert out['S2-2']['name_compact'] == out['S1-2']['name_compact'] == 'unifiedpinnacleesports'
    assert out['S2-2']['name_is_domain'] and out['S2-5']['name_is_domain'] and not out['S1-2']['name_is_domain']
    assert out['S1-2']['skel'] == 'unfd pnkl esprts'
    assert out['S1-4']['state_key'] == out['S2-5']['state_key'] == 'hauts de france'     # region vs department
    assert out['S2-6']['house_nums'] == ['12b'] and out['S2-6']['city_key'] == 'berlin'  # unknown country
    assert out['S2-7']['addr_norm'] == '' and out['S2-7']['house_nums'] == []


def test_house_nums_house_door_plot_not_units():
    out = {r['entity_id']: r for r in NF.add_normalized_columns(frame(), n_jobs=1).iter_rows(named=True)}
    assert out['S3-4']['house_nums'] == ['352']                  # 'Hn 352'
    assert out['S2-8']['house_nums'] == ['4-5-6', '12']          # door no + plot
    assert out['S2-9']['house_nums'] == ['1804a']                # 'Unit 5' excluded
    assert out['S1-2']['house_nums'] == ['221']


def test_city_key_most_frequent_candidate_and_prior():
    rows = [(f'S1-u{i}', 'X', f'{i} Lake Rd, Udaipur, Rajasthan', 'India') for i in range(3)]
    rows += [('S3-g', 'Girwa Projects', 'Hn 352 Niwasa, Udaipur, RJ, Girwa', 'India')]
    out = NF.add_normalized_columns(frame(rows), n_jobs=1, extras=True)
    assert out.filter(pl.col('entity_id') == 'S3-g')['city_key'][0] == 'udaipur'     # not the last ('girwa')
    lone = frame([('S3-g', 'Girwa Projects', 'Hn 352 Niwasa, Udaipur, RJ, Girwa', 'India')])
    assert NF.add_normalized_columns(lone, n_jobs=1)['city_key'][0] == 'girwa'       # tie -> later component
    prior = NF.city_counts(out)
    assert set(prior.columns) == {'country', 'city', 'n'}
    assert NF.add_normalized_columns(lone, n_jobs=1, city_counts=prior)['city_key'][0] == 'udaipur'


def test_city_candidates_junk_fixed():
    # regression for the normalize.py fix (po box / 'Suite C' / bare codes are not cities)
    assert N.address_parts('PO Box 123, Nashville, TN', 'US')['city_candidates'] == ['nashville']
    assert N.address_parts('12 Main St, Suite C, Waltham, MA', 'US')['city_candidates'] == ['waltham']
    assert 'dl' not in N.address_parts('2685 Naya Bazar, DL, Delhi', 'India')['city_candidates']
    out = {r['entity_id']: r for r in NF.add_normalized_columns(frame(), n_jobs=1).iter_rows(named=True)}
    assert out['S2-10']['city_key'] == 'nashville' and out['S2-11']['city_key'] == 'waltham'


def test_extras_columns():
    out = NF.add_normalized_columns(frame(), n_jobs=1, extras=True)
    for c in NF.EXTRA_COLUMNS:
        assert c in out.columns
    r = out.filter(pl.col('entity_id') == 'S2-8').row(0, named=True)
    assert r['all_nums'] == ['4-5-6', '12'] and 'udaipur' in r['city_cands']
    assert out.filter(pl.col('entity_id') == 'S1-2')['street_key'][0] == 'seneca'


def test_nulls_and_rerun_replace_columns():
    df = frame().with_columns(pl.when(pl.col('entity_id') == 'S2-9').then(None).otherwise(pl.col('business_address'))
                              .alias('business_address'))
    out = NF.add_normalized_columns(df, n_jobs=1)
    assert out.filter(pl.col('entity_id') == 'S2-9')['addr_norm'][0] == ''
    again = NF.add_normalized_columns(out, n_jobs=1)
    assert again.columns == out.columns and again.equals(out)


def test_process_pool_matches_inprocess_and_ships_dict():
    df = frame(ROWS * 20).with_columns((pl.col('entity_id') + '-' + pl.int_range(pl.len()).cast(pl.Utf8)).alias('entity_id'))
    df = df.with_columns((pl.col('business_name') + ' ' + (pl.int_range(pl.len()) % 7).cast(pl.Utf8)).alias('business_name'))
    a = NF.add_normalized_columns(df, n_jobs=1)
    b = NF.add_normalized_columns(df, n_jobs=2, chunk_size=13, _min_parallel=0)
    assert a.equals(b)
    assert b.filter(pl.col('entity_id').str.starts_with('S2-1-'))['name_core'].str.starts_with('fortune finance').all()


def test_use_native_dict_replaces():
    assert NF.use_native_dict(None) == 0
    assert N.normalize_name('ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', 'India') != 'fortune finance pvt ltd'
    assert NF.use_native_dict(EXAMPLE_DICT) > 1000
    assert N.normalize_name('ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', 'India') == 'fortune finance pvt ltd'


def test_native_dict_coverage():
    cov = NF.native_dict_coverage(['ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', 'Plain Latin', None])
    assert cov['tokens'] == 4 and cov['token_coverage'] == 1.0
    NF.use_native_dict(None)
    assert NF.native_dict_coverage(['ಫಾರ್ಚೂನ್ ಲಿಮಿಟೆಡ್'])['token_coverage'] == 0.0


def test_contract_entry_point_in_normalize_delegates():
    assert N.add_normalized_columns(frame(), n_jobs=1).equals(NF.add_normalized_columns(frame(), n_jobs=1))


# ------------------------------------------------------------------ mini data (skipped when absent)
def mini_train():
    s1 = read_tsv(os.path.join(MINI, 'train_source1.tsv'))
    pool = pl.concat([read_tsv(os.path.join(MINI, f'train_source{k}.tsv')) for k in (2, 3)])
    gt = (read_tsv(os.path.join(MINI, 'train_ground_truth.tsv'))
          .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('mid'))
          .explode('mid', empty_as_null=False).filter(pl.col('mid') != ''))
    return s1, pool, gt


@pytest.mark.skipif(not HAS_MINI, reason='mini dataset not available (set ER_MINI_DIR)')
def test_evaluate_native_heldout_gate_and_restore():
    s1, pool, gt = mini_train()
    os.makedirs(TMP, exist_ok=True)
    r = NF.evaluate_native_heldout(s1, pool.lazy(), gt.lazy(), holdout_frac=0.2, out_tsv=os.path.join(TMP, 'dict_gate.tsv'))
    assert r['pairs'] > 300 and r['dict']['tsr'] > r['rules']['tsr'] + 10, r
    # the example dictionary loaded by the fixture is back in place
    assert N.normalize_name('ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', 'India') == 'fortune finance pvt ltd'


@pytest.mark.skipif(not HAS_MINI, reason='mini dataset not available (set ER_MINI_DIR)')
def test_learn_dict_on_mini_train_heldout_uplift():
    from rapidfuzz.distance import JaroWinkler
    s1, pool, gt = mini_train()
    ids = s1['entity_id'].sort()
    held = set(ids.sample(fraction=0.2, seed=11).to_list())
    os.makedirs(TMP, exist_ok=True)
    out_tsv = os.path.join(TMP, 'dict_mini.tsv')
    n = NF.learn_dict_from_train(s1, pool.lazy(), gt, out_tsv, exclude_s1=held)
    assert n > 100
    with open(out_tsv, encoding='utf-8') as f:
        assert f.readline() == 'native\tlatin\tsupport\tshare\n'
    pairs = (pool.filter(pl.col('business_name').str.contains('[\u0900-\u0D7F]'))
             .join(gt.filter(pl.col('s1').is_in(list(held))), left_on='entity_id', right_on='mid')
             .join(s1, left_on='s1', right_on='entity_id', suffix='_s1')
             .select('business_name_s1', 'business_name').rows())
    assert len(pairs) > 300

    def score():
        jw = [JaroWinkler.normalized_similarity(N.name_core(a, 'India'), N.name_core(b, 'India')) for a, b in pairs]
        return sum(jw) / len(jw)
    with_dict = score()
    NF.use_native_dict(None)
    rules = score()
    assert with_dict > rules + 0.1, (rules, with_dict)


@pytest.mark.skipif(not HAS_MINI, reason='mini dataset not available (set ER_MINI_DIR)')
def test_mini_pool_frame_runs_and_reduces_distinct():
    pool = pl.concat([read_tsv(os.path.join(MINI, f'train_source{k}.tsv')) for k in (2, 3)]).head(6000)
    out = NF.add_normalized_columns(pool, n_jobs=1)
    assert out.height == pool.height and out.null_count().sum_horizontal()[0] == 0
    assert (out['state_key'] != '').mean() > 0.9
