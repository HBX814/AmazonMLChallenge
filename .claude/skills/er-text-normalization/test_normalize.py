# -*- coding: utf-8 -*-
"""Unit tests for normalize.py on REAL examples quoted in EDA_FACTS.md (verbatim strings from the data).
Run:  python -m pytest -q -p no:cacheprovider test_normalize.py"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import normalize as N  # noqa: E402

DICT = os.environ.get('NATIVE_DICT', os.path.join(HERE, 'native_token_dict.example.tsv'))


@pytest.fixture(scope='module', autouse=True)
def _dict():
    if os.path.exists(DICT):
        N.load_native_dict(DICT)
    yield


# ---------------------------------------------------------------- transliteration
def test_translit_matras_survive():
    # NFD + strip combining marks would give 'लमटड'; transliteration must keep the vowel signs
    assert N.transliterate('लिमिटेड', use_dict=False).strip() == 'limited'


@pytest.mark.parametrize('native,expected', [
    ('ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', 'fortune finance pvt ltd'),     # Kannada (EDA example)
    ('अरिहंत इंफ्रास्ट्रक्चर प्रा. लि.', 'arihant infrastructure pvt ltd'),        # Devanagari abbreviations
    ('Galaxy एनर्जी Private Limited', 'galaxy energy pvt ltd'),                      # mixed script
])
def test_native_names_with_dict(native, expected):
    assert N.normalize_name(native, 'India') == expected


def test_rule_translit_all_scripts_nonempty_ascii():
    samples = ['इंडो टेक्नोलॉजी', 'ব্রাইট ফুডস', 'ਸਨਰਾਈਜ਼ ਮਾਰਕੀਟਿੰਗ', 'ગ્લોબલ પાયોનિયર', 'ବ୍ରାଇଟ ଫୁଡସ',
               'ஈஸ்ட் டெக்னாலஜி', 'శివం ఫుడ్స్', 'ಕರ್ನಾಟಕ', 'ബാലാജി മാനേജ്മെന്റ്']
    for s in samples:
        out = N.transliterate(s, use_dict=False)
        assert out.strip() and all(ord(c) < 128 for c in out), (s, out)


def test_translit_passthrough_latin_and_unknown():
    assert N.transliterate('Crème Brûlée SARL') == 'Crème Brûlée SARL'   # accents untouched (fold later)
    assert N.transliterate('') == ''
    assert N.transliterate(None) == ''


# ---------------------------------------------------------------- names
US_GROUP = ['Unified Pinnacle Esports L.L.C.', 'Unified Pinnacle Esports  L.L.C.', 'Unified Pinnacle Ésports L.L.C.',
            'Onyxcalovantage Sys D.B.A. Unified Pinnacle Esports L.L.C.']


def test_us_group_same_core():
    cores = {N.name_core(x, 'US') for x in US_GROUP}
    assert cores == {'unified pinnacle esports'}
    assert N.normalize_name(US_GROUP[0], 'US') == 'unified pinnacle esports llc'


def test_domain_form_glued():
    assert N.normalize_name('unifiedpinnacleesports.com', 'US') == 'unifiedpinnacleesports'
    assert N.name_glued('Unified Pinnacle Esports L.L.C.', 'US') == 'unifiedpinnacleesports'
    assert N.split_concat('unifiedpinnacleesports', ['unified', 'pinnacle', 'esports', 'llc']) == 'unified pinnacle esports'


def test_domain_with_phone():
    assert N.normalize_name('prantechcorporate.com - 9895970478', 'India') == 'prantechcorporate'
    assert N.name_flags('prantechcorporate.com - 9895970478')['phone']


def test_dba_split():
    primary, alias = N.split_dba('Onyxcalovantage Sys D.B.A. Unified Pinnacle Esports L.L.C.')
    assert alias == 'Onyxcalovantage Sys' and primary.startswith('Unified Pinnacle')
    assert N.split_dba('Ectoarcsol doing business as Patriot Eco')[0] == 'Patriot Eco'
    assert N.split_dba('Plain Name LLC') == ('Plain Name LLC', None)


def test_duplicated_and_moved_legal_suffix():
    assert N.normalize_name('LLC LLC Womens Health', 'US') == 'llc womens health'
    assert N.name_core('LLC Womens Health', 'US') == 'womens health'


def test_india_suffix_variants_and_shuffles():
    s1 = 'Girwa Projects Private Limited'
    for v in ['GIRWA  PROJECTS PRIVATE', 'Girwa Projects  Private Limited', 'Girwa Projects Pvt. Ltd.']:
        assert N.name_core(v, 'India') == 'girwa projects'
    assert N.name_core(s1, 'India') == 'girwa projects'
    # suffix moved to front / tokens shuffled -> sorted key equal
    assert N.name_key('Private Exports Dharani Water Limited', 'India') == N.name_key('Exports Dharani Water Private Limited', 'India')
    assert N.name_key('Pvt. EFS Print Ventures Ltd.', 'India') == 'efs print ventures'
    assert N.name_key('Netra Limited Private Technologies', 'India') == 'netra technologies'


def test_token_shuffle_us():
    assert N.name_key('Urology Elite Physicians', 'US') == N.name_key('Physicians Elite Urology', 'US')


def test_honorifics_and_ms():
    assert N.name_core('M/s Sri Balaji Traders', 'India') == 'balaji traders'
    assert N.name_core('Dr Shri Ram Clinic', 'India') == 'ram clinic'
    assert N.name_core('Smt. Laxmi Stores', 'India') == 'laxmi stores'


def test_junk_symbols_brackets():
    assert N.normalize_name('>> Katz Seafood', 'US') == 'katz seafood'
    assert N.normalize_name('[[FFSAHIEGX]] Katz Seafood', 'US') == 'katz seafood'
    assert N.normalize_name('-- Katz Seafood <<', 'US') == 'katz seafood'
    assert N.name_core('Hospitality Partners [Ltd]', 'US') == 'hospitality partners'
    assert N.name_core('Dance Club (France)', 'France') == 'dance club'


def test_injected_accents_and_leet():
    assert N.normalize_name('DÀNCE Çlub', 'France') == 'dance club'
    assert N.normalize_name('Ássociates Frèrés', 'US') == 'associates freres'
    assert N.normalize_name('Hospita1ity Group', 'US') == 'hospitality group'


def test_handle_form():
    assert N.normalize_name('@COMITCERCLE', 'France') == 'comitcercle'
    assert N.name_flags('@COMITCERCLE')['handle']


def test_france_legal_forms():
    assert N.normalize_name('Compagnons  Sport S.A.R.L.', 'France') == 'compagnons sport sarl'
    assert N.name_core('Compagnons Sport SARL', 'France') == 'compagnons sport'
    assert N.name_core('Chretien Groupe SASU', 'France') == 'chretien groupe'
    assert N.name_core('Établissements Martin & Fils', 'France') == N.name_core('Ets Martin et Fils', 'France') == 'martin'
    assert N.name_core('des Union Retro', 'France') == 'union retro'        # injected leading article
    assert N.name_core('SAS Bordeaux Club', 'France') == 'bordeaux club'


def test_unknown_country_generic_rules():
    for c in ('Germany', None, '', 'XX', 42):
        assert N.name_core('SAS Bordeaux Club', c) == 'bordeaux club'
        assert N.name_core('Acme Widgets L.L.C.', c) == 'acme widgets'
        assert N.name_core('Acme Widgets Private Limited', c) == 'acme widgets'


def test_core_never_empty():
    assert N.name_core('LLC', 'US') == 'llc'
    assert N.name_core('Private Limited', 'India') == 'pvt ltd'


# ---------------------------------------------------------------- addresses
def test_us_address_variants_equal_key():
    a = ['221 Seneca Street, Corning, NY', '221 Seneca Saint, Corning, New York',   # wrong expansion Saint
         '221 Seneca Street, New York, Corning']                                   # reordered
    keys = {N.address_key(x, 'US') for x in a}
    assert keys == {'221 corning ny seneca st'}
    p = N.address_parts('221 SENECA STREET, CRNING, NY', 'US')
    assert p['house_numbers'] == ['221'] and p['street_tokens'] == ['seneca'] and p['state_canonical'] == 'ny'
    assert p['city_candidates'] == ['crning']


def test_null_tokens_removed():
    assert N.normalize_address('PLEASANT RUN ROAD, NULL, IRVING, TX', 'US') == 'pleasant run rd irving tx'
    assert N.normalize_address('123 Main St, N/A, <NULL>, None', 'US') == '123 main st'
    assert N.normalize_address('', 'US') == '' and N.normalize_address(None, 'India') == ''


def test_reordered_us_components():
    p = N.address_parts('MA, SWANSEA, 25 RONALD DR', 'US')
    assert p['state_canonical'] == 'ma' and p['house_numbers'] == ['25'] and p['street_type'] == 'dr'
    assert 'swansea' in p['city_candidates']


def test_house_number_suffix_and_units():
    p = N.address_parts('1804A Main St, Unit 5, Austin, Texas', 'US')
    assert p['house_numbers'] == ['1804a'] and p['house_base'] == ['1804'] and p['unit_numbers'] == ['5']
    assert p['state_canonical'] == 'tx'
    p = N.address_parts('#800 Oak Ave, Fl 1, Dallas, TX', 'US')
    assert p['house_numbers'] == ['800'] and p['unit_numbers'] == ['1']


def test_india_native_state_and_codes():
    for a in ['Niwasa, Near Sbi Atm, Punjawati Bhuwana, Girwa, Udaipur, Rajasthan',
              'राजस्थान, NIWASA, NEAR SBI ATM, PUNJAWATI BHUWANA, GIRWA',
              'Hn 352 Niwasa, Udaipur, RJ, Girwa']:
        assert N.address_parts(a, 'India')['state_canonical'] == 'rajasthan', a
    p = N.address_parts('Hn 352 Niwasa, Udaipur, RJ, Girwa', 'India')
    assert p['designated_numbers'] == {'hno': ['352']} and 'niwasa' in p['city_candidates']
    p = N.address_parts('राजस्थान, NIWASA, NEAR SBI ATM, PUNJAWATI BHUWANA, GIRWA', 'India')
    assert p['landmarks'] == ['near sbi atm']
    assert N.address_parts('2685, Naya Bazar, Delhi, DL', 'India')['state_canonical'] == 'delhi'
    assert N.address_parts('x, दिल्ली', 'India')['state_canonical'] == 'delhi'
    assert N.address_parts('x, ಕರ್ನಾಟಕ', 'India')['state_canonical'] == 'karnataka'
    assert N.address_parts('x, தமிழ்நாடு', 'India')['state_canonical'] == 'tamil nadu'
    assert N.address_parts('x, महाराष्ट्र', 'India')['state_canonical'] == 'maharashtra'
    assert N.address_parts('x, Orissa', 'India')['state_canonical'] == N.address_parts('x, OD', 'India')['state_canonical']


def test_india_floor_ordinals_and_door_numbers():
    assert N.normalize_address('2685, Iind Floor, Naya Bazar, Delhi', 'India') == '2685 sf naya bazar delhi'
    p = N.address_parts('Door No 4-5-6, Plot 12, Opp Bus Stand', 'India')
    assert p['designated_numbers'] == {'dno': ['4-5-6'], 'plot': ['12']}
    p = N.address_parts('12/3 MG Road, Pune - 411001', 'India')
    assert p['house_numbers'] == ['12/3'] and p['postcodes'] == ['411001'] and p['street_tokens'] == ['mg']


def test_france_addresses():
    p = N.address_parts('121 Rue Neuve, Calais, Hauts-de-France', 'France')
    assert (p['house_numbers'], p['street_type'], p['street_tokens'], p['state_canonical']) == \
        (['121'], 'rue', ['neuve'], 'hauts de france')
    p = N.address_parts('8 R. DE LA PORTE D’EAU, DUNKERQUE, Nord', 'France')        # department -> region
    assert p['street_type'] == 'rue' and p['street_tokens'] == ['porte', 'eau'] and p['state_canonical'] == 'hauts de france'
    p = N.address_parts('8 BIS RUE PAUL BERT, SAINT-NAZAIRE', 'France')              # region inferred from city
    assert p['house_numbers'] == ['8b'] and p['house_base'] == ['8'] and p['state_canonical'] == 'pays de la loire'
    p = N.address_parts('N°34 AVE DES CARAVELLES, Lège-Cap-Ferret, Gironde', 'France')
    assert p['house_numbers'] == ['34'] and p['street_type'] == 'av' and p['city_candidates'] == ['lege cap ferret']
    p = N.address_parts("TOURCOING, 35 RUE D' ATHENES, Nord", 'France')
    assert p['house_numbers'] == ['35'] and p['street_tokens'] == ['athenes'] and 'tourcoing' in p['city_candidates']
    p = N.address_parts('ST.-NAZAIRE, 4 ter AV. Foch', 'France')
    assert p['house_numbers'] == ['4t'] and 'saint nazaire' in p['city_candidates']
    a = N.address_key('11 Avenue de la Tramontane Pyla, La Teste-de-Buch, Nouvelle-Aquitaine', 'France')
    b = N.address_key('11 AV. DE LA TRAMONTANE PYLA, LA TESTE-DE-BUCH, Gironde', 'France')
    assert a == b


def test_generic_country_address():
    p = N.address_parts('12 bis Hauptstrasse, Berlin', 'Germany')
    assert p['house_numbers'] == ['12b'] and 'berlin' in p['city_candidates']
    p = N.address_parts('1804A Main Street, Austin, Texas', 'Mars')
    assert p['state_canonical'] == 'tx' and p['street_type'] == 'st'


def test_never_raises_on_garbage():
    for s in [None, '', ' , , ', '####', 'NULL', '‍‌', '𝔘𝔫𝔦𝔠𝔬𝔡𝔢 123', 'ॐ', '١٢٣ شارع']:
        for c in ['US', 'India', 'France', 'Brazil', None]:
            N.normalize_name(s, c); N.name_core(s, c); N.normalize_address(s, c); N.address_parts(s, c)
            N.transliterate(s)


# ---------------------------------------------------------------- added in the resumed run
def test_last_state_wins_and_earlier_becomes_city():
    p = N.address_parts('1600 K St, Washington, DC', 'US')
    assert p['state_canonical'] == 'dc' and 'washington' in p['city_candidates']
    p = N.address_parts('12 Main St, Washington, PA', 'US')
    assert p['state_canonical'] == 'pa'


def test_glued_similarity_domain_forms():
    assert N.glued_similarity('Unified Pinnacle Esports L.L.C.', 'unifiedpinnacleesports.com', 'US') == 1.0
    assert N.glued_similarity('Woven (India) Technologies Private Limited', 'wovenindiatechnologies.com', 'India') == 0.9
    assert N.glued_similarity('Shramik (India) Chit Pvt. Ltd.', 'shr\u00e1mikindiachit.com', 'India') >= 0.9   # accent in domain
    assert N.glued_similarity('Corner Tattoo LLC', 'tattoocorner.com', 'US') >= 0.8                       # reordered concat
    assert N.glued_similarity('Katz Seafood', 'Belomirabrix', 'US') == 0.0


def test_name_variants_consistent():
    for n, c in [('Girwa Projects Private Limited', 'India'), ('>> Netra Limited Private Technologies', 'India'),
                 ('Compagnons  Sport S.A.R.L.', 'France'), ('unifiedpinnacleesports.com', 'US')]:
        v = N.name_variants(n, c)
        assert {k: v[k] for k in ('norm', 'core', 'key', 'glued')} == {
            'norm': N.normalize_name(n, c), 'core': N.name_core(n, c), 'key': N.name_key(n, c), 'glued': N.name_glued(n, c)}


def test_address_all_consistent():
    for a, c in [('221 Seneca Street, Corning, NY', 'US'), ('Hn 352 Niwasa, Udaipur, RJ, Girwa', 'India'),
                 ('8 BIS RUE PAUL BERT, SAINT-NAZAIRE', 'France'), ('', 'Brazil')]:
        na, p = N.address_all(a, c)
        assert na == N.normalize_address(a, c) and p == N.address_parts(a, c)


def test_pmb_and_leading_zeros():
    assert N.normalize_address('002216 DUDLEY LANE, PMB 4593, BURLESON, TX', 'US') == '2216 dudley ln burleson tx'


def test_alias_markers_after_part_is_primary():
    for m in ['Fluxviox Formerly Veterans Association VI', 'Nylazeph formerly known as Veterans Association VI',
              'Jaxaria fka Veterans Association VI', 'Lumveo née Veterans Association VI', 'Vioiri aka Veterans Association VI',
              'Zetaflux t/a Veterans Association VI', 'Ariakelo trading as Veterans Association VI']:
        assert N.name_core(m, 'US') == 'veterans association vi', m


def test_leet_edges_and_l_for_i():
    assert N.normalize_name('5ervices 8usiness 0ffice', 'US') == 'services business office'
    assert N.normalize_name('Smile Denta1 Care', 'US') == 'smile dental care'
    assert N.normalize_name('Acme lnc', 'US') == 'acme inc'
    assert N.normalize_name('Tata lndia lnvestments', 'India') == 'tata india investments'
    assert N.normalize_name('3rd Street 4th Ave 7 Eleven', 'US') == '3rd street 4th ave 7 eleven'   # ordinals untouched


def test_distinct_drops_injected_generic_words():
    assert N.name_variants('Young Services', 'US')['distinct'] == 'young'
    assert N.name_variants('Girwa Private Limited Center', 'India')['distinct'] == 'girwa'
    assert N.name_variants('Services', 'US')['distinct'] == 'services'           # never empty


NATIVE_STATES = {'महाराष्ट्र': 'maharashtra', 'दिल्ली': 'delhi', 'उत्तर प्रदेश': 'uttar pradesh', 'ಕರ್ನಾಟಕ': 'karnataka',
                 'தமிழ்நாடு': 'tamil nadu', 'পশ্চিমবঙ্গ': 'west bengal', 'ગુજરાત': 'gujarat', 'తెలంగాణ': 'telangana',
                 'हरियाणा': 'haryana', 'राजस्थान': 'rajasthan', 'കേരളം': 'kerala', 'बिहार': 'bihar',
                 'मध्य प्रदेश': 'madhya pradesh', 'ఆంధ్రప్రదేశ్': 'andhra pradesh', 'ਪੰਜਾਬ': 'punjab', 'ଓଡ଼ିଶା': 'odisha'}


def test_all_observed_native_states_resolve():
    # the 16 native-script state strings seen in train/test S2/S3 addresses
    for native, canon in NATIVE_STATES.items():
        assert N.address_parts('12 MG Road, ' + native, 'India')['state_canonical'] == canon, native
        assert N.address_parts(native + ', 12 MG Road', 'India')['state_canonical'] == canon, native


# ---------------------------------------------------------------- France audit fixes (work/reports/diag/norm_audit.md)
def test_france_ch_is_chemin_not_chaussee():
    for a in ['12 Ch. des Vignes, Lille', '12 CH DES VIGNES, Lille', '12 Chemin des Vignes, Lille', '12 Chem. des Vignes, Lille']:
        p = N.address_parts(a, 'France')
        assert p['street_type'] == 'ch' and p['street_tokens'] == ['vignes'], a
    assert N.address_parts('5 Chaussee Brunehaut, Lille', 'France')['street_type'] == 'chau'
    assert N.address_key('12 Ch. des Vignes, Lille', 'France') == N.address_key('12 Chemin des Vignes, Lille', 'France')


def test_france_apartment_is_a_unit_word():
    ref = N.address_all('12 Rue X, Apt 8, Lille', 'France')
    for a in ['12 Rue X, Appt 8, Lille', '12 Rue X, Appartement 8, Lille', '12 Rue X, Appart. 8, Lille', '12 Rue X, APPT 8, Lille']:
        norm, p = N.address_all(a, 'France')
        assert norm == ref[0] == '12 rue x apt 8 lille', a
        assert p['house_numbers'] == ['12'] and p['unit_numbers'] == ['8'] and p['city_candidates'] == ['lille'], a


def test_france_single_letter_suffix_glued_like_bis():
    for a, hn in [('12 B Rue X, Lille', '12b'), ('12B Rue X, Lille', '12b'), ('12 bis Rue X, Lille', '12b'),
                  ('12 T Rue X, Lille', '12t'), ('12 ter Rue X, Lille', '12t'), ('12 a Rue X, Lille', '12a'),
                  ('12 C, Rue X, Lille', '12c'), ('Rue X 12 d, Lille', '12d')]:
        assert N.address_parts(a, 'France')['house_numbers'] == [hn], a
    assert N.address_key('12 B Rue X, Lille', 'France') == N.address_key('12bis Rue X, Lille', 'France')
    # elision / words are not suffixes; other countries unchanged ('500 C St' is C Street)
    assert N.address_parts("12 D'Arc, Lille", 'France')['house_numbers'] == ['12']
    assert N.address_parts('12 Bd Victor Hugo, Lille', 'France')['house_numbers'] == ['12']
    assert N.normalize_address('500 C St, Washington, DC', 'US').startswith('500 c st')


def test_france_alle_is_allee():
    a = N.address_parts('4 Alle des Pins, Pessac', 'France')
    b = N.address_parts('4 Allee des Pins, Pessac', 'France')
    assert a['street_type'] == b['street_type'] == 'all' and a['street_tokens'] == b['street_tokens'] == ['pins']
