# -*- coding: utf-8 -*-
"""normalize.py -- reference text normalization for Business Entity Resolution
(Amazon ML Challenge 2026; countries US / India / France(test-only) / ANY unknown country -> generic rules).

Pure standard library (re, unicodedata). No network, no external data. All maps are hand-written domain
knowledge; the optional native-token dictionary is LEARNED FROM THE PROVIDED TRAINING PAIRS (see
load_native_dict / learn_native_dict).

Public API
----------
transliterate(text)                  -> str   Indic scripts (9 blocks) -> Latin; other chars unchanged
normalize_name(name, country)        -> str   cleaned, lowercase, legal forms canonicalised (kept)
name_core(name, country)             -> str   normalize_name minus legal forms / honorifics / fillers
name_key(name, country)              -> str   sorted unique core tokens (token-shuffle invariant)
name_glued(name, country)            -> str   core with all spaces removed (domain-form comparisons)
name_flags(name)                     -> dict  noise indicators (domain, handle, dba, phone, bracket, native...)
split_dba(name)                      -> (primary, alias|None)
split_concat(glued, ref_tokens)      -> str   segment 'allenapexenergetics' using the OTHER record's tokens
normalize_address(addr, country)     -> str   cleaned, canonical street types / state, no NULL tokens
address_key(addr, country)           -> str   sorted unique address tokens (component-order invariant)
address_parts(addr, country)         -> dict  house_numbers, house_base, street_tokens, street_type,
                                              unit_numbers, designated_numbers, postcodes, city_candidates,
                                              state_canonical, landmarks, all_numbers
phonetic_key(token)                  -> str   ph/f, w/v, z/j, c/k/s, aspirates, y/i, doubled letters
skeleton_key(token)                  -> str   phonetic_key without non-initial vowels
load_native_dict(path)               -> int   load learned native->latin token dictionary (TSV)
enable_snapping(latin_counts, ...)   -> None  optional fallback: snap unknown transliterations to frequent vocab
add_normalized_columns(df, n_jobs)   -> polars frame with the CONTRACT columns (implemented in normalize_frame.py)
"""
import re
import unicodedata

__all__ = ['transliterate', 'normalize_name', 'name_core', 'name_key', 'name_glued', 'name_flags', 'split_dba',
           'split_concat', 'normalize_address', 'address_key', 'address_parts', 'name_variants', 'address_all', 'glued_similarity', 'phonetic_key', 'skeleton_key',
           'load_native_dict', 'learn_native_dict', 'enable_snapping', 'country_key', 'fold', 'add_normalized_columns']

# =====================================================================================================
# 0. country handling (open set)
# =====================================================================================================
_COUNTRY_ALIASES = {
    'us': 'us', 'usa': 'us', 'u.s.': 'us', 'u.s.a.': 'us', 'united states': 'us', 'united states of america': 'us',
    'america': 'us',
    'india': 'india', 'in': 'india', 'ind': 'india', 'bharat': 'india',
    'france': 'france', 'fr': 'france', 'fra': 'france', 'republique francaise': 'france',
}


def country_key(country):
    """Map any country label to 'us' | 'india' | 'france' | 'generic'. Never raises."""
    c = fold(str(country or '')).strip()
    return _COUNTRY_ALIASES.get(c, 'generic')


# =====================================================================================================
# 1. Indic -> Latin transliteration (dependency-free; ISCII-parallel Unicode layout)
# =====================================================================================================
_BLOCKS = {0x0900: 'deva', 0x0980: 'beng', 0x0A00: 'guru', 0x0A80: 'gujr', 0x0B00: 'orya',
           0x0B80: 'taml', 0x0C00: 'telu', 0x0C80: 'knda', 0x0D00: 'mlym'}
_NORTH = {'deva', 'beng', 'guru', 'gujr', 'orya'}
_DRAVIDIAN = {'taml', 'telu', 'knda', 'mlym'}
_CONS = {0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'n', 0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh',
         0x1E: 'n', 0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n', 0x24: 't', 0x25: 'th', 0x26: 'd',
         0x27: 'dh', 0x28: 'n', 0x29: 'n', 0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm', 0x2F: 'y',
         0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'l', 0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's',
         0x39: 'h', 0x58: 'k', 0x59: 'kh', 0x5A: 'g', 0x5B: 'z', 0x5C: 'r', 0x5D: 'rh', 0x5E: 'f', 0x5F: 'y'}
_NUKTA = {0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x1C: 'z', 0x1D: 'z', 0x21: 'r', 0x22: 'rh', 0x2B: 'f', 0x2F: 'y',
          0x38: 'sh', 0x32: 'l', 0x2C: 'b', 0x1A: 'ch'}
_VOW = {0x04: 'a', 0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u', 0x0B: 'ri', 0x0C: 'li',
        0x0D: 'e', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au', 0x60: 'ri', 0x61: 'li'}
_MATRA = {0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x43: 'ri', 0x44: 'ri', 0x45: 'a', 0x46: 'e',
          0x47: 'e', 0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au', 0x4E: 'e', 0x4F: 'aw',
          0x62: 'li', 0x63: 'li', 0x57: 'au', 0x56: 'ai', 0x55: 'e'}
_OVR = {
    ('taml', 0x1A): ('C', 's'),
    ('beng', 0x4E): ('C', 't'), ('beng', 0x70): ('C', 'r'), ('beng', 0x71): ('C', 'v'),
    ('orya', 0x71): ('C', 'v'),
    ('guru', 0x70): ('N', 'n'), ('guru', 0x71): ('X', ''), ('guru', 0x72): ('V', ''), ('guru', 0x73): ('V', 'u'),
    ('guru', 0x74): ('X', ' ik onkar '), ('guru', 0x75): ('X', ''),
    ('mlym', 0x7A): ('C!', 'n'), ('mlym', 0x7B): ('C!', 'n'), ('mlym', 0x7C): ('C!', 'r'), ('mlym', 0x7D): ('C!', 'l'),
    ('mlym', 0x7E): ('C!', 'l'), ('mlym', 0x7F): ('C!', 'k'), ('mlym', 0x54): ('C!', 'm'), ('mlym', 0x55): ('C!', 'y'),
    ('mlym', 0x56): ('C!', 'l'), ('mlym', 0x4E): ('C!', 'r'),
    ('telu', 0x04): ('N', 'M'), ('telu', 0x00): ('N', 'n'),
    ('telu', 0x58): ('C', 'ts'), ('telu', 0x59): ('C', 'dz'), ('telu', 0x5A): ('C', 'r'),
    ('knda', 0x5E): ('C', 'l'), ('knda', 0x5D): ('C!', 'n'),
    ('deva', 0x72): ('V', 'a'), ('deva', 0x73): ('V', 'e'), ('deva', 0x74): ('V', 'e'), ('deva', 0x75): ('V', 'aw'),
    ('deva', 0x76): ('V', 'ue'), ('deva', 0x77): ('V', 'ue'), ('deva', 0x79): ('C', 'z'), ('deva', 0x7A): ('C', 'y'),
    ('deva', 0x7B): ('C', 'g'), ('deva', 0x7C): ('C', 'j'), ('deva', 0x7E): ('C', 'd'), ('deva', 0x7F): ('C', 'b'),
}
_ZW = {'‌', '‍', '­', '﻿', '​'}
_HAS_INDIC = re.compile('[ऀ-ൿ]')
_INDIC_RUN = re.compile('[ऀ-ൿ‌‍]+')
_LABIAL = {'p', 'b', 'bh', 'm'}
_KEEP_AFTER_CLUSTER = {'r', 'y', 'v', 'n', 'm', 'l'}


def _classify(script, off):
    r = _OVR.get((script, off))
    if r is not None:
        return r
    if off in _CONS and (off < 0x58 or script in _NORTH):
        return ('C', _CONS[off])
    if off in _VOW:
        return ('V', _VOW[off])
    if off in _MATRA:
        return ('M', _MATRA[off])
    if off == 0x4D:
        return ('H', '')
    if off == 0x3C:
        return ('K', '')
    if off in (0x00, 0x01):
        return ('N', 'n')
    if off == 0x02:
        return ('N', 'M')
    if off == 0x03:
        return ('N', 'h')
    if 0x66 <= off <= 0x6F:
        return ('D', str(off - 0x66))
    if off in (0x64, 0x65):
        return ('X', ' ')
    if off == 0x50:
        return ('X', 'om')
    return ('X', '')


def _aksharas(chars, script):
    aks = []
    i, n = 0, len(chars)
    while i < n:
        cls, lat, off = chars[i]
        if cls in ('C', 'C!'):
            cons = lat
            j = i + 1
            if j < n and chars[j][0] == 'K':
                cons = _NUKTA.get(off, cons); j += 1
            prev = aks[-1] if aks else None
            ak = {'c': cons, 'v': None, 'inh': cls == 'C', 'virama': cls == 'C!', 'off': off, 'mods': '',
                  'cluster_prev': bool(prev) and prev['virama'] and prev['c'] is not None and prev['v'] is None}
            if j < n and chars[j][0] == 'M':
                ak['v'] = chars[j][1]; ak['inh'] = False; j += 1
            elif j < n and chars[j][0] == 'H':
                ak['inh'] = False; ak['virama'] = True; j += 1
            while j < n and chars[j][0] == 'N':
                ak['mods'] += chars[j][1]; j += 1
            aks.append(ak); i = j
        elif cls == 'V':
            ak = {'c': None, 'v': lat, 'inh': False, 'virama': False, 'off': off, 'mods': '', 'cluster_prev': False}
            j = i + 1
            while j < n and chars[j][0] in ('N', 'M'):
                if chars[j][0] == 'M':
                    ak['v'] = (ak['v'] or '') + chars[j][1]
                else:
                    ak['mods'] += chars[j][1]
                j += 1
            aks.append(ak); i = j
        elif cls == 'M':
            aks.append({'c': None, 'v': lat, 'inh': False, 'virama': False, 'off': off, 'mods': '', 'cluster_prev': False})
            i += 1
        elif cls == 'N':
            if aks:
                aks[-1]['mods'] += lat
            else:
                aks.append({'c': None, 'v': None, 'inh': False, 'virama': False, 'off': off, 'mods': lat, 'cluster_prev': False})
            i += 1
        else:
            aks.append({'c': None, 'v': None, 'inh': False, 'virama': False, 'off': off, 'mods': '', 'lit': lat,
                        'cluster_prev': False})
            i += 1
    return aks


def _schwa(aks, script, policy):
    if policy == 'auto':
        policy = 'hindi' if script in _NORTH else 'keep'
    n = len(aks)
    keep = [a['inh'] for a in aks]
    if policy == 'keep' or n == 0:
        return keep
    if policy == 'drop':
        return [k and idx == 0 for idx, k in enumerate(keep)]
    if script not in _NORTH:
        return keep
    last = n - 1
    while last >= 0 and aks[last]['c'] is None and aks[last].get('lit') is not None:
        last -= 1
    if last >= 1 and keep[last] and not aks[last]['mods']:
        if not aks[last]['cluster_prev'] or aks[last]['c'] not in _KEEP_AFTER_CLUSTER:
            keep[last] = False
    if policy == 'hindi':
        def has_vowel(k):
            return aks[k]['v'] is not None or keep[k]
        for k in range(last - 1, 0, -1):
            a = aks[k]
            if not keep[k] or a['mods']:
                continue
            q = k + 1
            while q < n and aks[q]['virama'] and aks[q]['c'] is not None and aks[q]['v'] is None:
                q += 1
            if q >= n or aks[q]['c'] is None or not has_vowel(q):
                continue
            p = k - 1
            while p >= 0 and aks[p]['virama'] and aks[p]['v'] is None and aks[p]['c'] is not None:
                p -= 1
            left = (k - p) + (1 if p >= 0 and aks[p]['mods'] else 0)
            if left + (q - k) > 3:
                continue
            if p >= 0 and has_vowel(p) and (p == k - 1 or a['cluster_prev']):
                keep[k] = False
    return keep


def _render(aks, keep, script):
    out = []
    n = len(aks)
    for k, a in enumerate(aks):
        if a.get('lit') is not None:
            out.append(a['lit']); continue
        c = a['c'] or ''
        if script == 'mlym' and a['off'] == 0x31 and k > 0 and aks[k - 1]['virama'] and aks[k - 1]['off'] in (0x31, 0x28):
            c = 't'                                     # Malayalam റ്റ -> tt, ന്റ -> nt
            if aks[k - 1]['off'] == 0x31 and out and out[-1].endswith('r'):
                out[-1] = out[-1][:-1] + 't'
        if script == 'taml' and a['off'] == 0x31 and k > 0 and aks[k - 1]['virama'] and aks[k - 1]['off'] == 0x31:
            if out and out[-1].endswith('r'):
                out[-1] = out[-1][:-1] + 't'            # Tamil ற்ற -> tr
        if script == 'taml' and k > 0 and aks[k - 1]['mods'].endswith('h') and aks[k - 1]['c'] is None and a['off'] == 0x2A:
            out[-1] = out[-1][:-1]; c = 'f'             # Tamil aytham + pa -> f
        v = a['v'] if a['v'] is not None else ('a' if keep[k] else '')
        mods = ''
        for m in a['mods']:
            if m == 'M':                                # anusvara: m before labials / Dravidian word-final, else n
                nxt = aks[k + 1] if k + 1 < n else None
                if nxt is not None and nxt['c'] in _LABIAL:
                    mods += 'm'
                elif nxt is None or nxt['c'] is None:
                    mods += 'm' if script in _DRAVIDIAN else 'n'
                else:
                    mods += 'n'
            else:
                mods += m
        out.append(c + v + mods)
    return ''.join(out)


def _translit_run(run, schwa='auto', ph_to_f=True):
    run = ''.join(ch for ch in run if ch not in _ZW)
    if not run:
        return ''
    o = ord(run[0])
    script = _BLOCKS.get(o & ~0x7F)
    chars = []
    for ch in run:
        oo = ord(ch)
        chars.append(_classify(script, oo - (oo & ~0x7F)) + (oo - (oo & ~0x7F),))
    s = _render(_aksharas(chars, script), _schwa(_aksharas(chars, script), script, schwa), script)
    return s.replace('ph', 'f') if ph_to_f else s


_NATIVE_DICT = {}         # native token -> latin token (learned from TRAIN pairs; load_native_dict)
_TL_CACHE = {}
_SNAP = None              # (tokens, keys, thr) for optional vocabulary snapping


def _split_script_runs(run):
    """Split an Indic run where the script block changes (mixed-script tokens)."""
    parts, cur, cur_b = [], '', None
    for ch in run:
        if ch in _ZW:
            cur += ch; continue
        b = ord(ch) & ~0x7F
        if cur_b is not None and b != cur_b and not (0x0964 <= ord(ch) <= 0x0965):
            parts.append(cur); cur = ''
        cur += ch; cur_b = b
    if cur:
        parts.append(cur)
    return parts


def _translit_token(tok, use_dict=True):
    key = ''.join(ch for ch in tok if ch not in _ZW)
    if use_dict and key in _NATIVE_DICT:
        return _NATIVE_DICT[key]
    v = _TL_CACHE.get(key)
    if v is None:
        v = ''.join(_translit_run(p) for p in _split_script_runs(key))
        if _SNAP is not None:
            v = _snap(v)
        if len(_TL_CACHE) < 500000:
            _TL_CACHE[key] = v
    return v


def transliterate(text, use_dict=True):
    """Indic (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) -> Latin.
    Uses the learned native-token dictionary first (if loaded), then rules. Non-Indic characters (incl.
    Latin accents) are returned unchanged -- fold accents AFTER this step (NFD would destroy matras)."""
    if not text or not _HAS_INDIC.search(text):
        return text or ''
    text = unicodedata.normalize('NFC', text)
    out = _INDIC_RUN.sub(lambda m: ' ' + _translit_token(m.group(0), use_dict) + ' ', text)
    return ' '.join(out.split())


def load_native_dict(path, min_support=2):
    """Load 'native<TAB>latin<TAB>support<TAB>share' TSV (header line). Returns #entries loaded."""
    n = 0
    with open(path, encoding='utf-8') as f:
        next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            sup = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else min_support
            if sup >= min_support:
                _NATIVE_DICT[parts[0]] = parts[1]; n += 1
    _TL_CACHE.clear()
    _register_native_states()
    return n


def learn_native_dict(pairs, min_support=2, min_share=0.4, sim_thr=0.55):
    """Learn native->latin token map from TRAIN true-link (latin_name, native_name) pairs.
    Alignment-vote: each native token is aligned to the most similar (phonetic-key Jaro-Winkler of its
    rule transliteration) unused Latin token of the paired S1 name; falls back to the Dice-best
    co-occurring token (abbreviations like प्रा.->pvt). Returns dict native -> (latin, support, share)."""
    from collections import Counter, defaultdict
    tok_re = re.compile(r"[0-9a-zऀ-ൿ‌‍]+")
    cnt_n, cnt_l, C = Counter(), Counter(), defaultdict(Counter)
    prepared = []
    for latin, native in pairs:
        N = [''.join(c for c in t if c not in _ZW) for t in tok_re.findall(unicodedata.normalize('NFC', native).lower())
             if _HAS_INDIC.search(t)]
        L = re.findall(r'[0-9a-z]+', fold(latin))
        if not N or not L:
            continue
        prepared.append((N, L))
        for t in set(N):
            cnt_n[t] += 1
            for l in set(L):
                C[t][l] += 1
        for l in set(L):
            cnt_l[l] += 1
    dice_best = {}
    for t, cn in cnt_n.items():
        best = max(C[t].items(), key=lambda kv: 2 * kv[1] / (cn + cnt_l[kv[0]]))
        dice_best[t] = best[0]
    A = defaultdict(Counter)
    for N, L in prepared:
        lk = [(l, phonetic_key(l)) for l in L]
        used = set()
        for t in N:
            k = phonetic_key(re.sub(r'[^a-z0-9]', '', _translit_token(t, use_dict=False)))
            best, bs = None, 0.0
            for l, lkey in lk:
                if l in used:
                    continue
                s = _jaro_winkler(k, lkey)
                if s > bs:
                    best, bs = l, s
            if best is not None and bs >= sim_thr:
                A[t][best] += 1; used.add(best)
            elif dice_best.get(t) in L:
                A[t][dice_best[t]] += 1
    out = {}
    for t, ctr in A.items():
        l, c = ctr.most_common(1)[0]
        share = min(1.0, c / cnt_n[t])
        if c >= min_support and share >= min_share:
            out[t] = (l, c, round(share, 3))
    return out


def _jaro_winkler(a, b):
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.normalized_similarity(a, b)
    except Exception:                                   # tiny pure-python fallback
        if a == b:
            return 1.0
        la, lb = len(a), len(b)
        if not la or not lb:
            return 0.0
        md = max(la, lb) // 2 - 1
        ma, mb = [False] * la, [False] * lb
        m = 0
        for i in range(la):
            for j in range(max(0, i - md), min(lb, i + md + 1)):
                if not mb[j] and a[i] == b[j]:
                    ma[i] = mb[j] = True; m += 1; break
        if not m:
            return 0.0
        t, j = 0, 0
        for i in range(la):
            if ma[i]:
                while not mb[j]:
                    j += 1
                t += a[i] != b[j]; j += 1
        jaro = (m / la + m / lb + (m - t / 2) / m) / 3
        p = 0
        for x, y in zip(a[:4], b[:4]):
            if x != y:
                break
            p += 1
        return jaro + p * 0.1 * (1 - jaro)


def _lev_sim(a, b):
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1 - prev[lb] / max(la, lb)


def enable_snapping(latin_token_counts, min_count=500, thr=0.7):
    """Optional fallback for native tokens NOT in the learned dictionary: snap the rule transliteration
    to the most similar FREQUENT Latin token (phonetic key, normalized Levenshtein >= thr, same first
    letter or both vowels, |len diff|<=3). Measured on held-out dictionary types: 62.7% correct, 3.1%
    wrong, 34% unchanged (min_count=500, thr=0.7). Pass None to disable."""
    global _SNAP
    _TL_CACHE.clear()
    if latin_token_counts is None:
        _SNAP = None; return
    toks = [t for t, c in latin_token_counts.items() if c >= min_count and t.isalpha()]
    _SNAP = (toks, [phonetic_key(t) for t in toks], thr)
    _register_native_states()


def _snap(v):
    toks, keys, thr = _SNAP
    out = []
    for w in v.split():
        k = phonetic_key(w)
        best, bs = None, thr
        for t, kk in zip(toks, keys):
            if kk[:1] != k[:1] and not (kk[:1] in 'aeiou' and k[:1] in 'aeiou'):
                continue
            if abs(len(kk) - len(k)) > 3:
                continue
            s = _lev_sim(k, kk)
            if s > bs:
                best, bs = t, s
        out.append(best or w)
    return ' '.join(out)


# =====================================================================================================
# 2. generic helpers
# =====================================================================================================
def fold(s):
    """Lowercase + strip Latin diacritics (NFKD). Call only AFTER transliterate()."""
    if not s:
        return ''
    if s.isascii():                                   # fast path (~95% of strings)
        return s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch) or _HAS_INDIC.match(ch))
    return s.lower().replace('ß', 'ss').replace('æ', 'ae').replace('œ', 'oe').replace('ø', 'o').replace('ł', 'l').replace('đ', 'd')


_QUOTES = str.maketrans({'’': "'", '‘': "'", '‛': "'", '`': "'", '´': "'", '′': "'",
                         '“': '"', '”': '"', '–': '-', '—': '-', '−': '-', ' ': ' ',
                         'º': '°', '＃': '#'})


def _prep(s):
    s = unicodedata.normalize('NFKC', s or '').translate(_QUOTES)
    return transliterate(s)


def phonetic_key(tok):
    """Cheap phonetic key for Latin tokens (after transliteration): ch/sh kept, ph->f, w->v, q->k, x->ks,
    z->j, c->s before e/i/y else k, aspirates dropped (kh->k, th->t, bh->b ...), y->i, doubled letters collapsed."""
    t = tok.replace('ch', 'C').replace('sh', 'S').replace('ph', 'f').replace('w', 'v').replace('q', 'k')
    t = t.replace('x', 'ks').replace('z', 'j')
    t = re.sub(r'c(?=[eiy])', 's', t).replace('ck', 'k').replace('c', 'k')
    t = re.sub(r'([kgtdpbj])h', r'\1', t).replace('C', 'c').replace('S', 's').replace('y', 'i')
    return re.sub(r'(.)\1+', r'\1', t)


def skeleton_key(tok):
    t = phonetic_key(tok)
    return t[:1] + re.sub(r'[aeiou]', '', t[1:])


def _dedup_consecutive(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


_PM_FIRSTS = {}


def _phrase_replace(toks, phrase_map, max_len=4):
    """Longest-match replacement of token sequences. phrase_map: tuple(tokens) -> replacement str ('' = drop)."""
    entry = _PM_FIRSTS.get(id(phrase_map))
    if entry is None or entry[0] is not phrase_map:
        entry = _PM_FIRSTS[id(phrase_map)] = (phrase_map, {k[0] for k in phrase_map if k})
    firsts = entry[1]
    out, i, n = [], 0, len(toks)
    while i < n:
        if toks[i] not in firsts:
            out.append(toks[i]); i += 1
            continue
        for L in range(min(max_len, n - i), 0, -1):
            rep = phrase_map.get(tuple(toks[i:i + L]))
            if rep is not None:
                if rep:
                    out.extend(rep.split())
                i += L
                break
        else:
            out.append(toks[i]); i += 1
    return out


def _pm(d):
    """{canonical: [variants...]} -> {tuple(variant tokens): canonical}"""
    m = {}
    for canon, variants in d.items():
        for v in variants + [canon]:
            m[tuple(v.split())] = canon
    return m


# =====================================================================================================
# 3. NAME maps (hand-written domain knowledge)
# =====================================================================================================
LEGAL_FORMS = {
    'us': {
        'llc': ['l l c', 'limited liability company', 'limited liability co', 'l.l.c'],
        'inc': ['incorporated', 'incorporation', 'inc'],
        'corp': ['corporation', 'corpn', 'corp'],
        'co': ['company', 'comp', 'compan'],
        'ltd': ['limited', 'ltd'],
        'lp': ['l p', 'limited partnership'],
        'llp': ['l l p', 'limited liability partnership'],
        'lllp': ['l l l p'],
        'pc': ['p c', 'professional corporation', 'prof corp'],
        'pllc': ['p l l c', 'professional limited liability company'],
        'pa': ['p a', 'professional association'],
        'plc': ['p l c'],
        'na': ['n a', 'national association'],
        'dba': ['d b a'],
    },
    'india': {
        'pvt': ['private', 'pvt', 'pte', 'prvt', 'pvt ltd'.split()[0]],
        'ltd': ['limited', 'ltd', 'ltd.', 'lmt', 'limted'],
        'llp': ['l l p', 'limited liability partnership'],
        'opc': ['o p c', 'one person company'],
        'co': ['company', 'comp'],
        'corp': ['corporation', 'corpn'],
        'inc': ['incorporated'],
    },
    'france': {
        'sarl': ['s a r l', 'societe a responsabilite limitee'],
        'sas': ['s a s', 'societe par actions simplifiee'],
        'sasu': ['s a s u', 'societe par actions simplifiee unipersonnelle'],
        'eurl': ['e u r l', 'entreprise unipersonnelle a responsabilite limitee'],
        'sa': ['s a', 'societe anonyme'],
        'sci': ['s c i', 'societe civile immobiliere'],
        'snc': ['s n c', 'societe en nom collectif'],
        'scp': ['s c p'], 'scm': ['s c m'], 'scop': ['s c o p'], 'selarl': ['s e l a r l'], 'gie': ['g i e'],
        'ei': ['e i', 'entreprise individuelle'],
        'eirl': ['e i r l'],
        'cie': ['compagnie', 'compagnies', 'cie'],
        'ets': ['etablissements', 'etablissement', 'ets', 'etabl', 'etab'],
        'fils': ['et fils', 'and fils'],
        'freres': ['et freres', 'and freres', 'frere'],
    },
}
# generic = union (first writer wins on conflicts) -> used for unknown countries and for cross-country noise
LEGAL_FORMS['generic'] = {}
for _c in ('us', 'india', 'france'):
    for _k, _v in LEGAL_FORMS[_c].items():
        LEGAL_FORMS['generic'].setdefault(_k, [])
        LEGAL_FORMS['generic'][_k] = sorted(set(LEGAL_FORMS['generic'][_k]) | set(_v))
# India: private/limited are always legal words; US: 'limited' is legal (Ltd), 'private' is NOT stripped
_LEGAL_PM = {c: _pm(d) for c, d in LEGAL_FORMS.items()}
LEGAL_TOKENS = {c: set(d.keys()) for c, d in LEGAL_FORMS.items()}
LEGAL_TOKENS_ALL = set().union(*LEGAL_TOKENS.values()) - {'na'}   # 'na' too ambiguous outside US banks

HONORIFICS = {'sri', 'shri', 'sree', 'shree', 'smt', 'dr', 'mr', 'mrs', 'ms', 'messrs', 'm s', 'mess', 'kumari', 'km',
              'the', 'thiru', 'tmt', 'selvi'}   # sree/shree: also injected (42/8 times) but real in S1 -> core-only drop
# measured: 'sri/shri/smt' never occur in S1 (0.00%) but in ~4% of India S2/S3 -> always drop
ALWAYS_DROP_NAME = {'sri', 'shri', 'smt', 'm/s'}
FR_ARTICLES = {'le', 'la', 'les', 'l', 'des', 'du', 'de', 'd', 'au', 'aux'}
NAME_FILLERS = {'and', 'et', 'of', 'the', 'und', 'y'}
COUNTRY_WORDS = {'india', 'france', 'usa', 'us', 'america', 'indian', 'bharat'}

# Alias markers. Measured on mini-train true links: the S1 name is ALWAYS the part AFTER the marker
# (dba 643/643, formerly 101/101, trading as 59/59, aka 51/51, nee 43/43, t/a 41/41, fka 37/37, formerly known as 32/32).
_DBA_RE = re.compile(r"\s*(?:\b[dD]\s*\.?\s*[bB]\s*\.?\s*[aA]\b\.?\s*:?|\bdoing\s+business\s+as\b:?|\btrading\s+as\b:?|"
                     r"\boperating\s+as\b:?|\bt/a\b:?|\bo/a\b:?|\b(?:also\s+|formerly\s+|previously\s+)?known\s+as\b:?|"
                     r"\ba\.?k\.?a\.?\b:?|\bformerly\b:?|\bf/k/a\b:?|\bf\.?k\.?a\b\.?:?|\bn[eé]e\b:?)\s*", re.I)
_PHONE_RE = re.compile(r"\s*[-|:]?\s*[\[(]?\s*\+?\d[\d\s\-().]{7,}\d\s*[\])]?")
_ID_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.I)
_URL_TOKEN_RE = re.compile(r"(?i)\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*?)"
                           r"\.(?:com|net|org|co\.in|org\.in|net\.in|in|co|biz|info|us|fr|io|co\.uk|uk|eu|asso\.fr)\b")
_HANDLE_RE = re.compile(r"(?<![\w])[@#]\s*([A-Za-z0-9_][\w.]*)")
_DOTTED_ABBR_RE = re.compile(r"\b(?:[A-Za-z]\s?\.\s?){2,}(?:[A-Za-z]\b\.?)?")
_LEET = {'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b'}
# digit look-alikes injected into words (measured added tokens: 5ervices, 8usiness, 0ffice, denta1, medica1, que5t):
#   inside a word; at word start before >=3 letters (so '3rd', '4th' are safe); at word end after >=2 letters
_LEET_RE = re.compile(r"(?<=[a-z])[0134578](?=[a-z])|(?<![0-9a-z])[034578](?=[a-z]{3})|(?<=[a-z]{2})[01](?![0-9a-z])")
# capital I typed as lowercase L at word start ('lnc', 'lndia', 'lnvestments'): 'ln' never starts an English word
_L_FOR_I_RE = re.compile(r"(?<![0-9a-z])ln(?=[a-z])")
# words the generator INJECTS as generic substitutes (most-added core tokens on true links: center 625/694,
# services 439/470, service 215/255, partners 188/181 for US/India); dropped only in name_variants()['distinct']
GENERIC_INJECTED = {'center', 'centre', 'services', 'service', 'partners'}


def split_dba(name):
    """'Ectoarcsol DBA: Patriot Eco, LLC' -> ('Patriot Eco, LLC', 'Ectoarcsol'); no DBA -> (name, None).
    Measured on true links: the S1 name is the part AFTER the DBA marker."""
    if not name:
        return '', None
    parts = _DBA_RE.split(name, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[1].strip(), parts[0].strip()
    return name, None


def name_flags(name):
    """Noise-type indicators usable as model features (computed on the RAW string)."""
    s = name or ''
    return {
        'native_script': bool(_HAS_INDIC.search(s)),
        'domain': bool(_URL_TOKEN_RE.search(s)),
        'handle': bool(re.match(r"^\W*[@#]\s*\w", s)) or bool(re.search(r"\s[@#]\w", s)),
        'dba': split_dba(s)[1] is not None,
        'phone': bool(re.search(r"\d{8,}", re.sub(r"[\s\-]", '', s))),
        'bracket': bool(re.search(r"[\[\](){}]", s)),
        'junk_prefix': bool(re.match(r"^\s*[^\w\sऀ-ൿ(\[]", s)),
        'allcaps': s.isupper(),
        'accented_latin': any(unicodedata.combining(c) for c in unicodedata.normalize('NFKD', s) if not _HAS_INDIC.match(c))
                          and not _HAS_INDIC.search(s),
        'leet': bool(re.search(r"[A-Za-z][0134578][A-Za-z]", s)),
    }


_RE_JUNK_MARKER = re.compile(r"\[\[[^\]]*\]\]")
_RE_DOTS_SPACES = re.compile(r"[\s.]")
_RE_MS = re.compile(r"(?i)\bm\s*/\s*s\b\.?")
_RE_POSSESSIVE = re.compile(r"'s\b")
_RE_BRACKETED = re.compile(r"[\[(]([^\])]*)[\])]")
_RE_TOKENS = re.compile(r"[0-9a-z]+")


def _tokens(s):
    return _RE_TOKENS.findall(s)


def _clean_name_string(name, _raw=None):
    """Shared string-level cleanup (before tokenization). Returns (clean_str, glued_alt_or_None)."""
    s = _prep(name) if _raw is None else _raw
    s, _alias = split_dba(s)
    if '(' in s:
        s = _ID_RE.sub(' ', s)
    if '[[' in s:
        s = _RE_JUNK_MARKER.sub(' ', s)                        # [[FFSAHIEGX]] junk markers
    if ' | ' in s:                                              # 'ANIMAL WELFARE LEAGUE LTD | www.animalwel.com'
        s = s.split(' | ')[0]
    s = _PHONE_RE.sub(' ', s)                                   # '... - 9895970478', '- [1226647866]'
    s = fold(s)                  # accents off BEFORE url/handle regexes ('shrámikindiachit.com'); lowercases
    glued = None
    if '.' in s:
        m = _URL_TOKEN_RE.search(s)
        if m:
            glued = m.group(1).replace('.', '').replace('-', '')
            s = s[:m.start()] + ' ' + glued + ' ' + s[m.end():]
    if '@' in s or '#' in s:
        s = _HANDLE_RE.sub(lambda mm: ' ' + mm.group(1) + ' ', s)  # '@sioreal', '#omindustries', 'Mr @gajpati'
    if '.' in s:
        s = _DOTTED_ABBR_RE.sub(lambda mm: _RE_DOTS_SPACES.sub('', mm.group(0)) + ' ', s)   # L.L.C. -> LLC
    if '/' in s:
        s = _RE_MS.sub(' m/s ', s)                              # M/s, M/S.
    s = s.replace('&', ' and ').replace('+', ' and ')
    s = fold(s)
    if "'" in s:
        s = _RE_POSSESSIVE.sub('s', s)                          # Cesya's -> cesyas (matches domain forms)
        s = s.replace("'", '')
    s = _LEET_RE.sub(lambda mm: _LEET[mm.group(0)], s)         # hospita1ity -> hospitality, 5ervices -> services
    if 'ln' in s:
        s = _L_FOR_I_RE.sub('in', s)                           # lnc -> inc, lndia -> india
    return s, glued


def normalize_name(name, country=None, _raw=None):
    """Lowercase, transliterated, accent-folded; junk/phone/brackets/handles/domains/DBA-alias removed;
    legal forms canonicalised but KEPT; sri/shri/smt dropped; consecutive duplicates collapsed."""
    ck = country_key(country)
    s, _ = _clean_name_string(name, _raw)
    s = s.replace('m/s', ' ')
    toks = [t for t in _tokens(s) if t not in ALWAYS_DROP_NAME]
    toks = _phrase_replace(toks, _LEGAL_PM[ck])
    if ck != 'generic':
        toks = _phrase_replace(toks, _LEGAL_PM['generic'])
    return ' '.join(_dedup_consecutive(toks))


_CORE_DROP = None


def name_core(name, country=None, _full=None, _raw=None):
    """normalize_name minus legal forms (all countries), honorifics, fillers (and/of/the/et), bracketed
    country words, leading French articles. Falls back to normalize_name if everything would be removed."""
    global _CORE_DROP
    if _CORE_DROP is None:
        _CORE_DROP = LEGAL_TOKENS_ALL | NAME_FILLERS | HONORIFICS | {'ms', 'pvt', 'ltd'}
    raw = _prep(name) if _raw is None else _raw
    full = normalize_name(name, country, raw) if _full is None else _full
    bracketed = set()
    if '(' in raw or '[' in raw:
        for b in _RE_BRACKETED.findall(raw):
            bracketed.update(_tokens(fold(b)))
    out = [t for t in full.split() if t not in _CORE_DROP and not (t in COUNTRY_WORDS and t in bracketed)]
    while out and out[0] in FR_ARTICLES and len(out) > 1:
        out = out[1:]
    return ' '.join(out) if out else full


def name_key(name, country=None):
    """Sorted unique core tokens -> invariant to token shuffles ('Physicians Elite Urology')."""
    return ' '.join(sorted(set(name_core(name, country).split())))


def name_glued(name, country=None):
    """Core without spaces: compare with domain/handle forms ('allenapexenergetics')."""
    return name_core(name, country).replace(' ', '')


def glued_similarity(name_a, name_b, country=None):
    """Pairwise score in [0,1] for domain/handle/concatenated forms ('tattoocorner.com' vs 'Corner Tattoo LLC').
    1.0 = glued forms equal (core or full); 0.9 = one glued form contains the other (len>=6: truncations,
    '(India)' kept in the domain); else the glued side is segmented with the OTHER record's tokens
    (split_concat) and scored by token-set overlap of cores (handles reordered concatenations)."""
    va, vb = name_variants(name_a, country), name_variants(name_b, country)
    fa = {va['glued'], va['norm'].replace(' ', '')} - {''}
    fb = {vb['glued'], vb['norm'].replace(' ', '')} - {''}
    if fa & fb:
        return 1.0
    for x in fa:
        for y in fb:
            s, l = (x, y) if len(x) <= len(y) else (y, x)
            if len(s) >= 6 and s in l:
                return 0.9
    best = 0.0
    for g, other in ((va['glued'], vb), (vb['glued'], va)):
        if ' ' in g or len(g) < 6:
            continue
        seg = split_concat(g, other['norm'].split())
        a = set(seg.split()) - _CORE_DROP
        b = set(other['core'].split())
        if a and b:
            best = max(best, 0.8 * len(a & b) / len(a | b))
    return best


def name_variants(name, country=None):
    """One pass: {'norm','core','key','glued','distinct'} (~4x cheaper than calling the functions separately).
    'distinct' = core minus GENERIC_INJECTED words (center/services/partners...), falls back to core."""
    raw = _prep(name)
    full = normalize_name(name, country, raw)
    core = name_core(name, country, full, raw)
    ctoks = core.split()
    distinct = [t for t in ctoks if t not in GENERIC_INJECTED]
    return {'norm': full, 'core': core, 'key': ' '.join(sorted(set(ctoks))), 'glued': core.replace(' ', ''),
            'distinct': ' '.join(distinct) if distinct else core}


def split_concat(glued, ref_tokens, min_len=2):
    """Segment a glued string using the OTHER record's tokens (DP maximizing covered chars, then fewest
    pieces). Uncovered spans are kept as separate tokens. split_concat('vijaysoftwareprivate',
    ['vijay','software','private','limited']) -> 'vijay software private'."""
    g = glued or ''
    vocab = sorted({t for t in ref_tokens if len(t) >= min_len}, key=len, reverse=True)
    n = len(g)
    best = [(0, 0, None)] * (n + 1)   # (covered, -pieces, backpointer (start, is_word))
    best = [None] * (n + 1)
    best[0] = (0, 0, None)
    for i in range(n):
        if best[i] is None:
            continue
        cov, pcs, _ = best[i]
        cand = (cov, pcs - 1, (i, False))
        if best[i + 1] is None or cand[:2] > best[i + 1][:2]:
            best[i + 1] = cand
        for w in vocab:
            if g.startswith(w, i):
                j = i + len(w)
                cand = (cov + len(w), pcs - 1, (i, True))
                if best[j] is None or cand[:2] > best[j][:2]:
                    best[j] = cand
    pieces, i = [], n
    while i > 0:
        start, is_word = best[i][2]
        pieces.append((g[start:i], is_word)); i = start
    pieces.reverse()
    out, buf = [], ''
    for p, w in pieces:
        if w:
            if buf:
                out.append(buf); buf = ''
            out.append(p)
        else:
            buf += p
    if buf:
        out.append(buf)
    return ' '.join(out)


# =====================================================================================================
# 4. ADDRESS maps
# =====================================================================================================
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california', 'co': 'colorado',
    'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana',
    'me': 'maine', 'md': 'maryland', 'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota',
    'ms': 'mississippi', 'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas',
    'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia', 'pr': 'puerto rico', 'gu': 'guam',
    'vi': 'virgin islands', 'as': 'american samoa', 'mp': 'northern mariana islands',
}
_US_STATE_CANON = {}
for _code, _nm in US_STATES.items():
    _US_STATE_CANON[_code] = _code
    _US_STATE_CANON[_nm] = _code
_US_STATE_CANON.update({'washington dc': 'dc', 'washington d c': 'dc', 'd c': 'dc', 'penn': 'pa', 'penna': 'pa',
                        'calif': 'ca', 'mass': 'ma', 'tex': 'tx', 'fla': 'fl', 'ill': 'il', 'wash': 'wa'})

# India: canonical = full English name (as S1 writes it). Codes = those seen in S2/S3 (MH DL UP KA TN GJ WB TG
# HR RJ KL BR MP AP PB OD) + standard alternates. Native forms: the 16 seen in data + common Hindi forms.
INDIA_STATES = {
    'andhra pradesh': ['ap', 'andhra', 'ఆంధ్రప్రదేశ్', 'ఆంధ్ర ప్రదేశ్', 'आंध्र प्रदेश'],
    'arunachal pradesh': ['ar', 'arunachal', 'अरुणाचल प्रदेश'],
    'assam': ['as', 'অসম', 'असम'],
    'bihar': ['br', 'बिहार'],
    'chhattisgarh': ['cg', 'ct', 'chattisgarh', 'chhatisgarh', 'छत्तीसगढ़'],
    'goa': ['ga', 'गोवा'],
    'gujarat': ['gj', 'gujrat', 'ગુજરાત', 'गुजरात'],
    'haryana': ['hr', 'हरियाणा'],
    'himachal pradesh': ['hp', 'हिमाचल प्रदेश'],
    'jharkhand': ['jh', 'झारखंड', 'झारखण्ड'],
    'karnataka': ['ka', 'ಕರ್ನಾಟಕ', 'कर्नाटक'],
    'kerala': ['kl', 'keralam', 'കേരളം', 'കേരള', 'केरल'],
    'madhya pradesh': ['mp', 'मध्य प्रदेश', 'मध्यप्रदेश'],
    'maharashtra': ['mh', 'maharastra', 'महाराष्ट्र'],
    'manipur': ['mn', 'मणिपुर'], 'meghalaya': ['ml', 'मेघालय'], 'mizoram': ['mz', 'मिजोरम'],
    'nagaland': ['nl', 'नागालैंड'],
    'odisha': ['od', 'or', 'orissa', 'ଓଡ଼ିଶା', 'ଓଡିଶା', 'ओडिशा', 'उड़ीसा'],
    'punjab': ['pb', 'ਪੰਜਾਬ', 'पंजाब'],
    'rajasthan': ['rj', 'राजस्थान'],
    'sikkim': ['sk', 'सिक्किम'],
    'tamil nadu': ['tn', 'tamilnadu', 'தமிழ்நாடு', 'தமிழ் நாடு', 'तमिलनाडु', 'तमिल नाडु'],
    'telangana': ['tg', 'ts', 'telengana', 'తెలంగాణ', 'तेलंगाना'],
    'tripura': ['tr', 'त्रिपुरा'],
    'uttar pradesh': ['up', 'उत्तर प्रदेश', 'उत्तरप्रदेश'],
    'uttarakhand': ['uk', 'ut', 'uttaranchal', 'उत्तराखंड', 'उत्तराखण्ड'],
    'west bengal': ['wb', 'পশ্চিমবঙ্গ', 'পশ্চিম বঙ্গ', 'पश्चिम बंगाल'],
    'delhi': ['dl', 'nct of delhi', 'nct delhi', 'दिल्ली'],
    'jammu and kashmir': ['jk', 'j and k', 'jammu kashmir', 'जम्मू और कश्मीर'],
    'ladakh': ['la', 'लद्दाख'], 'chandigarh': ['ch', 'चंडीगढ़'], 'puducherry': ['py', 'pondicherry', 'புதுச்சேரி'],
    'andaman and nicobar islands': ['an', 'andaman and nicobar'],
    'dadra and nagar haveli and daman and diu': ['dn', 'dd', 'daman and diu', 'dadra and nagar haveli'],
    'lakshadweep': ['ld'],
}
INDIA_CITY_ALIASES = {  # historical / variant spellings (domain knowledge)
    'bombay': 'mumbai', 'calcutta': 'kolkata', 'madras': 'chennai', 'bengaluru': 'bangalore',
    'gurugram': 'gurgaon', 'baroda': 'vadodara', 'poona': 'pune', 'trivandrum': 'thiruvananthapuram',
    'cochin': 'kochi', 'mysuru': 'mysore', 'mangaluru': 'mangalore', 'prayagraj': 'allahabad',
    'ahmadabad': 'ahmedabad', 'belagavi': 'belgaum', 'kalaburagi': 'gulbarga', 'vizag': 'visakhapatnam',
    'benares': 'varanasi', 'banaras': 'varanasi', 'pondy': 'puducherry', 'cuddapah': 'kadapa',
    'secunderabad': 'secunderabad', 'new delhi': 'delhi',
}
FR_REGIONS = {
    'hauts de france': ['nord pas de calais', 'picardie', 'hdf'],
    'nouvelle aquitaine': ['aquitaine', 'nouvelle aquitaine'],
    'pays de la loire': ['pays loire', 'pdl'],
    'ile de france': ['idf', 'iledefrance'], 'auvergne rhone alpes': ['aura'],
    'bourgogne franche comte': [], 'bretagne': [], 'centre val de loire': ['centre'], 'corse': [],
    'grand est': ['alsace', 'lorraine'], 'normandie': [], 'occitanie': [],
    'provence alpes cote d azur': ['paca', 'provence alpes cote dazur'],
}
# departments -> region (the 4 visible in test S2/S3 + all departments of the 3 test regions + big ones)
FR_DEPARTMENTS = {
    'nord': ('59', 'hauts de france'), 'pas de calais': ('62', 'hauts de france'), 'aisne': ('02', 'hauts de france'),
    'oise': ('60', 'hauts de france'), 'somme': ('80', 'hauts de france'),
    'gironde': ('33', 'nouvelle aquitaine'), 'landes': ('40', 'nouvelle aquitaine'),
    'charente': ('16', 'nouvelle aquitaine'), 'charente maritime': ('17', 'nouvelle aquitaine'),
    'correze': ('19', 'nouvelle aquitaine'), 'creuse': ('23', 'nouvelle aquitaine'),
    'dordogne': ('24', 'nouvelle aquitaine'), 'lot et garonne': ('47', 'nouvelle aquitaine'),
    'pyrenees atlantiques': ('64', 'nouvelle aquitaine'), 'deux sevres': ('79', 'nouvelle aquitaine'),
    'vienne': ('86', 'nouvelle aquitaine'), 'haute vienne': ('87', 'nouvelle aquitaine'),
    'loire atlantique': ('44', 'pays de la loire'), 'maine et loire': ('49', 'pays de la loire'),
    'mayenne': ('53', 'pays de la loire'), 'sarthe': ('72', 'pays de la loire'), 'vendee': ('85', 'pays de la loire'),
    'paris': ('75', 'ile de france'), 'rhone': ('69', 'auvergne rhone alpes'),
    'bouches du rhone': ('13', 'provence alpes cote d azur'), 'haute garonne': ('31', 'occitanie'),
    'bas rhin': ('67', 'grand est'), 'ille et vilaine': ('35', 'bretagne'), 'seine maritime': ('76', 'normandie'),
    'herault': ('34', 'occitanie'), 'alpes maritimes': ('06', 'provence alpes cote d azur'),
}
# cities visible in the test data -> (department, region); lets us compare S2/S3 rows that give only a city
FR_CITIES = {
    'lille': 'nord', 'roubaix': 'nord', 'tourcoing': 'nord', 'dunkerque': 'nord', 'calais': 'pas de calais',
    'bordeaux': 'gironde', 'pessac': 'gironde', 'merignac': 'gironde', 'la teste de buch': 'gironde',
    'lege cap ferret': 'gironde', 'nantes': 'loire atlantique', 'saint nazaire': 'loire atlantique',
    'saint herblain': 'loire atlantique', 'pornic': 'loire atlantique', 'la baule escoublac': 'loire atlantique',
}

STREET_TYPES = {
    'us': {
        'st': ['street', 'str', 'strt', 'saint', 'st'],          # 'Saint' is a WRONG expansion of St in S2/S3
        'rd': ['road', 'rd'], 'dr': ['drive', 'drv', 'dr'], 'ave': ['avenue', 'av', 'aven', 'avn', 'ave'],
        'ln': ['lane', 'ln'], 'ct': ['court', 'crt', 'ct'], 'blvd': ['boulevard', 'boul', 'blv', 'blvd'],
        'hwy': ['highway', 'hiway', 'hwy'], 'pkwy': ['parkway', 'pky', 'pkway', 'pkwy'], 'pl': ['place', 'pl'],
        'ter': ['terrace', 'terr', 'ter'], 'cir': ['circle', 'circ', 'crcl', 'cir'], 'way': ['wy', 'way'],
        'trl': ['trail', 'trl'], 'sq': ['square', 'sqr', 'sq'], 'rte': ['route', 'rte'], 'pike': ['pk', 'pike'],
        'expy': ['expressway', 'expy'], 'fwy': ['freeway', 'fwy'], 'aly': ['alley', 'aly'], 'loop': ['lp', 'loop'],
        'xing': ['crossing', 'xing'], 'plz': ['plaza', 'plz'], 'cv': ['cove', 'cv'], 'pt': ['point', 'pt'],
        'hts': ['heights', 'hts'], 'mt': ['mount', 'mt', 'mtn', 'mountain'], 'ft': ['fort', 'ft'],
        'n': ['north', 'n'], 's': ['south', 's'], 'e': ['east', 'e'], 'w': ['west', 'w'],
        'ne': ['northeast', 'ne'], 'nw': ['northwest', 'nw'], 'se': ['southeast', 'se'], 'sw': ['southwest', 'sw'],
        'unit': ['unit', 'apt', 'apartment', 'suite', 'ste', 'rm', 'room', 'bldg', 'building', 'cond', 'spc',
                 'space', 'trlr', 'lot', 'dept', 'office', 'ofc'],
        'fl': ['floor', 'flr', 'fl'],
    },
    'india': {
        'rd': ['road', 'rd'], 'st': ['street', 'str', 'st'], 'near': ['nr', 'near', 'nea'], 'opp': ['opposite', 'opp', 'oppo'],
        'behind': ['behind', 'bh', 'bhd'], 'no': ['number', 'num', 'no', 'nos'], 'hno': ['h no', 'house no', 'hn', 'hno', 'h'],
        'dno': ['d no', 'door no', 'dr no', 'dno'], 'plot': ['plot', 'plot no', 'pl no', 'plt'], 'flat': ['flat', 'flat no'],
        'shop': ['shop', 'shop no'], 'fl': ['floor', 'flr', 'fl'], 'gf': ['ground floor', 'grd floor', 'gf', 'g f'],
        'ff': ['first floor', '1st floor', 'ist floor', 'ff'], 'sf': ['second floor', '2nd floor', 'iind floor', 'sf'],
        'tf': ['third floor', '3rd floor', 'iiird floor'],
        '1st': ['first', 'ist', '1st'], '2nd': ['second', 'iind', 'iird', '2nd'], '3rd': ['third', 'iiird', '3rd'],
        '4th': ['fourth', 'ivth', '4th'], '5th': ['fifth', 'vth', '5th'],
        'nagar': ['ngr', 'nagar'], 'colony': ['col', 'colony'], 'sector': ['sec', 'sect', 'sector'],
        'marg': ['marg'], 'cross': ['crs', 'cross'], 'main': ['main'], 'layout': ['lyt', 'layout'],
        'extn': ['extension', 'ext', 'extn'], 'apt': ['apartment', 'apartments', 'apts', 'appt', 'apt'],
        'bldg': ['building', 'bldg'], 'soc': ['society', 'soc'], 'estate': ['est', 'estate'],
        'indl': ['industrial', 'indl', 'ind'], 'po': ['post office', 'po', 'p o', 'post'], 'dist': ['district', 'dist', 'distt'],
        'tal': ['taluka', 'taluk', 'tal', 'tq'], 'vill': ['village', 'vill', 'vpo', 'vil'], 'ps': ['police station'],
        'co': ['c o', 'care of'], 'so': ['s o', 'son of'], 'wo': ['w o', 'wife of'],
        'n': ['north'], 's': ['south'], 'e': ['east'], 'w': ['west'],
    },
    'france': {
        'rue': ['r', 'rue', 'ru'], 'av': ['avenue', 'ave', 'av', 'aven'], 'bd': ['boulevard', 'blvd', 'boul', 'bld', 'bd'],
        'pl': ['place', 'pl', 'plc'], 'imp': ['impasse', 'imp'], 'ch': ['chemin', 'chem', 'che', 'ch', 'chmn'],
        'all': ['allee', 'allees', 'alle', 'all', 'al'], 'sq': ['square', 'sq'], 'rte': ['route', 'rte', 'rt'],
        'quai': ['quai', 'qu', 'q', 'qua'], 'crs': ['cours', 'crs'], 'fg': ['faubourg', 'fbg', 'fg'],
        'cour': ['cour'], 'cite': ['cite', 'cit'], 'res': ['residence', 'res', 'resid'], 'pass': ['passage', 'pass', 'psg'],
        # 'ch' is chemin only (it used to be listed under chaussee too: last writer won -> 'Ch.' became 'chau';
        # measured on 0.42% of confident France test pairs, work/reports/diag/norm_audit.md B2/B3)
        'lot': ['lotissement', 'lot', 'lotis'], 'chau': ['chaussee', 'chau'], 'sen': ['sentier', 'sen'],
        # apartment = unit word: 'Appt 8' is a unit number, not a house number / city candidate (norm_audit B3)
        'apt': ['appartement', 'appart', 'appt', 'apt'],
        'prom': ['promenade', 'prom'], 'rpt': ['rond point', 'rpt'], 'za': ['zone artisanale'], 'zi': ['zone industrielle'],
        'zac': ['zone d amenagement concerte'], 'hameau': ['ham', 'hameau'], 'lieu dit': ['lieudit', 'ld'],
        'saint': ['st', 'saint'], 'sainte': ['ste', 'sainte'], 'grande': ['gde', 'grande'], 'general': ['gal', 'gen', 'general'],
        'docteur': ['dr', 'docteur'], 'marechal': ['mal', 'marechal'], 'president': ['pdt', 'pres', 'president'],
        'bat': ['batiment', 'bat', 'bt'], 'etage': ['etage', 'et', 'etg'], 'bp': ['boite postale', 'b p'], 'cs': ['c s'],
    },
}
_gen = {}
for _c in ('us', 'india', 'france'):
    for _k, _v in STREET_TYPES[_c].items():
        for _x in _v + [_k]:
            _gen.setdefault(_x, _k)          # first writer wins (US English first)
for _amb in ('dr', 'ste', 'ch', 'r', 'q', 'h', 'e', 'w', 'n', 's', 'et', 'al', 'ld', 'rt', 'ind', 'bh', 'bt', 'mg', 'post', 'lot'):
    _gen.pop(_amb, None)                     # ambiguous without country context -> leave as is
STREET_TYPES['generic'] = {}
for _x, _k in _gen.items():
    STREET_TYPES['generic'].setdefault(_k, []).append(_x)
_ADDR_PM = {c: _pm(d) for c, d in STREET_TYPES.items()}
STREET_TYPE_CANON = {
    'us': {'st', 'rd', 'dr', 'ave', 'ln', 'ct', 'blvd', 'hwy', 'pkwy', 'pl', 'ter', 'cir', 'way', 'trl', 'sq', 'rte',
           'pike', 'expy', 'fwy', 'aly', 'loop', 'xing', 'plz', 'cv'},
    'india': {'rd', 'st', 'marg', 'cross', 'main', 'nagar', 'colony', 'sector', 'layout'},
    'france': {'rue', 'av', 'bd', 'pl', 'imp', 'ch', 'all', 'sq', 'rte', 'quai', 'crs', 'fg', 'cour', 'cite', 'res',
               'pass', 'lot', 'chau', 'sen', 'prom', 'rpt', 'hameau'},
}
STREET_TYPE_CANON['generic'] = STREET_TYPE_CANON['us'] | STREET_TYPE_CANON['france'] - {'ch'}
UNIT_WORDS = {'unit', 'fl', 'apt', 'bldg', 'flat', 'shop', 'gf', 'ff', 'sf', 'tf', 'bat', 'etage', 'office'}
DESIGNATORS = {'hno', 'plot', 'dno', 'pmb', 'flat', 'shop', 'no', 'unit', 'fl', 'apt', 'office', 'kh', 'khasra',
               'sno', 'survey', 'gali', 'ward', 'block', 'bp', 'cs', 'sector', 'phase', 'khno', 'ps'}
NULL_TOKENS = {'null', 'none', 'nil', 'nan', 'n/a', 'n a', '<null>', 'na', 'n.a.', 'unknown', 'not available', '-', '--', 'null null'}
CITY_AFFIXES = re.compile(r"^(?:city|town|village|township|borough|municipality) of\s+|\s+(?:city|cdp|township|twp|town|village|ownship|ciyt)$")
FR_ADDR_STOP = {'de', 'du', 'des', 'la', 'le', 'les', 'd', 'l', 'au', 'aux', 'et', 'en', 'sur', 'sous'}
_FLOOR_WORDS = {'gf', 'ff', 'sf', 'tf', 'fl', 'etage', 'rdc', 'lgf', 'ugf'}


def _state_lookup(ck):
    if ck == 'us':
        return _US_STATE_CANON
    if ck == 'india':
        return _INDIA_STATE_CANON
    if ck == 'france':
        return _FR_STATE_CANON
    return _GENERIC_STATE_CANON


# '-' or '/' glued (no spaces) between a digit and a digit / a 1-2 letter prefix ('b-1201') / a 1-letter suffix
# ('15-a', '2/b'). 'Pune - 411001', 'Sector-21' (prefix longer than 2 letters) are split.
_NUM_HYPHEN_RE = re.compile(r"(?:(?<=\d)|(?<=\b[a-z])|(?<=\b[a-z]{2}))[-/](?=\d)|(?<=\d)[-/](?=[a-z]\b)")
_RE_CARE_OF = re.compile(r"\bc\s*/\s*o\b")
_RE_COMP_TOKENS = re.compile(r"[0-9a-z/#°&\-]+")


def _norm_component_text(c):
    c = fold(c)
    c = c.replace("'", ' ').replace('.', ' ').replace('_', ' ')
    # keep '-' and '/' inside number groups: India door numbers '4-5-6', '31-7-15/a', 'B-1201', '12/3'
    if '-' in c or '/' in c:
        c = _NUM_HYPHEN_RE.sub(lambda m: '\x01' if m.group(0) == '-' else '\x02', c)
        if '/' in c:
            c = _RE_CARE_OF.sub(' co ', c)                     # c/o (care of)
        c = c.replace('-', ' ').replace('/', ' ').replace('\x01', '-').replace('\x02', '/')
    return ' '.join(_RE_COMP_TOKENS.findall(c.replace('&', ' and ') if '&' in c else c))


_INDIA_STATE_CANON = {}


def _register_native_states():
    """(Re)build India state keys. Native aliases are keyed by their CURRENT transliteration, so this is re-run
    after load_native_dict / enable_snapping (addresses are transliterated before the state lookup)."""
    for canon, al in INDIA_STATES.items():
        for x in [canon] + al:
            if _HAS_INDIC.search(x):
                _INDIA_STATE_CANON[x] = canon
                for use_dict in (False, True):
                    _INDIA_STATE_CANON.setdefault(_norm_component_text(transliterate(x, use_dict=use_dict)), canon)
            else:
                _INDIA_STATE_CANON[_norm_component_text(x)] = canon
    for k, v in _INDIA_STATE_CANON.items():
        if len(k) > 2:
            _GENERIC_STATE_CANON[k] = _GENERIC_STATE_CANON.get(k, v)


_GENERIC_STATE_CANON = {}
_register_native_states()
_FR_STATE_CANON = {}
for _canon, _al in FR_REGIONS.items():
    for _x in [_canon] + _al:
        _FR_STATE_CANON[_norm_component_text(_x)] = _canon
for _d, (_num, _reg) in FR_DEPARTMENTS.items():
    _FR_STATE_CANON.setdefault(_d, _reg)
for _m in (_FR_STATE_CANON, _INDIA_STATE_CANON):
    for _k2, _v2 in _m.items():
        if len(_k2) > 2:                      # never map bare 2-letter codes without country context
            _GENERIC_STATE_CANON.setdefault(_k2, _v2)
for _k2, _v2 in _US_STATE_CANON.items():
    if len(_k2) > 2:
        _GENERIC_STATE_CANON.setdefault(_k2, _v2)


_RE_NUM_PREFIX = re.compile(r'^(?:no|n°|n)(?=\d)')
_RE_BIS = re.compile(r'(\d)\s*(?:bis|b)$')
_RE_TER = re.compile(r'(\d)\s*ter$')
_RE_QUATER = re.compile(r'(\d)\s*quater$')
_RE_LETTER_NUM = re.compile(r'[a-z]{1,2}-\d+[a-z]?')
_RE_LEAD_ZERO = re.compile(r'(?<![\d])0+(?=\d)')
_RE_DIGITS = re.compile(r'\d+')
_RE_HAS_DIGIT = re.compile(r'\d')


def _canon_number(tok):
    """'004007'->'4007', '1804A'->'1804a', '8bis'->'8b', '33ter'->'33t', 'B-1201'->'b1201', '31-7-15/A'->'31-7-15/a'."""
    if tok.isdigit():
        return tok.lstrip('0') or '0'
    t = tok.lower().strip('#°.,')
    t = _RE_NUM_PREFIX.sub('', t)
    t = _RE_BIS.sub(r'\1b', t)
    t = _RE_TER.sub(r'\1t', t)
    t = _RE_QUATER.sub(r'\1q', t)
    if _RE_LETTER_NUM.fullmatch(t):                        # B-1201, GF-24
        t = t.replace('-', '')
    return _RE_LEAD_ZERO.sub('', t)                        # strip leading zeros in every digit group


def _base_number(tok):
    m = _RE_DIGITS.search(tok)
    return str(int(m.group(0))) if m else None


def normalize_address(addr, country=None, _comps=None):
    """Transliterated, folded, lowercase; NULL/None/N/A tokens and injected 'PMB nnnn' removed; leading zeros
    stripped; street types / directionals / unit words / state names canonicalised (country-keyed with a
    generic fallback); '#', 'N°' removed; bis/ter -> b/t glued to the number. Commas are dropped."""
    parts = _address_components(addr, country) if _comps is None else _comps
    toks = []
    for comp in parts:
        toks.extend(comp['tokens'])
    return ' '.join(toks)


def address_key(addr, country=None):
    return ' '.join(sorted(set(normalize_address(addr, country).split())))


def address_all(addr, country=None):
    """One pass: returns (normalize_address, address_parts) -- ~2x cheaper than calling both."""
    comps = _address_components(addr, country)
    return normalize_address(addr, country, comps), address_parts(addr, country, comps)


_RE_INLINE_NULL = re.compile(r'(?:^|\s)(?:null|none|nan)(?=\s|$)')
_RE_ANGLE_NULL = re.compile(r'<\s*null\s*>')
_RE_FR_NUMPFX = re.compile(r'(?:^|\s)(?:n°|n °|no|#+)\s*(?=\d)')
_RE_HASH_NUM = re.compile(r'#+\s*(?=\d)')
_RE_BIS_TER_SPLIT = re.compile(r'\b(\d+)\s+(bis|ter|quater)\b')
# France: a single suffix letter written apart from the house number ('12 B', '12 T') is the same number as the glued
# form ('12B', '12bis' -> '12b'). Applied to the RAW component (before quotes become spaces) and only when the letter
# is followed by whitespace / the end, so elisions ("12 D'Arc") and words are left alone.
_RE_FR_LETTER_SUFFIX = re.compile(r"(?i)(?<!\w)(\d+)\s+([abcdt])(?=\s|$)")


def _address_components(addr, country):
    ck = country_key(country)
    s = _prep(addr)
    if not s.strip():
        return []
    states = _state_lookup(ck)
    comps = []
    for raw in s.split(','):
        raw_st = raw.strip()
        if not raw_st:
            continue
        # native state names are matched before transliteration side effects
        if _HAS_INDIC.search(raw_st) and raw_st in _INDIA_STATE_CANON:
            comps.append({'raw': raw_st, 'tokens': [_INDIA_STATE_CANON[raw_st]], 'state': _INDIA_STATE_CANON[raw_st]}); continue
        if ck == 'france' and ' ' in raw_st:
            raw_c = _RE_FR_LETTER_SUFFIX.sub(r'\1\2', raw_st)
        else:
            raw_c = raw_st
        c = _norm_component_text(raw_c)
        if not c or c in NULL_TOKENS:
            continue
        if 'n' in c:
            c = _RE_INLINE_NULL.sub(' ', c).strip()
            if '<' in c:
                c = _RE_ANGLE_NULL.sub(' ', c).strip()
        if not c:
            continue
        st = states.get(c)
        if st is not None:
            comps.append({'raw': raw_st, 'tokens': st.split(), 'state': st}); continue
        # numbers: N°34 / No.34 / #401 / ##39 -> 34 / 401 / 39 ; bis/ter glue
        c = _RE_FR_NUMPFX.sub(' ', c) if ck == 'france' else (_RE_HASH_NUM.sub(' ', c) if '#' in c else c)
        c = c.replace('°', ' ').replace('#', ' # ')
        if 'bis' in c or 'ter' in c or 'quater' in c:
            c = _RE_BIS_TER_SPLIT.sub(r'\1\2', c)
        toks = c.split()
        if ck != 'us':
            toks = [t for t in toks if t != '#']
        if ck in ('us', 'generic') and 'pmb' in toks:            # injected private-mailbox numbers
            toks = _drop_after(toks, {'pmb'})
        toks = _phrase_replace(toks, _ADDR_PM[ck])
        toks = [_canon_number(t) if _RE_HAS_DIGIT.search(t) else t for t in toks]
        toks = ['unit' if t == '#' else t for t in toks]
        toks = _dedup_consecutive(toks)                          # 'unit unit 100' ('Unit SUITE 100'), 'fl fl 1'
        if ck == 'france':
            fr_c = ' '.join(toks)
            if fr_c in FR_DEPARTMENTS:
                comps.append({'raw': raw_st, 'tokens': FR_DEPARTMENTS[fr_c][1].split(), 'state': FR_DEPARTMENTS[fr_c][1],
                              'dept': fr_c}); continue
        comps.append({'raw': raw_st, 'tokens': toks, 'state': None})
    return comps


def _drop_after(toks, keys):
    out, skip = [], False
    for t in toks:
        if skip and _RE_HAS_DIGIT.search(t):
            skip = False; continue
        skip = False
        if t in keys:
            skip = True; continue
        out.append(t)
    return out


_RE_SPLIT_PIN = re.compile(r'\b(\d{3})\s(\d{3})\b')
_RE_ST_SAINT = re.compile(r'^st(e?)\b')


def address_parts(addr, country=None, _comps=None):
    """Structured parse. Returns dict:
      house_numbers      canonical bare street numbers ('221', '1804a', '8b', 'b1201', '31-7-15/a')
      house_base         integer parts of house_numbers ('1804a'->'1804')  -> tolerant comparisons
      unit_numbers       numbers following unit words (unit/apt/suite/fl/flat/shop/#)
      designated_numbers {designator: [numbers]} e.g. {'hno': ['352'], 'plot': ['66']} (India 'Hn 352' is
                         often INJECTED noise in S2/S3: 2.2% of S1 vs 6% of S2/S3)
      postcodes          6-digit (India) / 5-digit trailing (US, France) codes
      all_numbers        every canonical number token
      street_tokens      street-name tokens without numbers / street type / stopwords
      street_type        canonical street type or None
      city_candidates    cleaned non-street, non-state, non-landmark components (+ alias-resolved variants)
      state_canonical    canonical state/region (US 2-letter code, India full name, France region) or None
      department         France department if given
      landmarks          India 'near/opp/behind ...' phrases
      other_tokens       every non-number word token outside the state component, minus designators / unit /
                         floor words / street types / French articles (bag-of-words for Jaccard)
    """
    ck = country_key(country)
    comps = _address_components(addr, country) if _comps is None else _comps
    res = {'house_numbers': [], 'house_base': [], 'unit_numbers': [], 'designated_numbers': {}, 'postcodes': [],
           'all_numbers': [], 'street_tokens': [], 'street_type': None, 'city_candidates': [], 'state_canonical': None,
           'department': None, 'landmarks': [], 'other_tokens': []}
    types = STREET_TYPE_CANON.get(ck, STREET_TYPE_CANON['generic'])
    street_idx = None
    for i, comp in enumerate(comps):
        if any(t in types for t in comp['tokens']) and street_idx is None and not comp['state']:
            street_idx = i
    if street_idx is None:
        for i, comp in enumerate(comps):
            if comp['tokens'] and comp['tokens'][0][:1].isdigit() and not comp['state']:
                street_idx = i; break
    state_idx = [i for i, comp in enumerate(comps) if comp['state']]
    last_state = state_idx[-1] if state_idx else None
    for i, comp in enumerate(comps):
        toks = comp['tokens']
        if comp['state']:
            if i == last_state:          # S1 puts the state LAST: 'Washington, DC' -> dc, 'Washington, PA' -> pa
                res['state_canonical'] = comp['state']
            elif comp.get('dept') is None:
                # an earlier state-like component is most likely a CITY ('Washington', 'Delhi', 'New York');
                # a bare code ('DL, Delhi') is not
                city = _norm_component_text(comp['raw']) if not _HAS_INDIC.search(comp['raw']) else comp['state']
                if city and len(city) > 2:
                    res['city_candidates'].append(city)
            if comp.get('dept'):
                res['department'] = comp['dept']
            continue
        if toks and toks[0] in ('near', 'opp', 'behind') or (len(toks) > 1 and toks[0] == 'next' and toks[1] == 'to'):
            res['landmarks'].append(' '.join(toks)); continue
        prev = None
        is_street = (i == street_idx)
        other_words = []
        for j, t in enumerate(toks):
            if _RE_HAS_DIGIT.search(t):
                res['all_numbers'].append(t)
                if (ck == 'india' and len(t) == 6 and t.isdigit()) or \
                        (ck in ('us', 'france') and len(t) == 5 and t.isdigit() and j == len(toks) - 1 and not is_street):
                    res['postcodes'].append(t)
                elif prev in UNIT_WORDS:
                    res['unit_numbers'].append(t)
                elif prev in DESIGNATORS:
                    res['designated_numbers'].setdefault(prev, []).append(t)
                else:
                    res['house_numbers'].append(t)
                    b = _base_number(t)
                    if b is not None:
                        res['house_base'].append(b)
            else:
                other_words.append(t)
            prev = t
        # India: 'Chennai 602 102' split PIN
        if ck == 'india' and len(toks) > 1:
            m = _RE_SPLIT_PIN.search(' '.join(toks))
            if m:
                res['postcodes'].append(m.group(1) + m.group(2))
        if is_street:
            words = [w for w in other_words if w not in DESIGNATORS and w not in UNIT_WORDS]
            st = [w for w in words if w in types]
            if st:
                res['street_type'] = st[0]
            res['street_tokens'].extend(w for w in words if w not in types and w not in FR_ADDR_STOP)
        else:
            # 'Hn 352 Niwasa' -> locality 'niwasa'; pure unit/floor components ('fl 1', 'sf') yield nothing
            words = [w for w in other_words if w not in DESIGNATORS and w not in UNIT_WORDS and w not in _FLOOR_WORDS]
            # measured junk "cities" (mini pool): 'po box' was the most frequent US candidate; 'a'/'c' come from
            # 'Suite C', 'n w' / 'ii' / 'dl' / 'and' from fragments -> a city never consists of <=2-char words only
            if words and words[:2] != ['po', 'box'] and words[0] != 'box' and any(len(w) > 2 and w != 'and' for w in words):
                res['city_candidates'].append(' '.join(words))
        res['other_tokens'].extend(w for w in other_words if w not in DESIGNATORS and w not in UNIT_WORDS
                                   and w not in _FLOOR_WORDS and w not in FR_ADDR_STOP and w not in types)
    # city clean-up + aliases
    cands = []
    for c in res['city_candidates']:
        c2 = CITY_AFFIXES.sub('', c).strip()
        if ck in ('france', 'generic'):
            if c2.startswith('st'):
                c2 = _RE_ST_SAINT.sub(lambda m: 'saint' + ('e' if m.group(1) else ''), c2)
        if ck in ('india', 'generic'):
            c2 = INDIA_CITY_ALIASES.get(c2, c2)
        for x in (c, c2):
            if x and x not in cands and any(len(w) > 2 for w in x.split()):
                cands.append(x)
    res['city_candidates'] = cands
    if res['state_canonical'] is None and ck == 'france':
        for c in cands:
            if c in FR_CITIES:
                res['department'] = FR_CITIES[c]
                res['state_canonical'] = FR_DEPARTMENTS[FR_CITIES[c]][1]
                break
    return res


# =====================================================================================================
# 5. contract entry point for frames (polars lives in normalize_frame.py; this module stays stdlib-only)
# =====================================================================================================
def add_normalized_columns(df, n_jobs=-1, **kw):
    """Contract name: normalize.add_normalized_columns(df). Delegates to normalize_frame.add_normalized_columns."""
    try:
        from . import normalize_frame as _nf
    except ImportError:
        import normalize_frame as _nf
    return _nf.add_normalized_columns(df, n_jobs=n_jobs, **kw)
