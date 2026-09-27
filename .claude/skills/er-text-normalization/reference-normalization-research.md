> Research notes captured 2026-09-25 while building these skills (measured on the real data on the team laptop). Paths such as `research/...`, `scratchpad/...` or `blocking_exp/...` point to a temporary scratch folder that is NOT part of the project — rely on the numbers and conclusions, and re-run measurements with the project code when needed. Sections marked PENDING/TODO were not finished.

# R2 — Text normalization for US / India / France (France test-only), with measurements

Team Master Bolt — Amazon ML Challenge 2026, Business Entity Resolution.
All numbers below were measured on this laptop (Windows 11, py3.10 venv) on the provided data only.
No dataset content was sent anywhere. Everything lives under `scratchpad/research/`.

> STATUS: report is written incrementally. Sections marked (pending) are filled later in the same run.

## 0. TL;DR (pending, filled at the end)

## 1. Indic script -> Latin transliteration: libraries and measured quality

### 1.1 Evaluation set
- Source: `research/data/pairs_mini.parquet` = all true links (S1 x S2/S3) of the 12k-S1 mini train set
  (42,094 links: India 21,073, US 21,021; 11,374 matched S1). Built by `research/build_pairs.py` (DuckDB, 2.5 GB cap).
- Subset for this section: India links whose S2/S3 **name** contains Indic script -> **3,852 pairs**
  (deva 2,235 | telu 321 | knda 293 | taml 261 | gujr 247 | beng 238 | mlym 141 | orya 69 | guru 47).
- Negatives: same S2/S3 name vs a random *other* India S1 name (seed 7) -> AUC = P(sim(true) > sim(random)).
- Both sides are then folded (NFKD, drop combining marks, lowercase, non-alnum -> space). IMPORTANT: folding is
  done **after** transliteration; doing NFD+strip first destroys matras ("लिमिटेड" -> "लमटड").
- Script: `research/eval_translit.py` -> `research/data/translit_eval.out` / `.csv`.

### 1.2 Candidate libraries (versions installed in the venv, licenses from PyPI metadata)
| library | version | license | pip on Win py3.10 | notes |
|---|---|---|---|---|
| `anyascii` | 0.3.3 | **ISC** (permissive) | yes, pure python | covers all scripts; drops schwa aggressively ("praivet"); fastest (≈32 µs/name) |
| `indic-transliteration` (sanscript) | 2.3.82 | **MIT** | yes, pure python | scholarly schemes (IAST/ITRANS/OPTITRANS/HK); keeps every inherent 'a' ("limiṭeḍa") -> worse for English loanwords; ≈0.29 ms/name; needs per-run script detection |
| `unidecode` | 1.4.0 | **GPL-2.0+ — DO NOT SHIP** | yes | good quality but GPL is incompatible with the MIT/Apache requirement for the submitted code; measured only as a reference |
| `aksharamukha` | 2.3 | **AGPL-3.0 — DO NOT SHIP** | yes (heavy) | slowest (≈40 ms/name, 1,000x anyascii) and worst for this task |
| `ai4bharat-transliteration` (IndicXlit) | 1.1.3 | MIT (code + models) | **not practical**: needs `fairseq` built from source (+ old torch), no Windows wheels | 11M-param transformer (6+6 layers, d=256), Indic->Roman model exists, beam 4. Not installed. Overkill: the data's native names come from a closed vocabulary (see §2), which a learned dictionary solves exactly. |
| custom rule transliterator (ours) | — | ours (MIT-compatible) | stdlib only | ISCII-parallel table over all 9 blocks + schwa deletion for north scripts; in `normalize_ref.py` (`transliterate`) and `translit_core.py` |

### 1.3 Measured similarity on true links (3,852 native-script pairs)
TSR = mean rapidfuzz `fuzz.token_set_ratio` (0-100); JW = mean rapidfuzz `JaroWinkler.normalized_similarity`.
Post-processing layers are cumulative per token: P0 = fold only; P1 = +collapse doubled letters;
P2 = +phonetic key (ph->f, w->v, z->j, c->k/s, aspirates kh/gh/th/dh/bh->k/g/t/d/b, y->i, doubles collapsed);
P3 = +skeleton (phonetic key minus non-initial vowels).

| variant | P0 TSR | P0 JW | P2 TSR | P2 JW | P3 TSR | P3 JW | AUC(JW) P2 | AUC(JW) P3 | time / name |
|---|---|---|---|---|---|---|---|---|---|
| crude (NFD-strip, no translit) | 14.8 | 0.425 | 14.9 | 0.425 | 16.7 | 0.442 | 0.635 | 0.645 | — |
| unidecode (GPL, reference) | 67.9 | 0.800 | 80.7 | 0.881 | 90.6 | 0.945 | 0.991 | 0.996 | 69 µs |
| **anyascii (ISC)** | 74.3 | 0.840 | 79.6 | 0.874 | 89.1 | 0.935 | 0.989 | 0.995 | 32 µs |
| sanscript IAST (MIT) | 68.5 | 0.806 | 73.6 | 0.843 | 86.8 | 0.926 | 0.974 | 0.989 | 289 µs |
| sanscript OPTITRANS (MIT) | 68.9 | 0.809 | 73.9 | 0.842 | 86.4 | 0.922 | 0.967 | 0.980 | 312 µs |
| custom, schwa=keep | 72.7 | 0.836 | 77.9 | 0.868 | 92.2 | 0.955 | 0.990 | 0.998 | 206 µs |
| custom, schwa=final | 76.0 | 0.856 | 81.1 | 0.889 | 92.2 | 0.955 | 0.991 | 0.998 | 180 µs |
| custom, schwa=drop | 76.4 | 0.859 | 81.7 | 0.891 | 92.2 | 0.955 | 0.993 | 0.998 | 271 µs |
| **custom, schwa=auto + ph->f** (default in normalize_ref) | **77.5** | **0.865** | 81.5 | 0.891 | 92.2 | 0.955 | 0.991 | 0.998 | 773 µs* |
| aksharamukha RomanReadable (AGPL) | 59.8 | 0.784 | 64.0 | 0.829 | 71.1 | 0.866 | 0.977 | 0.993 | 39.8 ms |

\* uncached; in `normalize_ref.py` results are cached per native token (only ~1.5k distinct native tokens exist
in the whole test set), so the amortised cost is negligible.

Per-script TSR at P2 (custom auto vs anyascii): deva 82.7/80.7, telu 82.1/81.6, knda 81.1/80.0, taml 72.8/69.6,
gujr 83.3/82.7, beng 79.2/78.3, mlym 81.6/73.9, orya 79.7/80.5, guru 76.4/75.1. Tamil is hardest (no voiced/unvoiced
distinction in the script: "டெக்னாலஜி" -> "teknalaji"), Malayalam gains most from the custom chillu/റ്റ handling.
Tamil intervocalic voicing ("tamil_voicing") was tested and HURT (72.8 -> 70.5) because loanwords keep voiceless stops.

**Take-aways**
1. Any transliteration is mandatory: crude folding gives JW 0.43 on these pairs; transliteration -> 0.84-0.87.
2. Best permissive off-the-shelf choice: `anyascii` (ISC, 32 µs, JW 0.840 raw). Our stdlib rule transliterator is
   slightly better (0.865) and has zero dependencies; either is fine **as a fallback** behind the learned dictionary.
3. The phonetic key (P2) adds +4-5 TSR points and the vowel-less skeleton (P3) another +10: use skeleton keys as an
   *extra* feature/blocking key for native-script names, not as the only representation (P3 also raises the
   random-pair TSR from 44 to 50).
4. Do NOT use unidecode (GPL) or aksharamukha (AGPL) in the shipped code.


## 2. Learned native-token dictionary (from TRAINING true links)

### 2.1 Key fact: the native-script names come from a CLOSED vocabulary
- All India train true links whose S2/S3 name contains Indic script, excluding the mini-train S1 ids (held out):
  **162,174 distinct (S1 name, native name) pairs** -> only **1,347 distinct native tokens** in total.
  The generator renders each English token with an (almost) fixed per-script spelling: "Private" -> "प्राइवेट"
  (Devanagari), "ಪ್ರೈವೇಟ್" (Kannada). Devanagari/Gujarati keep abbreviations ("Pvt Ltd" -> "प्रा. लि."), while the
  southern scripts usually EXPAND them ("Krishna Media Pvt Ltd" -> "కృష్ణ మీడియా ప్రైవేట్ లిమిటెడ్"); that is why
  the vote share of ప్రైవేట్->private is only 0.74 (vs 0.97 for Devanagari). Canonicalise pvt/private and
  ltd/limited AFTER the dictionary lookup (normalize_ref does).
- So a token dictionary learned from train pairs is (almost) an exact inverse of the generator.

### 2.2 Method (`research/learn_dict.py`, 53 s on the full India train links, <1 GB RAM)
1. Tokenize: native tokens = NFC-lowercased runs of `[0-9a-z\u0900-\u0D7F\u200c\u200d]` containing Indic chars
   (ZWJ/ZWNJ removed); Latin tokens = folded `[0-9a-z]+` of the S1 name.
2. Co-occurrence: Dice(n, l) = 2·c(n,l) / (c(n)+c(l)) over the deduplicated pairs.
3. Alignment vote per pair: each native token is aligned to the unused S1 token with the highest
   Jaro-Winkler(phonetic_key(rule_translit(n)), phonetic_key(l)) if >= 0.55, else to its Dice-best token if that
   token is in the pair (this catches abbreviations: प्रा.->pvt, लि.->ltd, एलएलपी->llp).
4. Keep n -> majority l if support >= 2 and share >= 0.4. Result: 1,347 entries (`research/data/native_token_dict.tsv`,
   columns `native, latin, support, share`). Pure co-occurrence (`native_token_dict_cooc.tsv`) is almost as good.
   (Cosmetic bug in the old script: `share` can exceed 1.0 when a token repeats inside one name; the copy in
   `normalize_ref.learn_native_dict` caps it at 1.0. It does not change the argmax.)

Top entries (support): लिमिटेड->limited 33,825 · प्राइवेट->private 24,410 · లిమిటెడ్->limited 10,021 ·
लि->ltd 9,599 · प्रा->pvt 9,217 · एलएलपी->llp 4,973 · इंफ्रास्ट्रक्चर->infrastructure 1,215
(rule translit: "infrastrakchar") · सर्विसेज->services (rule: "sarvisej") · एंटरप्राइजेज->enterprises (rule: "entarpraijej").

### 2.3 Measured uplift on the held-out mini pairs (3,852 native-name true links; `research/eval_dict.py`)
| representation | TSR | JW | % TSR=100 | % JW>=0.9 | AUC(TSR) | AUC(JW) |
|---|---|---|---|---|---|---|
| rule transliteration only (custom auto, ph->f) | 77.49 | 0.8653 | 1.5 | 37.3 | 0.9693 | 0.9830 |
| rule translit + phonetic key | 81.50 | 0.8912 | 2.0 | 52.1 | 0.9821 | 0.9919 |
| rule translit + skeleton key | 92.21 | 0.9555 | 31.5 | 88.7 | 0.9854 | 0.9981 |
| **dictionary (align) + rule fallback** | **99.18** | **0.9945** | **92.0** | **98.8** | 0.9993 | 0.9996 |
| dictionary (cooc) + rule fallback | 99.04 | 0.9907 | 91.8 | 99.2 | 0.9994 | 0.9998 |

End-to-end through `normalize_ref.name_core` (same 3,852 links, `research/eval_uplift.py`):
without dictionary core TSR 63.5 / JW 0.781 / exact-core 1.4% -> **with dictionary core TSR 99.7 / JW 0.996 /
exact-core 95.3%**. The dictionary is the single most valuable normalization component for India.

### 2.4 Coverage on TEST (unlabeled, all India S2/S3 rows with Indic script)
- Names: 867,245 rows, 3,348,321 native tokens, 1,518 types -> **96.39% of token occurrences covered**
  (88.7% of types). The uncovered 171 types (121k occurrences) are almost all NEW generic shop words that never
  occur in train: मोटर्स (motors), स्टोर्स (stores), एजेंसीज (agencies), ज्वेलर्स (jewellers), जनरल (general), बेकरी (bakery),
  प्रोविजन (provision), स्वीट्स (sweets), फार्मेसी (pharmacy), मेडिकल्स (medicals), ऑटोमोबाइल्स, ट्रेडर्स (traders),
  हार्डवेयर, गारमेंट्स, स्टील, इलेक्ट्रॉनिक्स, फर्नीचर, रेस्टोरेंट, टेक्सटाइल्स (+ the same words in the other scripts).
  => Test India has a new name category (shops) — expect a slightly different name distribution than train.
- Addresses: only **17 native token types** appear in test addresses, and all of them are STATE names
  (महाराष्ट्र 244k, दिल्ली 154k, प्रदेश, उत्तर, ಕರ್ನಾಟಕ, தமிழ்நாடு, পশ্চিমবঙ্গ, తెలంగాణ, हरियाणा, राजस्थान, കേരളം, बिहार, मध्य,
  ఆంధ్రప్రదేశ్, ਪੰਜਾਬ, ଓଡ଼ିଶା). The same 16 state strings appear in train S2/S3 (`data/geo2.out`). They are handled by
  the hand map `INDIA_STATES` in `normalize_ref.py` (exact native string -> canonical English state).

### 2.5 Fallback for tokens not in the dictionary (`research/eval_snap.py`, `eval_snap2.py`)
Simulation: hold out 25% of dictionary types with support >= 5 (324 types) and try to recover them.
| fallback | correct | wrong | unchanged |
|---|---|---|---|
| plain rule transliteration (exact) | 21.0% | — | — |
| snap to S1-vocab, JW(pkey) >= 0.88, all tokens with count >= 2 (50k types) | 38.6% | 58.0% | 3.4% |
| snap, Levenshtein(pkey) >= 0.7, vocab count >= 20 (7.6k types) | 57.4% | 25.3% | 17.3% |
| **snap, Levenshtein(pkey) >= 0.7, vocab count >= 500 (299 types)** | **62.7%** | **3.1%** | 34.3% |
| snap, Levenshtein(pkey) >= 0.8, vocab count >= 500 | 46.0% | 0.6% | 53.4% |

Snapping to a big vocabulary is harmful (58% wrong: "லைஃப்"->"latif", "குட்"->"cut"). Snapping only to the ~300 most
frequent Latin name tokens is safe and is implemented as `normalize_ref.enable_snapping(counts, min_count=500, thr=0.7)`
(off by default). For the real test-uncovered words it recovers e.g. स्टोर्स->stores, ट्रेडर्स->traders, मेडिकल्स->medicals,
गारमेंट्स->garments, हार्डवेयर->hardware, ఫార్మసీ->pharmacy, ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್->electronics, but also makes errors
(मोटर्स->"mota", स्टील->"still", स्वीट्स->"swiss") — so treat snapping as optional.
Better: at SCORING time the pairwise soft match already works for unseen tokens: best pkey-JW of an unseen native
token vs the tokens of the TRUE S1 name = 0.864 on average vs 0.616 vs a random S1 (>=0.8: 77.5% vs 17.1%).
Use token-level soft matching (e.g. rapidfuzz `token_set_ratio` on phonetic keys, or Monge-Elkan with JW) as a feature,
rather than hard snapping.

### 2.6 Working code (shipped inside `normalize_ref.py`)
```python
import normalize_ref as N
# training time (pairs = [(s1_latin_name, s2s3_native_name), ...] from TRAIN true links only)
d = N.learn_native_dict(pairs, min_support=2, min_share=0.4, sim_thr=0.55)   # -> {native: (latin, support, share)}
with open('native_token_dict.tsv', 'w', encoding='utf-8') as f:
    f.write('native\tlatin\tsupport\tshare\n')
    for n, (l, c, sh) in sorted(d.items(), key=lambda x: -x[1][1]):
        f.write(f'{n}\t{l}\t{c}\t{sh}\n')
# inference time
N.load_native_dict('native_token_dict.tsv')          # returns #entries (1,347)
N.normalize_name('ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', 'India')   # -> 'fortune finance pvt ltd'
N.normalize_name('अरिहंत इंफ्रास्ट्रक्चर प्रा. लि.', 'India')      # -> 'arihant infrastructure pvt ltd'
```
Legality: the dictionary is derived only from the provided training data (true links). It is a data artifact of
the pipeline and must be regenerated by the submitted code (`learn_native_dict`) — do not hand-edit it.

## 3. Hand-written normalization maps (in `normalize_ref.py`; domain knowledge only, no downloaded gazetteers)

All maps are **country-keyed** (`country_key(country)` -> `'us' | 'india' | 'france' | 'generic'`; any other label,
`None`, `''` or a non-string maps to `'generic'` and never raises). `'generic'` = union of the three country maps
with ambiguous short forms removed, so an unseen test country still gets sensible legal-form / street-type /
state handling. Aliases: `US/USA/United States/...`, `India/IN/IND/Bharat`, `France/FR/FRA`.

### 3.1 Legal forms (`LEGAL_FORMS`, canonical <- variants; matched on tokens AFTER dotted-abbreviation collapse)
| country | canonical tokens (variants) |
|---|---|
| US | llc (L.L.C., limited liability company), inc (incorporated), corp (corporation, corpn), co (company, comp), ltd (limited), lp, llp, lllp, pc (professional corporation), pllc, pa (professional association), plc, na (national association), dba |
| India | pvt (private, pte, prvt), ltd (limited, lmt, limted), llp, opc (one person company), co (company), corp, inc |
| France | sarl (S.A.R.L., societe a responsabilite limitee), sas (S.A.S.), sasu, eurl, sa (societe anonyme), sci, snc, scp, scm, scop, selarl, gie, ei (entreprise individuelle), eirl, cie (compagnie), ets (etablissement(s), etabl), fils (et fils), freres (et freres) |
- Dotted abbreviations are collapsed first by regex (`L.L.C.` -> `LLC`, `S.A.R.L` -> `SARL`, `S.A.S` -> `SAS`).
- `normalize_name` KEEPS the canonical legal token (useful as a weak feature: legal-form agreement);
  `name_core` DROPS every legal token of every country + honorifics + fillers (and/et/of/the/und/y), then leading French
  articles (le/la/les/l/des/du/de/d/au/aux) and bracketed country words ("(France)"). If everything would be dropped,
  `name_core` falls back to `normalize_name` (e.g. "Private Limited" -> "pvt ltd"), so it is never empty for a non-empty name.
- Honorifics (`HONORIFICS`): sri, shri, smt, dr, mr, mrs, ms, messrs, m/s, kumari, km, the, thiru, tmt, selvi.
  `sri/shri/smt/m/s` are dropped already in `normalize_name` (injected noise); the rest only in `name_core`.

### 3.2 Street types / address words (`STREET_TYPES`, canonical <- variants)
- US (USPS style): st (street, str, strt, **saint** = the generator's WRONG expansion "221 Seneca Saint"), rd, dr (drive),
  ave (avenue, av), ln, ct, blvd, hwy, pkwy, pl, ter, cir, way, trl, sq, rte, pike, expy, fwy, aly, loop, xing, plz, cv, pt,
  hts, mt, ft; directionals n/s/e/w/ne/nw/se/sw; unit words (apt, suite, ste, rm, bldg, spc, lot, dept, ofc -> `unit`); floor -> `fl`.
- France: rue (r, r., ru), av (avenue, ave, av.), bd (boulevard, blvd, bd.), pl (place), imp (impasse), ch (chemin, chem),
  all (allee, all.), sq (square), rte (route), quai (qu, q.), crs (cours), fg (faubourg, fbg), cour, cite, res (residence),
  pass (passage), lot (lotissement), chau (chaussee), sen, prom, rpt (rond point), za, zi, zac, hameau, lieu dit;
  saint/sainte (st/ste), grande (gde), general (gal), docteur (dr), marechal (mal), president; bat, etage, bp, cs.
  `N°34`, `No 34`, `#401` -> `34`, `401`; `8 BIS` / `8bis` / `8 B` -> `8b`, `4 ter` -> `4t`, `quater` -> `q`.
- India: rd, st, near (nr), opp (opposite, oppo), behind (bh, bhd), no (number, num), hno (h no, h.no, house no, hn),
  dno (door no, d no), plot (plot no), flat, shop, fl (floor), gf/ff/sf/tf (ground/first/second/third floor; `Iind Floor` -> `sf`),
  ordinals 1st..5th (ist, iind, iiird, ivth, vth), nagar (ngr), colony (col), sector (sec), marg, cross, main, layout,
  extn, apt, bldg, soc, estate, indl, po (post office), dist (distt), tal (taluka, tq), vill (vpo), co (c/o), so, wo.
  NOTE: `MG` is NOT mapped to marg (MG Road = Mahatma Gandhi Road) — this was a bug in the previous run's draft.
- generic: union with first-writer-wins, minus ambiguous short forms (dr, ste, ch, r, q, h, e, w, n, s, et, al, ld, rt, ind, bh, bt, mg, post, lot).
- NULL tokens removed as whole components or inline: null, none, nil, nan, n/a, `<NULL>`, unknown, not available, -, --.
- US `PMB nnnn` (private mailbox, injected noise, e.g. "002216 DUDLEY LANE, PMB 4593, BURLESON, TX") is dropped with its number.

### 3.3 States / regions
- US: all 50 states + DC (+ PR, GU, VI, AS, MP) code <-> name; canonical = 2-letter code (S1 style). Extra aliases:
  washington dc, d c, penn/penna, calif, mass, tex, fla, ill, wash. Measured (`data/geo2.out`): S1 always ends with the
  2-letter code; S3 writes full names (Texas 253k, New York 194k, ...), plus null/NULL/<NULL>/N/A as last component (~1.4k each).
- India: 36 states/UTs with the codes seen in S2/S3 (MH 317k, DL 201k, UP 122k, KA 116k, TN 104k, GJ 95k, WB 95k, TG 74k,
  HR 60k, RJ 53k, KL 47k, BR 38k, MP 37k, AP 29k, PB 22k, OD 18k) + standard alternates (TS, OR, CG/CT, UK/UT, ...) and the
  16 native-script state strings observed in the data (महाराष्ट्र 213k, दिल्ली 134k, उत्तर प्रदेश 82k, ಕರ್ನಾಟಕ 78k, தமிழ்நாடு 70k,
  পশ্চিমবঙ্গ 63k, ગુજરાત 63k, తెలంగాణ 50k, हरियाणा 39k, राजस्थान 36k, കേരളം 31k, बिहार 25k, मध्य प्रदेश 25k,
  ఆంధ్రప్రదేశ్ 19k, ਪੰਜਾਬ 14k, ଓଡ଼ିଶା 12k) + Hindi forms of the others. Canonical = full English name. S1 uses
  **"Orissa"** (11.7k) while S2/S3 use OD / ଓଡ଼ିଶା -> all map to `odisha`.
  City aliases (historic names): bombay->mumbai, calcutta->kolkata, madras->chennai, bengaluru->bangalore,
  gurugram->gurgaon, baroda->vadodara, poona->pune, trivandrum->thiruvananthapuram, cochin->kochi, mysuru->mysore,
  mangaluru->mangalore, prayagraj->allahabad, new delhi->delhi, ... (S1 itself uses both "Calcutta" (430x) and "Kolkata").
- France (test only; `data/geo_france.out`): the S1 last component is always one of 3 regions (Hauts-de-France 88k,
  Nouvelle-Aquitaine 74k, Pays de la Loire 63k). S2/S3 give the region (Hauts-de-France 162k, ...), the DEPARTMENT
  (Nord 130k, Gironde 130k, Loire-Atlantique 110k, Pas-de-Calais 24k), only a city, or nothing (43k NULL).
  Only 15 cities: Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac, La Teste-de-Buch,
  Merignac, Lege-Cap-Ferret, Pornic, Saint-Herblain, La Baule-Escoublac, with variants ST-NAZAIRE / St.-nazaire /
  LA TESTE DE BUCH / Merignac / LEGE CAP FERRET. No 5-digit postcode at all in France S2 (0 of 703k rows).
  Maps: `FR_REGIONS` (13 regions + old-region aliases), `FR_DEPARTMENTS` (the 4 visible + all departments of the 3 test
  regions + a few big ones; dept -> (number, region)), `FR_CITIES` (the 15 test cities -> department). A department is
  converted to its region (`state_canonical`), and when no region/department is given the region is inferred from a
  known city, so S1 (region) and S2/S3 (department / city only) become comparable.
- Rule for multiple state-like components: the LAST one is the state (S1 order), earlier ones become city
  candidates ("1600 K St, Washington, DC" -> state dc, city washington; "2685, Naya Bazar, Delhi, DL" -> state delhi,
  cities naya bazar + delhi). (The previous draft took the FIRST one, which turned "Washington, DC" into WA.)

## 4. Name / address cleanup operators (matching the observed noise)
| noise operator (from EDA) | operator in `normalize_ref` | example (verbatim input -> output) |
|---|---|---|
| native-script names | `transliterate` (dict first, rules fallback, cached per token) | "ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" -> "fortune finance pvt ltd" |
| injected accents | NFKC, then transliterate, then `fold` (NFKD strip; ASCII fast path) | "Unified Pinnacle Ésports L.L.C." -> "unified pinnacle esports llc"; "DÀNCE Çlub" -> "dance club" |
| junk leading symbols (>>, --, <<, @, #) | tokenization on `[0-9a-z]+` after fold | ">> Katz Seafood" -> "katz seafood" |
| bracket insertions | `[[...]]` removed; other bracket text kept as tokens, except bracketed country words in core | "[[FFSAHIEGX]] Katz Seafood" -> "katz seafood"; "Dance Club (France)" core -> "dance club" |
| DBA / aka / trading as / t/a / f/k/a / formerly | `split_dba` keeps the part AFTER the marker (that is the S1 name in the true links) | "Onyxcalovantage Sys D.B.A. Unified Pinnacle Esports L.L.C." -> "unified pinnacle esports llc" |
| domain form + phone | `_URL_TOKEN_RE` strips scheme/www/TLD (com, net, org, co.in, in, fr, asso.fr, ...); `_PHONE_RE` strips 8+ digit runs; `name_glued` = core without spaces; `split_concat(glued, other_record_tokens)` segments using the other record | "prantechcorporate.com - 9895970478" -> "prantechcorporate"; "unifiedpinnacleesports" + S1 tokens -> "unified pinnacle esports" |
| handles | `@x` / `#x` -> x | "@COMITCERCLE" -> "comitcercle" |
| leetspeak digits inside words | 0/1/3/4/5/7/8 between letters -> o/l/e/a/s/t/b | "Hospita1ity Group" -> "hospitality group" |
| duplicated / moved legal suffix | legal canonicalisation + consecutive dedup; `name_core` drops; `name_key` sorts | "LLC LLC Womens Health" -> "llc womens health"; core "womens health" |
| suffix abbreviation swaps | phrase map per country + generic | "Girwa Projects Pvt. Ltd." / "GIRWA  PROJECTS PRIVATE" core -> "girwa projects" |
| honorific / prefix insertion | sri/shri/smt/m/s dropped; dr/mr/the/... dropped in core; leading French articles dropped in core | "M/s Sri Balaji Traders" core -> "balaji traders"; "des Union Retro" core -> "union retro" |
| token shuffles | `name_key` = sorted unique core tokens | "Physicians Elite Urology" == "Urology Elite Physicians" |
| '&' / '+' / possessive | -> "and" (filler, dropped in core); "Cesya's" -> "cesyas" (matches domain forms) | "Guiclan & Frères EURL" core -> "guiclan" |
| address NULL/None tokens | component- and token-level removal | "PLEASANT RUN ROAD, NULL, IRVING, TX" -> "pleasant run rd irving tx" |
| component reordering | `address_key` = sorted unique tokens; `address_parts` is order-free | "221 Seneca Street, New York, Corning" key == "221 Seneca Street, Corning, NY" key |
| wrong expansion (Saint for St) | US map st <- saint | "221 Seneca Saint" -> "221 seneca st" |
| house numbers: suffix / bis / ter / leading zeros | `_canon_number` | "1804A" -> "1804a" (base 1804); "8 BIS" -> "8b"; "002216" -> "2216"; "B-1201" -> "b1201"; "4-5-6" and "31-7-15/A" kept whole |
| unit designators | unit/apt/suite/ste/#/fl -> `unit_numbers`; India hno/dno/plot/flat/shop/sector -> `designated_numbers` | "#800 Oak Ave, Fl 1" -> house 800, unit 1; "Hn 352 Niwasa" -> {hno: 352}, locality niwasa |
| landmarks (India, ~12% of S1) | components starting near/opp/behind/next to -> `landmarks` (not city candidates) | "Near Sbi Atm" |
| state code <-> name <-> native | `state_canonical` | "राजस्थान" / "RJ" / "Rajasthan" -> "rajasthan" |
| France dept vs region | dept -> region; city -> dept -> region | "..., DUNKERQUE, Nord" and "..., Calais, Hauts-de-France" -> "hauts de france" |

### 4.1 Public API of `research/normalize_ref.py` (stdlib only: `re`, `unicodedata`; rapidfuzz optional)
```python
import normalize_ref as N
N.load_native_dict('research/data/native_token_dict.tsv')   # optional but strongly recommended for India
N.transliterate(text) -> str                 # Indic (9 scripts) -> Latin; other chars unchanged; '' for None
N.normalize_name(name, country) -> str       # cleaned, folded, legal forms canonicalised (kept)
N.name_core(name, country) -> str            # legal forms / honorifics / fillers / leading FR articles removed
N.name_key(name, country) -> str             # sorted unique core tokens (shuffle-invariant)
N.name_glued(name, country) -> str           # core without spaces (for domain-form comparisons)
N.name_variants(name, country) -> dict       # {'norm','core','key','glued'} in ONE pass (use this in pipelines)
N.name_flags(name) -> dict                   # native_script, domain, handle, dba, phone, bracket, junk_prefix, allcaps, accented_latin, leet
N.split_dba(name) -> (primary, alias|None);  N.split_concat(glued, ref_tokens) -> str
N.normalize_address(addr, country) -> str    # cleaned, canonical street types/state/numbers, NULLs removed
N.address_key(addr, country) -> str          # sorted unique normalized address tokens
N.address_parts(addr, country) -> dict       # house_numbers, house_base, unit_numbers, designated_numbers, postcodes,
                                             # all_numbers, street_tokens, street_type, city_candidates,
                                             # state_canonical, department, landmarks, other_tokens
N.address_all(addr, country) -> (str, dict)  # normalize_address + address_parts in ONE pass
N.phonetic_key(tok), N.skeleton_key(tok)     # cheap phonetic keys for transliterated tokens
N.learn_native_dict(pairs) / N.enable_snapping(counts, min_count=500, thr=0.7)
```
Unit tests: `research/test_normalize_ref.py` (verbatim EDA examples + garbage inputs + unknown countries), run with
`.venv/Scripts/python -m pytest -q -p no:cacheprovider research/test_normalize_ref.py`.

Speed (this laptop, py3.10, 20k real S2/S3 strings; the machine was busy so timings vary ±50%): `name_variants`
≈ 50-80 µs, `normalize_name` ≈ 45-60 µs, `transliterate` ≈ 3-5 µs (cached per token), `address_all` ≈ 150-240 µs,
`normalize_address` ≈ 75-140 µs. The previous run's version needed 732 µs per name (4 separate calls) and
1,445 µs per address; the speed-ups come from an ASCII fast path in `fold`, precompiled regexes with cheap
substring guards, a first-token guard in the phrase replacer and the one-pass `name_variants` / `address_all`.
Full-scale estimate: 12.5M train + 11.7M test records ≈ 24M × ~0.25 ms ≈ 100 CPU-minutes ≈ 12-15 min on 8 processes;
less in practice because S2/S3 duplicates repeat identical strings -> normalise DISTINCT strings only (polars
`unique()`, then join back).

## 5. Measured uplift of `normalize_ref` vs crude normalisation (mini train true links)

Script `research/eval_uplift.py` (final module; output `data/eval_uplift.out`, table `data/eval_uplift.csv`, per-pair
scores `data/eval_uplift_scores.parquet`). Positives = all 42,094 true links of the mini train (US 21,021, India
21,073). Negatives per positive: **rand** = same S2/S3 record vs a random other S1 of the same country; **hard** = vs
another S1 sharing a rare crude name token (df <= 50; 28,198 cases), else the same crude state (5,814), else random
(8,082). AUC = P(score(true) > score(negative)), ties = 0.5. Crude = NFD strip + lowercase + non-alnum -> space.
The native-token dictionary is loaded (learned without the mini S1 ids -> held out).

### 5.1 Names
| metric | US pos | US AUC_hard | India pos | India AUC_hard | India native-name pos (n=3,852) |
|---|---|---|---|---|---|
| crude JW | 0.928 | 0.951 | 0.745 | 0.795 | 0.060 |
| crude token_set_ratio | 0.932 | 0.939 | 0.758 | 0.795 | 0.077 |
| crude exact equality | 30.5% | — | 18.5% | — | 0.0% |
| `normalize_name` JW | 0.942 | 0.962 | 0.940 | 0.954 | 0.997 |
| `normalize_name` exact | 38.0% | — | 47.9% | — | 95.3% |
| **`name_core` JW** | **0.959** | **0.964** | **0.962** | **0.966** | **0.997** |
| `name_core` token_set_ratio | 0.940 | 0.932 | 0.941 | 0.946 | 0.998 |
| **`name_core` exact** | **61.5%** | — | **71.2%** | — | **96.6%** |
| `name_key` (sorted) exact | 64.7% | — | 72.1% | — | 96.6% |
| `distinct` (core minus center/services/partners) TSR | 0.954 | 0.926 | 0.958 | 0.943 | 0.998 |
| glued equal/substring | 75.9% | 0.865 | 81.6% | 0.902 | 97.2% |
| `glued_similarity` | 0.842 | 0.948 | 0.865 | 0.954 | 0.981 |
| name_best = max(core TSR, glued_sim, norm TSR) | 0.960 | 0.953 | 0.960 | 0.955 | 0.998 |

- India: crude name JW 0.745 -> 0.962, exact 18.5% -> 71.2%, AUC_hard 0.795 -> 0.966 (native-script names go from
  unusable, AUC 0.50, to AUC_hard 0.997). Latin-only India names: crude JW 0.899 -> core 0.954.
- US: JW 0.928 -> 0.959 and exact equality doubles (30.5% -> 61.5%; sorted key 64.7%).
- Hard negatives show why exact equality alone is dangerous: token-set ratio is HIGHER on hard negatives after
  removing legal words (US TSR AUC_hard 0.939 crude -> 0.932 core) — generic cores ("primary care group") collide.
  JW on the core and exact-key features gain; keep both crude and normalized similarities as model features.
- Dictionary ablation on the 3,852 native-name links (`name_core`): without dictionary TSR 63.5 / JW 0.786 / exact 1.7%
  -> with dictionary TSR 99.8 / JW 0.997 / exact 96.6%.
- Domain-form names only (US 1,202 / India 941 links with .com/.in/...): glued equal-or-substring 86.0% / 84.2%;
  `glued_similarity` > 0 for 98.4% / 97.7% of them vs 1.2% / 1.7% of random S1 pairs.
- DBA / alias markers: in 643/643 DBA links and 364/364 other alias-marker links (formerly, trading as, aka, née, t/a,
  fka, formerly known as) the S1 name is the part AFTER the marker.
- Honorifics sri/shri/smt/m/s: 0.00% of India S1 names vs 5.22% of India linked S2/S3 names (0% in US).

### 5.2 Addresses
| metric | US pos | US rand | US AUC | India pos | India rand | India AUC |
|---|---|---|---|---|---|---|
| crude token Jaccard | 0.556 | 0.010 | 0.974 | 0.695 | 0.024 | 0.973 |
| normalized token Jaccard | **0.795** | 0.019 | 0.972 | **0.761** | 0.020 | 0.975 |
| crude token_set_ratio | 0.858 | 0.345 | 0.954 | 0.917 | 0.367 | 0.961 |
| normalized token_set_ratio | 0.922 | 0.342 | 0.954 | 0.939 | 0.368 | 0.962 |
| state: crude last component equal | 37.6% | 1.7% | 0.680 | 30.0% | 3.0% | 0.635 |
| **state: `state_canonical` equal** | **95.1%** | 4.4% | 0.954 | **95.1%** | 9.9% | 0.926 |
| house number: crude first number equal | 72.8% | 0.06% | 0.864 | 63.3% | 0.86% | 0.812 |
| house number: `house_base` overlap | **80.7%** | 0.12% | 0.903 | 54.0% | 1.4% | 0.763 |
| any number: `all_numbers` base overlap | 81.5% | 0.26% | 0.906 | **81.4%** | 3.4% | 0.890 |
| street token Jaccard | 0.814 | 0.001 | 0.910 | 0.394 | 0.0004 | 0.698 |
| city: exact candidate overlap (crude comps) | 76.3% | 0.1% | 0.881 | 92.1% | 2.0% | 0.950 |
| city: `city_candidates` overlap | 80.2% | 0.1% | 0.901 | 94.8% | 3.3% | 0.958 |
| city: best JW between candidates | 0.914 | 0.440 | 0.925 | 0.954 | 0.578 | 0.939 |
| `other_tokens` Jaccard | 0.813 | 0.002 | 0.968 | 0.767 | 0.009 | 0.976 |
- Addresses with native-script state (India, 4,757 links): state equality crude 7.8% -> canonical 100.0%.
- Both addresses non-empty in 95.3% (US) / 96.2% (India) of true links; `state_canonical` is found for every non-empty
  address in these links.
- For India use `all_numbers` and `other_tokens`; `house_base` and `street_tokens` are for US/France (see pitfalls).

### 5.3 Exact-key view (blocking) — `research/eval_keys.py`, output `data/eval_keys.out`
Every mini S2/S3 record (57k, incl. ~25% distractors) is looked up in an index of the 12k mini S1 by exact key within
its country. recall = linked records whose true S1 is among the hits; prec = hits containing the true S1 / records
with >= 1 hit; fp = distractor records with a hit (the index has only 12k S1 — at full scale collisions grow).
KEYS_TABLE_PLACEHOLDER

### 5.4 France (test-only, unlabeled) — same script, 259,452 France S1 vs a 300k reservoir sample of France S2+S3
FRANCE_TABLE_PLACEHOLDER

### 5.5 What normalisation cannot fix (residual true links with max(core TSR, glued_similarity) < 0.7)
`data/residuals.out`: US 4.0% and India 6.0% of true links. Categories (share of ALL links):
| category | US | India | address sim >= 0.8 in the category |
|---|---|---|---|
| alias name (no shared token, e.g. "Delta Rivernorth" vs "Solxyloyuma") | 1.76% | 2.35% | 97% / 99% |
| generic-word substitution / heavy typo ("Young Empire" vs "Young Services", "Om Ecom Corp" vs "OM CORP-SERVICE") | 1.81% | 3.40% | 94% / 93% |
| truncation / initials in domain ("fgsmart.com" = Fuller Great Smart, "tbellinger.com") | 0.42% | 0.22% | 94% / 100% |
=> Nearly all residual links have a matching address: the model needs an address-driven path (house number + street +
city/state agreement) that can accept a near-zero name score. The added `distinct` variant and leetspeak rules
(5ervices, 8usiness, lnc) were introduced after this residual analysis to shrink the second category.

## 6. Pitfalls (each one was hit or measured in this work)
1. **NFD/NFKD + strip-combining destroys Indic matras** ("लिमिटेड" -> "लमटड"; crude JW on native pairs 0.43).
   Always `transliterate()` first, `fold()` after. `normalize_ref.fold` also keeps Indic combining marks as a guard.
2. **unidecode is GPL-2.0+ and aksharamukha is AGPL-3.0** — never import them in the submitted code (MIT/Apache rule).
   `text-unidecode` is Artistic/GPL too. Use `anyascii` (ISC) or the stdlib transliterator in `normalize_ref.py`.
3. **Pandas NA parsing / csv quoting**: read TSVs with `quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False`
   (literal "NULL"/"None" are data, and names contain quotes). `normalize_ref` treats `None` input as ''.
4. **The native-token dictionary is essential but incomplete on test** (96.4% token coverage): test India has a new
   shop-word vocabulary (stores, traders, jewellers, bakery, ...). Keep the rule-transliteration fallback and use
   soft token matching (phonetic keys) in the scorer; do not hard-snap to a large vocabulary (58% wrong snaps).
5. **Southern scripts expand abbreviations** ("Pvt Ltd" -> "ప్రైవేట్ లిమిటెడ్") — canonicalise private/pvt and limited/ltd
   after dictionary lookup, and drop them in `name_core`, otherwise exact-core equality fails.
6. **"Saint" is a wrong expansion of "St" in US S2/S3** ("221 Seneca Saint") — map saint->st for US only; in France
   "St"/"Ste" are Saint/Sainte (city "St-Nazaire"), so street maps must be country-keyed.
7. **"MG Road" is Mahatma Gandhi Road**, not "Marg" — do not map `mg` (fixed in this run).
8. **2-letter codes are ambiguous across countries** (IN = Indiana vs India, CA, PA = Pennsylvania vs Professional
   Association, "OR" = Oregon vs Orissa code, "FL" = Florida vs floor). State maps are applied ONLY to whole comma
   components and 2-letter codes are never mapped under the generic (unknown-country) fallback.
9. **Multiple state-like components**: "Washington, DC", "Delhi, DL", "New York, New York" — take the LAST as state and
   keep earlier ones as city candidates (the previous draft took the first -> "Washington, DC" became WA).
10. **India S1 uses "Orissa"**, S2/S3 use "OD"/"ଓଡ଼ିଶା"; S1 also mixes Calcutta/Kolkata and New Delhi/Delhi. Canonicalise both sides.
11. **India house numbers are messy** ("4-5-6", "31-7-15/A", "B-1201", "Hn 352" injected into S2/S3, "Door No", "Plot").
    `house_base` equality works for US (80.7% of true links vs 72.8% crude-first-number) but is WORSE than crude for
    India (54.0% vs 63.3%) because numbers often sit behind designators; for India compare `all_numbers`
    (base-number overlap 81.4% vs 3.5% on random pairs).
12. **India street detection is weak** (only ~30% of India addresses have a recognisable street type): use
    `other_tokens` Jaccard (AUC 0.976) rather than `street_tokens` (AUC 0.70) for India.
13. **Hyphen handling**: keep '-' / '/' only when glued between digits or a 1-2 letter prefix and a digit
    ("B-1201", "4-5-6", "15-A"); split elsewhere ("Pune - 411001", "Sector-21", "Lège-Cap-Ferret").
14. **Alias names are unmatchable by name** (US 1.7%, India 2.3% of true links share no token with S1, e.g.
    "Delta Rivernorth" vs "Solxyloyuma"); ~97% of them have address similarity >= 0.8 -> the matcher MUST allow
    address-only matches (house number + street + city/state agreement) with a name score near 0.
15. **Generic words are injected** ("Young Empire" -> "Young Services", "Girwa Private Limited Center"): center,
    services, service, partners are the most-added tokens on true links. Use `name_variants()['distinct']`
    or IDF-weighted token similarity so these words carry little weight; do not treat them as mismatches.
16. **Alias markers put the real name LAST** ("Fluxlum Sys fka Integrated Beyond"): dba, aka, fka, née, formerly
    (known as), trading as, t/a — 100% of 1,007 measured cases. `split_dba` keeps the part after the marker; keep the
    alias part too if you want a "has alias" feature (`split_dba(...)[1]`).
17. **France is test-only and generic-token-heavy**: legal forms (SARL 73k, SAS 52k S1 names) and generic words (club,
    amicale, comite, ecole, maison ...) dominate names; only 15 cities -> city blocks are huge. Normalized
    house-number+street keys exist for 93% of France S2/S3 records and hit a France S1 key for 63% (vs 8% for the
    crude full-address string), similar to the US calibration (66% recall at 99.8% precision on mini train).
18. **Speed**: calling `normalize_name`, `name_core`, `name_key`, `name_glued` separately recomputes everything
    (732 µs/name in the old draft). Use `name_variants` / `address_all` and normalise DISTINCT strings only.
19. **Machine noise**: timings on this laptop vary ±50% because other agents' Python processes use 0.5-1.1 GB and
    several CPU cores; benchmark on a quiet machine before sizing a cloud box.

## 7. Recommendations for the pipeline / SKILL.md writers
- Put `normalize_ref.py` (stdlib only) into `code/business_entity_resolution/src/` and regenerate the native-token
  dictionary from TRAIN true links inside the pipeline with `learn_native_dict` (53 s). Do not ship hand-edited data.
- Normalisation stage (per country partition, DISTINCT strings, multiprocessing pool of 6-8 workers, chunks of ~50k):
  `name_variants(name, country)` -> norm / core / key / glued / distinct; `address_all(addr, country)` -> normalized
  address + parts. Store as parquet columns; ~24M records ≈ 12-15 min on 8 processes.
- Blocking keys that were measured to be precise (mini train, exact key, same country):
  US `num+street` (recall 66.3%, precision 99.8%), `name_key` (63.5%, 97.1%), `addr_key` (47.0%, 99.9%);
  India `name_key` (71.0%, 95.7%), `name_glued` (72.4%, 95.8%), `allnum+other` (34.9%, 99.9%).
  Crude keys are far worse (US name 30.4%, India name 18.5%, addresses 7-9%). Union several keys + fuzzy/ANN blocking.
- Pairwise features that separate true from random/hard pairs best (AUC_hard, final module): name_core_JW,
  name_glued_SIM (domain forms), name_best = max(core TSR, glued_sim, norm TSR), state_norm_EQ, num_any_EQ,
  street_JAC (US), other_JAC (India), city_norm_EQ / city_fuzzy, addr_norm_JAC. Keep crude versions too; the model
  can learn when normalisation over-merges.
- For native-script names keep a second representation with `skeleton_key` tokens (TSR 92 without dictionary) as a
  safety net for words the dictionary lacks.
- Unknown countries: every function works with `country_key(...) == 'generic'`; never filter or one-hot {US, India};
  France is handled through its own maps, but everything still runs if the label is something else.
