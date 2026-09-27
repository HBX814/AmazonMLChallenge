> Research notes captured 2026-09-25 while building these skills (measured on the real data on the team laptop). Paths such as `research/...`, `scratchpad/...` or `blocking_exp/...` point to a temporary scratch folder that is NOT part of the project — rely on the numbers and conclusions, and re-run measurements with the project code when needed. Sections marked PENDING/TODO were not finished.

# Amazon ML Challenge 2026 — Business Entity Resolution — MEASURED DATA FACTS
(Measured 2026-09-25 with DuckDB on the full files. Treat these as ground truth; do not re-guess.)

## Paths
- Project root: C:\Users\harsh\Downloads\AmazonMLChallenge
- Data: student_resource/dataset/{train,test}/{train,test}_source{1,2,3}.tsv, train/train_ground_truth.tsv
- Validator: student_resource/utils/validate_submission.py (stdlib; checks header, dup rows, dup ids in list, S1 ids in lists, S2/S3 prefix, missing/extra S1 rows; --check-ids optional)
- Doc template: student_resource/Documentation_template.md
- Parquet copies (zstd, all columns VARCHAR) already exist in the session scratchpad: pq/{train,test}_source{1,2,3}.parquet, pq/train_ground_truth.parquet, pq/gt_long.parquet (columns s1, mid)
- Project venv: C:\Users\harsh\Downloads\AmazonMLChallenge\.venv (Python 3.10; has duckdb 1.5.5, polars 1.44.2, pyarrow, pandas, numpy, rapidfuzz 3.14.5, unidecode, psutil). Global Python 3.10 has torch 2.6 CPU, sentence-transformers 3.2.1, transformers 4.50, faiss-cpu 1.10, scikit-learn 1.7.2, groq 0.25.
- AWS CLI is NOT installed. aws-core MCP servers failed to connect in this session.

## Machine
- Windows 11 laptop, Intel i5-1235U (10 cores / 12 threads), 15.7 GB RAM (often only ~1.5-4 GB free because of other apps), NO CUDA GPU (Intel UHD iGPU only), ~25 GB free disk.
- => Anything full-scale must be sharded / out-of-core, or run on a rented cloud box (AWS credits ~$100-200 available).

## File format gotchas
- TAB separated, header row, UTF-8. Fields contain commas, apostrophes, quotes (e.g. "D’EAU", "D' ATHENES"). Read with quoting disabled (pandas: sep='\t', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, na_filter=False; duckdb: delim='\t', quote='', escape='', all_varchar=true).
- Literal strings "NULL", "null", "None" appear INSIDE addresses as noise tokens (e.g. "PLEASANT RUN ROAD, NULL, IRVING, TX"). Pandas default NA parsing would also turn some values into NaN — disable it.
- Empty address occurs (S2/S3 ~2.3-3.7%); S1 never has empty name/address. Names never empty.
- entity_id format: S1-/S2-/S3- + random integer (NOT sequential, not zero-padded, variable length e.g. S1-965667, S1-925783039). IDs are unique within each file. No ID overlap between train and test.

## Sizes (rows, excl. header)
| file | US | India | France | total |
|---|---|---|---|---|
| train_source1 | 1,323,633 | 883,188 | 0 | 2,206,821 |
| train_source2 | 3,016,817 | 2,017,799 | 0 | 5,034,616 |
| train_source3 | 3,170,056 | 2,115,547 | 0 | 5,285,603 |
| test_source1 | 663,106 | 809,986 | 259,452 | 1,732,544 (header line excluded) |
| test_source2 | 1,871,330 | 2,312,565 | 703,378 | 4,887,273 |
| test_source3 | 1,945,701 | 2,405,000 | 731,615 | 5,082,316 |
- File sizes: ~200 MB (S1) and ~500 MB (S2, S3) each TSV; parquet ~65-200 MB each.
- (S2+S3)/S1 ratio: train 4.68 for both US and India; test 5.82 India, 5.76 US, 5.53 France => test pools are ~23% denser per S1 than train (more distractors or more matches per entity — unknown). Precision risk: monitor predicted links per S1 on test vs validation.

## Ground truth structure (train)
- train_ground_truth.tsv has exactly one row per train S1 entity (2,206,821 rows). Empty matched_entity_ids = singleton.
- Matches per S1 (k): 0: 5.58% | 1: 5.40% | 2: 17.00% | 3: 24.05% | 4: 21.94% | 5: 14.59% | 6: 7.47% | 7: 2.90% | 8: 0.85% | 9: 0.19% | 10: 0.02% | 11: ~0. Mean k = 3.46. Singleton rate identical in US and India (5.58%).
- Most matched S1 have both S2 and S3 matches; S2 and S3 each contain MULTIPLE duplicate records of the same entity (up to ~5 per source). So Source 2 and Source 3 are NOT deduplicated internally.
- EXCLUSIVITY: every S2/S3 id appears in AT MOST ONE S1's list (7,638,365 links, 7,638,365 distinct ids, 0 ids linked to >1 S1). => A hard 1-to-many constraint: assign each S2/S3 record to at most one S1.
- Linked fraction: S2 3,693,619 / 5,034,616 (73.4%); S3 3,944,746 / 5,285,603 (74.6%). ~25% of S2/S3 records are unlinked distractors.
- Country agreement: 100% of links have identical country label on both sides (US-US, India-India). Country is a safe hard partition key. (Test adds France; assume same.)
- All GT ids exist in the source files.

## Difficulty of true links (crude normalisation: lowercase, strip accents, drop punctuation)
| country | exact normalised name equal | mean Jaro-Winkler(name) | JW>=0.9 | JW<0.7 | matched addr empty | mean char-Jaccard(addr) |
|---|---|---|---|---|---|---|
| US | 30.5% | 0.927 | 78.8% | 4.5% | 4.7% | 0.89 |
| India | 18.8% | 0.811 | 54.0% | 23.7% | 3.9% | 0.85 |
- India low-similarity links are mostly NATIVE-SCRIPT names (Devanagari, Kannada, Tamil, Telugu, Malayalam, Gurmukhi, Bengali, Gujarati) of Latin S1 names, e.g. "Fortune Finance Private Limited" vs "ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "Aditya Tech LLP" vs Tamil script.
- US low-similarity links include pure ALIAS names (random gibberish e.g. "Vimaon Partners" vs "Nylagildxylo", "Katz Seafood" vs "Belomirabrix") that match only via identical address, and token reorders ("Urology Elite Physicians" vs "Physicians Elite Urology").
- Low name sim AND empty address among true links: US 0.03%, India 0.11% (essentially unmatchable).
- Unlinked (distractor) S2/S3 records whose normalised name exactly equals some S1 name in same country: US 6.1%, India 4.4% (vs linked 35.2% / 20.3%). Distractors are mostly different entities; some share generic names.

## Scripts / formatting noise (percent of rows)
| file/country | Devanagari name | other Indic name | Indic in address | accented Latin in name | ALLCAPS name | ALLCAPS address | junk leading punct in name | domain-like name (.com etc.) |
|---|---|---|---|---|---|---|---|---|
| S1 (all) | 0 | 0 | 0 | 0 (France S1 15.7%, legit French accents) | 0 | 0 | 0.14 | 0 |
| S2 India | 13.4 | 10.2 | 23.7 | 4.4 | 37.5 | 23.9 | 1.7 | 3.4 |
| S2 US | 0 | 0 | 0 | 6.7 | 21.6 | 89.8 | 2.5 | 4.4 |
| S3 India | 7.5 | 5.7 | 22.5 | 5.3 | 14.1 | 0.04 | 1.8 | 3.6 |
| S3 US | 0 | 0 | 0 | 6.8 | 3.2 | 0 | 2.4 | 4.2 |
| test S2 France | 0 | 0 | 0 | 24.5 | 20.8 | 29.0 | 0.9 | 3.5 |
| test S3 France | 0 | 0 | 0 | 23.9 | 5.6 | 0 | 0.85 | 3.4 |
- S1 is always clean Latin script, Title Case, canonical-ish format. S2 tends to UPPERCASE addresses (US 90%). S3 uses full state names ("Texas", "Ohio") where S1 uses 2-letter codes; India S3 uses state abbreviations (RJ, DL, MH, KA, UP) where S1 uses full names; India S2/S3 often write the STATE in native script (राजस्थान, ಕರ್ನಾಟಕ, दिल्ली, महाराष्ट्र).
- GOTCHA: naive accent stripping (NFD + drop combining marks) DESTROYS Indic vowel signs (matras) — e.g. "लिमिटेड" -> "लमटड". Transliterate Indic → Latin FIRST, fold Latin accents only afterwards.

## Observed noise operators (same generator applied to all countries incl. France test)
Names: injected random accents on Latin letters ("Ésports", "Ássociates", "DÀNCE", "Çlub", "Frèrés"); typos/char substitutions ("Tetlecommunication", "Dnmoss", "Cihooern", "Projtds", "PINVARTE"); double spaces; ALLCAPS/lowercase; token order shuffles ("Netra Limited Private Technologies", "Of S. Levine DPM Wilow Doylestown"); legal suffix added/removed/duplicated/moved to front ("LLC LLC", "LLC Womens Health...", "Private Exports Dharani Water Limited", "Pvt. EFS Print Ventures Ltd.", "SAS Bordeaux Club"); suffix abbreviation swaps (Private Limited/Pvt Ltd/Pvt/Ltd; L.L.C./LLC; S.A.R.L./SARL; S.A.S/SAS); honorific/prefix insertion (Dr, Sri, Shri, Smt, Mr, M/s, "des", "The"); junk leading symbols (">>", "--", "<<", "@", "#"); bracket insertions ("[[FFSAHIEGX]]", "[Ltd]", "[Club]", "(France)"); DBA prefixes ("Onyxcalovantage Sys D.B.A. Unified Pinnacle Esports L.L.C."); domain form ("unifiedpinnacleesports.com", "prantechcorporate.com - 9895970478" incl. phone numbers; "@COMITCERCLE"); truncation (drop trailing tokens: "Unified Pinnacle", "Express Data"); generic word substitution ("Caouette Services" for "Caouette Cardiology", "Girwa Private Limited Center"); complete alias names (gibberish); native-script transliteration (India).
Addresses: UPPERCASE; literal "NULL"/"null"/"None" tokens; missing components (no house number, no city, no state, empty); component reordering ("MA, SWANSEA, 25 RONALD DR", "TOURCOING, 35 RUE D' ATHENES, Nord"); abbreviations (Street/St, Road/Rd, Drive/Dr, Lane/Ln, Avenue/Ave, Court/Ct; Rue/R./R, Avenue/AV./AVE, Boulevard/BD; No./NO./N°/#); WRONG expansions ("221 Seneca Saint" for Street); house-number perturbation (1804 -> 1804A, 3412 -> 3411, 495 -> 95, 213 -> 13) — so house numbers are strong but not exact; city typos ("CRNING", "WAARBA", "MDEINA", "Niwsa"); duplicated city tokens; state code<->name (NY/New York, TX/Texas; Rajasthan/RJ/राजस्थान; Delhi/DL/दिल्ली); prefixes like "#800", "Hn 352", "Door No", "Plot", "Flat", "Shop No"; landmark references ("Near Sbi Atm", "Opp ...", "Behind ...") in ~12% of India S1 addresses; France: region in S1 ("Hauts-de-France", "Nouvelle-Aquitaine", "Pays de la Loire") vs DEPARTMENT in S2/S3 ("Nord", "Gironde", "Loire-Atlantique") or missing; "bis"/"ter"/"B" house suffixes; elisions "D'", "L'", "D’".

## Postcodes / geography
- US addresses: only ~10.8% contain a 5-digit number (ZIP rarely present). S1 US format: "<num> <street>, <city>, <ST>" (+ optional "Unit ..."). Last comma component of S1 US address = 2-letter state.
- India: 6-digit PIN present in only ~0.3% (S1) / ~1.1% (S2/S3) — PIN blocking is useless. S1 India format ends with ", <City>, <State full name>" (sometimes district). Addresses long (mean 78 chars S1).
- France (test only): S1 format "<num> [bis|ter] <Rue/Avenue/Boulevard/Place/Allée/Impasse...> <name>, <City>, <Region>". Only a handful of cities appear (Calais, Lille, Roubaix, Tourcoing, Dunkerque, Bordeaux, Pessac, Mérignac, La Teste-de-Buch, Lège-Cap-Ferret, Nantes, Saint-Nazaire, Saint-Herblain, La Baule-Escoublac, ...) in 3 regions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire) => city blocks are HUGE (tens of thousands of entities per city); need name/street-level blocking inside city.
- France names are built from a small generic vocabulary (sarl 73k, sas 52k, club, france, ecole, amicale, comite, maison, union, sportive, college, pharmacie, federation, ...) + legal forms SARL, SAS, SASU, EURL, SA, SCI, EI, Cie, Ets/Établissements, "& Fils", "& Frères". Generic-token-heavy => rare-token/IDF weighting and street+number matching are essential.

## Name ambiguity
- Many S1 entities share the same normalised name, esp. US generic medical names ("primary care group" x253, "ear nose throat group" x251, "pediatric group" x222, "meridian llc" x184). Address must disambiguate. India S1 normalised-name multiplicity also has a long tail.
- Train S1 vs test S1: 0 exact (name,address) overlaps, 0 id overlaps => disjoint entities, no leakage to exploit.

## Top name tokens (normalised) — for legal-suffix / stopword lists
- US S1: llc, inc, and, s, c, l (from L.L.C.), care, of, associates, center, group, partners, p, d, corp, pc, health, clinic, pllc, medicine, lp, pediatric, dental, family, global, specialists, a, physicians, capital, co, holdings, corporation, incorporated.
- US S2/S3 add: com, ltd, the, service, summit, highland, valley, coastal, harbor, metro.
- India S1: limited, private, ltd, pvt, india, llp, services, solutions, brothers, trading, co, technologies, international, foundation, global, tech, industries, enterprises, consultants.
- India S2/S3 add: com, center, public, exports, sri, shri, dr, mr, smt, m (M/s), s, holdings, infratech, overseas, corporation, company + native-script legal words (लिमिटेड, प्राइवेट, ಲಿಮಿಟೆಡ್, ప్రైవేట్, லிமிடெட், ...).
- France test: sarl, sas, club, france, de, eurl, ecole, amicale, comite, sa, maison, du, sasu, centre, union, sci, sportive, des, college, amis, primaire, federation, pharmacie, cie, sante, saint, etablissements, la, parents, ets, freres, lycee, fils, maternelle, ei, societe, association, groupe, developpement, participations, distribution, holding, international.
- Top address tokens US S2/S3: st, rd, street, dr, road, drive, ave, city, avenue, new, tx, texas, north, ny, york, ln, nc, carolina, oh, lane, unit, ohio, virginia, ct, null, il, illinois, washington, way, va, tn.
- Top address tokens India S2/S3: no, delhi, road, nagar, floor, mumbai, plot, maharashtra, mh, west, bangalore, pradesh, city, door, pune, महाराष्ट्र, kolkata, block, dl, near, hyderabad, colony, south, sector, flat, east, chennai, north, house, uttar, दिल्ली, street, karnataka, ka, up, 2nd, main, 1st, tamil.

## Example true-match groups (verbatim)
US S1-222736164 "Unified Pinnacle Esports L.L.C." | "221 Seneca Street, Corning, NY"
  S2 "Unified Pinnacle Esports  L.L.C." | (empty)
  S2 "Unified Pinnacle Esports" | "221 SENECA STREET, CRNING, NY"
  S2 "Unified Pinnacle Ésports L.L.C." | "221 SENECA STREET, CRNING, NY"
  S3 "Onyxcalovantage Sys D.B.A. Unified Pinnacle Esports L.L.C." | "221 Seneca Street, Corning, NY"
  S3 "Unified Pinnacle" | "221 Seneca Street, New York, Corning"
  S3 "unifiedpinnacleesports.com" | "221 Seneca Saint, Corning, New York"
India S1-925694678 "Girwa Projects Private Limited" | "Niwasa, Near Sbi Atm, Punjawati Bhuwana, Girwa, Udaipur, Rajasthan"
  S2 "GIRWA  PROJECTS PRIVATE" | "राजस्थान, NIWASA, NEAR SBI ATM, PUNJAWATI BHUWANA, GIRWA"
  S2 "GIRWA PRIVATE LIMITED CENTER" | same
  S2 "Girwa Projtds Private Limited" | "NIWASA, NEAR SBI ATM, PUNJAWATI BHUWANA, GIRWA, राजस्थान"
  S3 "Girwa Projects  Private Limited" | "Hn 352 Niwasa, Udaipur, RJ, Girwa"
India S1-536075008 "Exports Dharani Water Private Limited" | "2685, Iind Floor, Naya Bazar, Delhi"
  S2 "EXPORTS  DHARANI WATER PRIVATE" | (empty)
  S3 "Arcmira" | "2685, Naya Bazar, Delhi, DL"   <- alias, address-only match
  S3 "Private Exports Dharani Water Limited" | "2685, Naya Bazar, Delhi, DL"
France test S1: "Chretien Groupe SASU" | "121 Rue Neuve, Calais, Hauts-de-France"; "Guiclan & Frères EURL" | "11 Avenue de la Tramontane Pyla, La Teste-de-Buch, Nouvelle-Aquitaine"
France test S2/S3: "@COMITCERCLE" | "8 R. DE LA PORTE D’EAU, DUNKERQUE, Nord"; "Compagnons  Sport S.A.R.L." | "8 BIS RUE PAUL BERT, SAINT-NAZAIRE"; "des Union Retro" | "N°34 AVE DES CARAVELLES, Lège-Cap-Ferret, Gironde"

## Metric reminders (from problem statement)
- Per-S1 F0.5 = 1.25 P R / (0.25 P + R), macro-averaged over ALL test S1 (incl. singletons). Truth empty & pred empty = 1.0; truth empty & pred non-empty = 0.0; truth non-empty & pred empty = 0.0 (P undefined, R = 0).
- Example: pred {a,b,c}, truth {a,c} -> P=2/3, R=1 -> 0.714.
- With mean k=3.46 and only 5.6% singletons, predicting empty for a matched entity costs a full 1.0; a precise partial list (e.g. 2 of 4 correct, no FP) scores 0.833.

## Rules (hard)
- Final model must be MIT or Apache-2.0 licensed and <= 8B parameters.
- NO external databases/APIs/services to look up businesses or resolve entities, NO geocoding APIs, NO external data augmentation from internet sources. Only provided data. Violations => disqualification. Code is audited.
- Deliverables: output/matching_results.tsv (scored), output/candidate_pairs.tsv (the exact final candidate set the model scored; matches must be a subset), code/business_entity_resolution/{src/,README.md,requirements.txt}, Documentation_template.md filled. Zip: <team_name>_submission.zip. Team name: "Master Bolt" (use Master_Bolt_submission.zip).
- Treat country as an open set: never hard-code/filter/one-hot {US, India}; every test S1 (incl. France) needs exactly one row.
