# -*- coding: utf-8 -*-
"""diag_fr_b.py -- normalization audit for French text (label-free) + residual token-difference tables.

B1  legal-form / generic-word survival: raw-token rate vs name_norm rate vs name_core rate (France S1 + pool sample)
B1b tokens over-represented in the pool vs S1 (injected noise words) per country
B2  residual differences on high-confidence selected pairs (p >= 0.98) and on uncertain pairs (0.3 <= p < 0.7):
    name-core tokens only on one side, address tokens only on one side, street-type confusion, house-number,
    city / state agreement -- France vs US vs India (test)
B3  abbreviation -> canonical token table (full word vs abbreviation) + raw frequency S1 vs pool
B4  accent residue, St/Ste/Saint, CEDEX / BP / postcodes, city_key / state_key distributions
Writes /vol/diag/fr/norm_audit.md
"""
import collections
import os
import re
import sys
import time

import polars as pl

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import normalize as N  # noqa: E402

V3 = "/vol/work_v3"
OUT = "/vol/diag/fr"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
MD = []


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def md(*lines):
    MD.extend(lines)


def table(header, rows):
    md("| " + " | ".join(header) + " |", "|" + "---|" * len(header))
    for r in rows:
        md("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |")
    md("")


TOK = re.compile(r"[0-9a-z°]+")


def rtoks(s):
    return TOK.findall(N.fold(s or ""))


COLS = ["entity_id", "business_name", "business_address", "name_norm", "name_core", "addr_norm", "house_nums",
        "city_key", "state_key"]


def load(country, side, n=None, seed=1):
    lf = pl.scan_parquet(f"{V3}/norm/test_{country}_{side}.parquet").select(COLS)
    df = lf.collect()
    if n is not None and df.height > n:
        df = df.sample(n, seed=seed)
    return df


# ------------------------------------------------------------------------------------------------ B1
WORDS = ["sarl", "sas", "sasu", "sa", "eurl", "ei", "eirl", "sci", "snc", "scop", "scp", "selarl", "gie", "cie",
         "compagnie", "ets", "etablissements", "etablissement", "fils", "freres", "associes", "societe", "ste",
         "association", "asso", "club", "comite", "groupe", "groupement", "holding", "participations", "centre",
         "maison", "france", "international", "distribution", "developpement", "services", "institut", "federation",
         "amicale", "union", "ecole", "the", "le", "la", "les", "de", "du", "des", "d", "l", "et", "and"]
fr_s1 = load("France", "s1")
fr_pool = load("France", "pool", 200_000)
log("loaded France", fr_s1.height, fr_pool.height)
md("# France normalization audit (test data, label-free)", "",
   f"France S1 {fr_s1.height:,} rows (all), pool sample {fr_pool.height:,} rows.", "")
md("## B1 legal forms / generic words: share of names containing the token", "",
   "raw = token in fold(raw name) split on non-alphanumerics (dotted S.A.R.L. is NOT caught here); norm = token in name_norm; "
   "core = token in name_core (what the name similarity features compare). A legal form should have core ~ 0.", "")
rows = []
for side, df in (("S1", fr_s1), ("pool", fr_pool)):
    raw = [set(rtoks(x)) for x in df["business_name"].to_list()]
    nrm = [set(x.split()) for x in df["name_norm"].to_list()]
    cor = [set(x.split()) for x in df["name_core"].to_list()]
    n = len(raw)
    for w in WORDS:
        rows.append((side, w, sum(w in s for s in raw) / n, sum(w in s for s in nrm) / n, sum(w in s for s in cor) / n))
by = collections.defaultdict(dict)
for side, w, a, b, c in rows:
    by[w][side] = (a, b, c)
table(["token", "S1 raw", "S1 norm", "S1 core", "pool raw", "pool norm", "pool core"],
      [(w, *by[w]["S1"], *by[w]["pool"]) for w in WORDS])
# core length and share of core made only of generic words
GEN = {"club", "amicale", "association", "comite", "groupe", "maison", "centre", "societe", "ecole", "union", "federation",
       "institut", "france", "international", "distribution", "developpement", "services", "participations", "holding",
       "de", "du", "des", "la", "le", "les", "d", "l"}
for side, df in (("S1", fr_s1), ("pool", fr_pool)):
    cores = [x.split() for x in df["name_core"].to_list()]
    n = len(cores)
    md(f"- {side}: mean core tokens {sum(len(c) for c in cores) / n:.2f}; core has <=1 non-generic token "
       f"{sum(sum(t not in GEN for t in c) <= 1 for c in cores) / n:.3f}; core tokens of length<=3 share "
       f"{sum(sum(len(t) <= 3 for t in c) for c in cores) / max(1, sum(len(c) for c in cores)):.3f}")
md("")

# ------------------------------------------------------------------------------------------------ B1b
md("## B1b core tokens over-represented in S2/S3 vs S1 (injected noise words), per country", "",
   "ratio = (pool doc-freq share) / (S1 doc-freq share); tokens with >= 300 pool occurrences in the sample.", "")
for country in ("France", "US", "India"):
    s1 = fr_s1 if country == "France" else load(country, "s1", 250_000)
    pool = fr_pool if country == "France" else load(country, "pool", 250_000)
    c1 = collections.Counter(t for x in s1["name_core"].to_list() for t in set(x.split()))
    c2 = collections.Counter(t for x in pool["name_core"].to_list() for t in set(x.split()))
    n1, n2 = s1.height, pool.height
    rat = [(t, c2[t] / n2, c1[t] / n1, (c2[t] / n2) / max(c1[t] / n1, 0.5 / n1)) for t in c2 if c2[t] >= 300]
    up = sorted(rat, key=lambda r: -r[3])[:25]
    down = sorted(rat, key=lambda r: r[3])[:12]
    md(f"### {country}: top over-represented in pool")
    table(["token", "pool share", "S1 share", "ratio"], up)
    md(f"### {country}: top under-represented in pool (dropped / replaced by the noise)")
    table(["token", "pool share", "S1 share", "ratio"], down)
log("B1 done")

# ------------------------------------------------------------------------------------------------ B2
md("## B2 residual differences on selected / uncertain pairs (test, label-free)", "",
   "hi = selected links with p >= 0.98 (pseudo-true); unc = candidates with 0.3 <= p < 0.7. Shares are per pair.", "")
NUM = re.compile(r"\d")


def pair_frame(country, n_hi=60_000, n_unc=30_000):
    sc = pl.scan_parquet(f"{V3}/pred/test_{country}_scored.parquet")
    links = pl.scan_parquet(f"{V3}/pred/test_{country}_links.parquet").rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel"))
    hi = (sc.filter(pl.col("p") >= 0.98).join(links, on=["s1", "cand"], how="inner").collect())
    hi = hi.sample(min(n_hi, hi.height), seed=3).with_columns(pl.lit("hi").alias("grp"))
    unc = sc.filter((pl.col("p") >= 0.3) & (pl.col("p") < 0.7)).collect()
    unc = unc.sample(min(n_unc, unc.height), seed=3).with_columns(pl.lit(False).alias("sel"), pl.lit("unc").alias("grp"))
    P = pl.concat([hi.select("s1", "cand", "p", "grp"), unc.select("s1", "cand", "p", "grp")])
    s1 = pl.scan_parquet(f"{V3}/norm/test_{country}_s1.parquet").select(COLS).filter(pl.col("entity_id").is_in(P["s1"].unique().to_list())).collect()
    pool = pl.scan_parquet(f"{V3}/norm/test_{country}_pool.parquet").select(COLS).filter(pl.col("entity_id").is_in(P["cand"].unique().to_list())).collect()
    P = P.join(s1.rename({c: c + "_q" for c in COLS if c != "entity_id"}).rename({"entity_id": "s1"}), on="s1", how="left")
    P = P.join(pool.rename({c: c + "_c" for c in COLS if c != "entity_id"}).rename({"entity_id": "cand"}), on="cand", how="left")
    return P


def street_type(addr, country):
    try:
        return N.address_parts(addr, country)["street_type"] or "-"
    except Exception:
        return "ERR"


summary = []
for country in ("France", "US", "India"):
    P = pair_frame(country)
    log("pairs", country, P.height)
    for grp in ("hi", "unc"):
        G = P.filter(pl.col("grp") == grp)
        n = G.height
        if n == 0:
            continue
        only_q, only_c = collections.Counter(), collections.Counter()
        aq, ac = collections.Counter(), collections.Counter()
        st_conf = collections.Counter()
        hn_exact = hn_base = hn_both = empty_c = city_eq = city_both = state_eq = state_both = core_eq = 0
        for r in G.iter_rows(named=True):
            q, c = set(r["name_core_q"].split()), set((r["name_core_c"] or "").split())
            core_eq += q == c
            only_q.update(q - c)
            only_c.update(c - q)
            a1 = {t for t in r["addr_norm_q"].split() if not NUM.search(t)}
            a2 = {t for t in (r["addr_norm_c"] or "").split() if not NUM.search(t)}
            if r["addr_norm_c"]:
                aq.update(a1 - a2)
                ac.update(a2 - a1)
            else:
                empty_c += 1
            h1, h2 = set(r["house_nums_q"] or []), set(r["house_nums_c"] or [])
            if h1 and h2:
                hn_both += 1
                hn_exact += bool(h1 & h2)
                b1 = {re.sub(r"\D.*$", "", x) for x in h1}
                b2 = {re.sub(r"\D.*$", "", x) for x in h2}
                hn_base += bool(b1 & b2)
            if r["city_key_q"] and r["city_key_c"]:
                city_both += 1
                city_eq += r["city_key_q"] == r["city_key_c"]
            if r["state_key_q"] and r["state_key_c"]:
                state_both += 1
                state_eq += r["state_key_q"] == r["state_key_c"]
            if grp == "hi" and r["addr_norm_c"] and len(st_conf) < 10**6:
                t1, t2 = street_type(r["business_address_q"], country), street_type(r["business_address_c"], country)
                if t1 != t2:
                    st_conf[(t1, t2)] += 1
        summary.append((country, grp, n, core_eq / n, empty_c / n, hn_both / n, hn_exact / max(hn_both, 1),
                        hn_base / max(hn_both, 1), city_both / n, city_eq / max(city_both, 1), state_both / n,
                        state_eq / max(state_both, 1)))
        md(f"### {country} / {grp} (n={n:,})")
        md("name-core tokens only in S1 (top 20): " + ", ".join(f"{t} {c / n:.3f}" for t, c in only_q.most_common(20)))
        md("")
        md("name-core tokens only in candidate (top 20): " + ", ".join(f"{t} {c / n:.3f}" for t, c in only_c.most_common(20)))
        md("")
        md("address words only in S1 (top 20): " + ", ".join(f"{t} {c / n:.3f}" for t, c in aq.most_common(20)))
        md("")
        md("address words only in candidate (top 20): " + ", ".join(f"{t} {c / n:.3f}" for t, c in ac.most_common(20)))
        md("")
        if grp == "hi":
            md("street-type disagreement (S1 type -> cand type), share of hi pairs: " +
               ", ".join(f"{a}->{b} {c / n:.4f}" for (a, b), c in st_conf.most_common(15)))
            md("")
md("### summary")
table(["country", "grp", "n", "core_eq", "cand_addr_empty", "hn_both", "hn_exact|both", "hn_base|both", "city_both",
       "city_eq|both", "state_both", "state_eq|both"], summary)
log("B2 done")

# ------------------------------------------------------------------------------------------------ B3
md("## B3 abbreviation canonicalization (France)", "",
   "canon(x) = normalize_address('12 <x> Dupont, Lille', 'France') minus the fixed tokens; MISMATCH = full word and "
   "abbreviation map to different tokens. Raw rates = share of addresses containing the raw token.", "")
PAIRS = [("rue", "r"), ("boulevard", "bd"), ("boulevard", "bld"), ("boulevard", "blvd"), ("boulevard", "boul"),
         ("avenue", "av"), ("avenue", "ave"), ("avenue", "aven"), ("chemin", "ch"), ("chemin", "che"), ("chemin", "chem"),
         ("chemin", "chmn"), ("impasse", "imp"), ("place", "pl"), ("allée", "all"), ("allée", "al"), ("allée", "alle"),
         ("route", "rte"), ("route", "rt"), ("faubourg", "fbg"), ("faubourg", "fg"), ("quai", "qu"), ("quai", "q"),
         ("cours", "crs"), ("cours", "cour2"), ("cours", "cour"), ("square", "sq"), ("passage", "pas"), ("passage", "pass"),
         ("passage", "psg"), ("résidence", "res"), ("résidence", "resid"), ("lotissement", "lot"),
         ("zone artisanale", "za"), ("zone industrielle", "zi"), ("chaussée", "chau"), ("promenade", "prom"),
         ("rond point", "rpt"), ("hameau", "ham"), ("lieu dit", "ld"), ("saint", "st"), ("sainte", "ste"),
         ("général", "gal"), ("général", "gen"), ("docteur", "dr"), ("maréchal", "mal"), ("président", "pdt"),
         ("bâtiment", "bat"), ("appartement", "appt"), ("appartement", "apt"), ("étage", "et"), ("cité", "cit"),
         ("boîte postale", "bp")]
FIX = {"12", "dupont", "lille", "hauts", "de", "france"}


def canon(x):
    return " ".join(t for t in N.normalize_address(f"12 {x} Dupont, Lille", "France").split() if t not in FIX) or "(dropped)"


raw_s1 = [set(rtoks(x)) for x in fr_s1["business_address"].to_list()]
raw_pl = [set(rtoks(x)) for x in fr_pool["business_address"].to_list()]


def rate(tok, raws):
    t = N.fold(tok).split()[0]
    return sum(t in s for s in raws) / len(raws)


rows = []
for full, ab in PAIRS:
    cf, ca = canon(full), canon(ab)
    rows.append((full, ab, cf, ca, "MISMATCH" if cf != ca else "ok", rate(full, raw_s1), rate(full, raw_pl),
                 rate(ab, raw_s1), rate(ab, raw_pl)))
table(["full", "abbr", "canon(full)", "canon(abbr)", "status", "S1 full", "pool full", "S1 abbr", "pool abbr"], rows)
md("house-number suffix forms:")
for x in ["12 bis Rue X, Lille", "12bis Rue X, Lille", "12 B Rue X, Lille", "12B Rue X, Lille", "12 ter Rue X, Lille",
          "12 T Rue X, Lille", "12 a Rue X, Lille", "N° 12 Rue X, Lille", "Nº 12 Rue X, Lille", "(12) Rue X, Lille",
          "12 - Rue X, Lille", "0012 Rue X, Lille", "12 Rue X, Appt 8, Lille", "12 Rue X, Appartement 8, Lille",
          "12 Rue X, 6eme Etage, Lille", "12 Rue X, Bat F1, Lille", "303 Cour2 de la Somme, Bordeaux",
          "BP 45, 59000 Lille CEDEX", "12 Rue X, 59000 Lille"]:
    n_, p_ = N.address_all(x, "France")
    md(f"- `{x}` -> `{n_}` | house {p_['house_numbers']} | city {p_['city_candidates']} | type {p_['street_type']}")
md("")
# how often do 'bis/ter' spellings disagree between S1 and pool (raw)
for tok in ("bis", "ter", "b", "t", "a", "n°", "no", "cedex", "bp", "appt", "appartement", "apt", "etage", "bat", "cour2"):
    md(f"- raw token `{tok}`: S1 {rate(tok, raw_s1):.4f}  pool {rate(tok, raw_pl):.4f}")
md("")
log("B3 done")

# ------------------------------------------------------------------------------------------------ B4
md("## B4 accents, Saint, postcodes, city / state keys (France)", "")
for side, df in (("S1", fr_s1), ("pool", fr_pool)):
    for col in ("name_norm", "name_core", "addr_norm"):
        v = df[col].to_list()
        md(f"- {side} {col}: non-ASCII rows {sum(not x.isascii() for x in v) / len(v):.5f}")
for side, df in (("S1", fr_s1), ("pool", fr_pool)):
    nn = [set(x.split()) for x in df["name_norm"].to_list()]
    an = [set(x.split()) for x in df["addr_norm"].to_list()]
    n = len(nn)
    md(f"- {side} names: st {sum('st' in s for s in nn) / n:.4f} ste {sum('ste' in s for s in nn) / n:.4f} "
       f"saint {sum('saint' in s for s in nn) / n:.4f} sainte {sum('sainte' in s for s in nn) / n:.4f}")
    md(f"- {side} addr_norm: st {sum('st' in s for s in an) / n:.4f} ste {sum('ste' in s for s in an) / n:.4f} "
       f"saint {sum('saint' in s for s in an) / n:.4f} sainte {sum('sainte' in s for s in an) / n:.4f} "
       f"cedex {sum('cedex' in s for s in an) / n:.4f} 5-digit {sum(any(len(t) == 5 and t.isdigit() for t in s) for s in an) / n:.4f} "
       f"empty {sum(len(s) == 0 for s in an) / n:.4f}")
md("")
for side, df in (("S1", fr_s1), ("pool", fr_pool)):
    ck = df["city_key"].value_counts().sort("count", descending=True)
    n = df.height
    md(f"- {side} city_key: empty {(df['city_key'] == '').sum() / n:.4f}, distinct {ck.height}; top: " +
       ", ".join(f"{a} {b / n:.3f}" for a, b in ck.head(18).rows()))
    sk = df["state_key"].value_counts().sort("count", descending=True)
    md(f"- {side} state_key: " + ", ".join(f"'{a}' {b / n:.4f}" for a, b in sk.head(8).rows()))
s1c = set(fr_s1["city_key"].to_list())
md(f"- pool city_key not among S1 city keys (non-empty): "
   f"{fr_pool.filter((pl.col('city_key') != '') & ~pl.col('city_key').is_in(list(s1c))).height / fr_pool.height:.4f}; examples: "
   + ", ".join(fr_pool.filter((pl.col('city_key') != '') & ~pl.col('city_key').is_in(list(s1c)))['city_key'].value_counts()
               .sort('count', descending=True).head(15)['city_key'].to_list()))
md("")
with open(f"{OUT}/norm_audit.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("done")
