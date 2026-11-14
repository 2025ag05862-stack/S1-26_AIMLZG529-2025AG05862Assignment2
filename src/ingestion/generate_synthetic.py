"""Synthetic data generator (used because no external dataset download is required).

It simulates a movie-style platform with a latent-factor structure, so personalised
models CAN beat a popularity baseline, then deliberately corrupts a small fraction of
rows so the validation stage has real problems to find. Everything is seeded.
"""
import numpy as np
import pandas as pd

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger

log = get_logger("ingestion.generate")

COUNTRIES = ["India", "USA", "UK", "Germany", "Brazil", "Japan"]


def _build_clean(cfg, rng):
    s = cfg["synthetic"]
    n_u, n_i, k = s["n_users"], s["n_items"], s["n_latent_factors"]
    cats = s["categories"]
    start, end = pd.Timestamp(s["start_date"]), pd.Timestamp(s["end_date"])

    # ---- items: category embedding + noise gives items a latent vector
    cat_emb = rng.normal(size=(len(cats), k))
    item_cat = rng.integers(0, len(cats), n_i)
    Q = 0.7 * cat_emb[item_cat] + 0.5 * rng.normal(size=(n_i, k))
    Q /= Q.std()
    ranks = rng.permutation(n_i) + 1
    log_pop = -0.9 * np.log(ranks)                 # zipf-like popularity
    item_bias = rng.normal(0, 0.4, n_i)
    genres = []
    for c in item_cat:
        extra = rng.choice(len(cats), size=rng.integers(0, 3), replace=False)
        genres.append("|".join([cats[c]] + [cats[e] for e in extra if e != c]))
    items = pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(1, n_i + 1)],
        "title": [f"Item Title {i:04d}" for i in range(1, n_i + 1)],
        "category": [cats[c] for c in item_cat],
        "genres": genres,
        "release_year": rng.integers(1980, 2024, n_i),
    })

    # ---- users: two favourite categories + noise
    fav = np.stack([rng.choice(len(cats), 2, replace=False) for _ in range(n_u)])
    P = 0.8 * cat_emb[fav].mean(axis=1) + 0.5 * rng.normal(size=(n_u, k))
    P /= P.std()
    user_bias = rng.normal(0, 0.4, n_u)
    signup = start + pd.to_timedelta(rng.integers(0, 730, n_u), unit="D")
    users = pd.DataFrame({
        "user_id": [f"U{u:05d}" for u in range(1, n_u + 1)],
        "age": rng.integers(18, 70, n_u),
        "gender": rng.choice(["M", "F", "Other"], n_u, p=[0.48, 0.48, 0.04]),
        "country": rng.choice(COUNTRIES, n_u),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })

    # ---- interactions: Gumbel-top-k sampling without replacement per user
    dot = P @ Q.T / np.sqrt(k)
    n_per_user = np.clip(rng.lognormal(3.0, 0.7, n_u).astype(int), 8, 250)
    rows = []
    for u in range(n_u):
        score = log_pop + 1.2 * dot[u] + rng.gumbel(size=n_i)
        chosen = np.argsort(-score)[: n_per_user[u]]
        rating = 3.3 + user_bias[u] + item_bias[chosen] + 0.7 * dot[u, chosen] + rng.normal(0, 0.6, len(chosen))
        rating = np.clip(np.rint(rating), 1, 5).astype(int)
        span = (end - signup[u]).total_seconds()
        ts = signup[u] + pd.to_timedelta(rng.random(len(chosen)) * span, unit="s")
        rows.append(pd.DataFrame({"user_id": users.user_id[u], "item_id": items.item_id.values[chosen],
                                  "rating": rating, "timestamp": ts}))
    inter = pd.concat(rows, ignore_index=True)
    inter["timestamp"] = inter["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return users, items, inter


def _pick(rng, pool, frac, n):
    """Take `frac * n` indices from the shrinking pool (so corruptions never overlap)."""
    m = int(round(frac * n))
    chosen, rest = pool[:m], pool[m:]
    return chosen, rest


def _dirty_interactions(df, cfg, rng):
    d = cfg["synthetic"]["dirty"]["interactions"]
    df = df.astype(object).copy()
    n = len(df)
    pool = rng.permutation(n)

    def take(key):
        nonlocal pool
        idx, pool = _pick(rng, pool, d[key], n)
        return idx

    df.loc[take("missing_rating"), "rating"] = np.nan
    df.loc[take("missing_timestamp"), "timestamp"] = np.nan
    df.loc[take("missing_user_id"), "user_id"] = np.nan
    df.loc[take("invalid_rating_value"), "rating"] = rng.choice([0, 6, 7, -1, 11], size=int(round(d["invalid_rating_value"] * n)))
    df.loc[take("non_numeric_rating"), "rating"] = rng.choice(["five", "N/A", "good"], size=int(round(d["non_numeric_rating"] * n)))
    df.loc[take("unparseable_timestamp"), "timestamp"] = rng.choice(["not_a_date", "31/02/2023", "0000-00-00"], size=int(round(d["unparseable_timestamp"] * n)))
    df.loc[take("future_timestamp"), "timestamp"] = "2031-05-01 10:00:00"
    df.loc[take("too_old_timestamp"), "timestamp"] = "1999-01-01 00:00:00"
    df.loc[take("orphan_user"), "user_id"] = [f"U9{x:04d}" for x in rng.integers(0, 9999, int(round(d["orphan_user"] * n)))]
    df.loc[take("orphan_item"), "item_id"] = [f"I9{x:03d}" for x in rng.integers(0, 999, int(round(d["orphan_item"] * n)))]

    # valid re-ratings (same user+item rated again later) and exact duplicates are APPENDED
    clean_idx = pool
    rr_idx = rng.choice(clean_idx, int(round(d["re_rating"] * n)), replace=False)
    rr = df.loc[rr_idx].copy()
    ts = pd.to_datetime(rr["timestamp"]) + pd.to_timedelta(rng.integers(1, 60, len(rr)), unit="D")
    rr["timestamp"] = ts.clip(upper=pd.Timestamp(cfg["synthetic"]["end_date"])).dt.strftime("%Y-%m-%d %H:%M:%S")
    rr["rating"] = rng.integers(1, 6, len(rr))
    dup = df.loc[rng.choice(n, int(round(d["exact_duplicate"] * n)), replace=False)]
    out = pd.concat([df, rr, dup], ignore_index=True)
    return out.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)


def _dirty_users(df, cfg, rng):
    d = cfg["synthetic"]["dirty"]["users"]
    df = df.astype(object).copy()
    n = len(df)
    pool = rng.permutation(n)

    def take(key):
        nonlocal pool
        idx, pool = _pick(rng, pool, d[key], n)
        return idx

    df.loc[take("missing_age"), "age"] = np.nan
    df.loc[take("invalid_age"), "age"] = rng.choice(["abc", -5, 230], size=int(round(d["invalid_age"] * n)))
    df.loc[take("missing_country"), "country"] = np.nan
    g = take("messy_gender")
    df.loc[g, "gender"] = rng.choice(["male", "Female", "m", "f", "MALE", "other"], size=len(g))
    c = take("messy_country")
    df.loc[c, "country"] = [x.upper() if i % 2 else f" {x.lower()} " for i, x in enumerate(df.loc[c, "country"].astype(str))]
    df.loc[take("bad_signup_date"), "signup_date"] = "2023-13-45"
    df.loc[take("missing_user_id"), "user_id"] = np.nan
    dup = df.loc[rng.choice(n, int(round(d["exact_duplicate"] * n)), replace=False)]
    conflict = df.loc[rng.choice(pool, int(round(d["conflicting_duplicate"] * n)), replace=False)].copy()
    conflict["country"] = "Unknownland"
    return pd.concat([df, dup, conflict], ignore_index=True).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)


def _dirty_items(df, cfg, rng):
    d = cfg["synthetic"]["dirty"]["items"]
    df = df.astype(object).copy()
    n = len(df)
    pool = rng.permutation(n)

    def take(key):
        nonlocal pool
        idx, pool = _pick(rng, pool, d[key], n)
        return idx

    df.loc[take("missing_category"), "category"] = np.nan
    df.loc[take("missing_title"), "title"] = np.nan
    df.loc[take("invalid_year"), "release_year"] = rng.choice(["unknown", 2090, 1500], size=int(round(d["invalid_year"] * n)))
    c = take("messy_category")
    df.loc[c, "category"] = [f" {x.upper()} " if i % 2 else x.capitalize() for i, x in enumerate(df.loc[c, "category"].astype(str))]
    dup = df.loc[rng.choice(n, int(round(d["exact_duplicate"] * n)), replace=False)]
    return pd.concat([df, dup], ignore_index=True).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    rng = np.random.default_rng(cfg["seed"])
    users, items, inter = _build_clean(cfg, rng)
    raw = path_for(cfg, "raw")
    out = {
        "users.csv": _dirty_users(users, cfg, rng),
        "items.csv": _dirty_items(items, cfg, rng),
        "interactions.csv": _dirty_interactions(inter, cfg, rng),
    }
    for name, df in out.items():
        df.to_csv(raw / name, index=False)
        log.info("wrote %s (%d rows)", raw / name, len(df))
    return {k: len(v) for k, v in out.items()}


if __name__ == "__main__":
    run()
