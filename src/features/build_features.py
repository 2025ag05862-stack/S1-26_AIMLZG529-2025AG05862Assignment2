"""Feature computation (pure functions). The SAME functions feed offline training and
online/batch prediction through FeatureStore, so there is no train/serve skew."""
import pandas as pd


def compute_user_features(train, users, items, thr):
    x = train.merge(items[["item_id", "category"]], on="item_id", how="left")
    g = x.groupby("user_id")
    f = g.agg(user_activity_count=("item_id", "size"), user_avg_rating=("rating", "mean"),
              user_rating_std=("rating", "std"), user_last_interaction_ts=("timestamp", "max")).reset_index()
    f["user_rating_std"] = f.user_rating_std.fillna(0.0)
    f["user_positive_ratio"] = f.user_id.map(x.assign(p=x.rating >= thr).groupby("user_id").p.mean())
    pos = x[x.rating >= thr].groupby(["user_id", "category"]).size().reset_index(name="n")
    pos = pos.sort_values(["user_id", "n", "category"], ascending=[True, False, True]).drop_duplicates("user_id")
    f["user_preferred_category"] = f.user_id.map(pos.set_index("user_id").category).fillna("unknown")
    f = f.merge(users[["user_id", "age"]].rename(columns={"age": "user_age"}), on="user_id", how="left")
    return f


def compute_item_features(train, items, thr, m, as_of):
    g = train.groupby("item_id")
    f = g.agg(item_popularity=("user_id", "size"), item_avg_rating=("rating", "mean"),
              _sum=("rating", "sum")).reset_index()
    f = items[["item_id", "category", "release_year"]].merge(f, on="item_id", how="left")
    gm = train.rating.mean()
    f["item_popularity"] = f.item_popularity.fillna(0).astype(int)
    f["item_bayes_rating"] = (f._sum.fillna(0) + m * gm) / (f.item_popularity + m)
    f["item_positive_ratio"] = f.item_id.map(train.assign(p=train.rating >= thr).groupby("item_id").p.mean()).fillna(0.0)
    f["item_popularity_pct"] = f.item_popularity.rank(pct=True)
    f["item_age_years"] = as_of.year - f.release_year
    return f.rename(columns={"category": "item_category"}).drop(columns=["_sum", "release_year"])


def compute_user_category_affinity(train, items):
    x = train.merge(items[["item_id", "category"]], on="item_id", how="left")
    counts = x.groupby(["user_id", "category"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)
