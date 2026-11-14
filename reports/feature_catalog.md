# Feature Catalog

Features are computed from the TRAIN split only, as of `2024-04-01 09:43:31+00:00`.

| Feature | Entity | Description | Logic | Type |
|---|---|---|---|---|
| `user_activity_count` | user | Number of training interactions by the user | `count(interactions)` | int |
| `user_avg_rating` | user | Mean rating given by the user | `mean(rating)` | float |
| `user_rating_std` | user | Spread of the user's ratings (0 if single rating) | `std(rating)` | float |
| `user_positive_ratio` | user | Share of the user's ratings that are positive (>= threshold) | `mean(rating >= 4)` | float |
| `user_last_interaction_ts` | user | Timestamp of user's most recent interaction (stored so recency can be recomputed at any as-of time) | `max(timestamp)` | datetime |
| `user_days_since_last_interaction` | user | Recency: days between the as-of time and the last interaction | `(as_of - user_last_interaction_ts).days` | float |
| `user_preferred_category` | user | Category with the most positive interactions (ties -> alphabetical) | `mode(category | rating >= 4)` | string |
| `user_age` | user | Cleaned user age (median-imputed when missing) | `users_clean.age` | int |
| `item_popularity` | item | Number of training interactions with the item | `count(interactions)` | int |
| `item_popularity_pct` | item | Popularity percentile among all items (0-1) | `rank(item_popularity, pct=True)` | float |
| `item_avg_rating` | item | Raw mean rating of the item (NaN if never rated) | `mean(rating)` | float |
| `item_bayes_rating` | item | Bayesian-smoothed mean rating (shrinks rarely-rated items to the global mean) | `(sum + m*global_mean)/(n + m)` | float |
| `item_positive_ratio` | item | Share of the item's ratings that are positive | `mean(rating >= 4)` | float |
| `item_category` | item | Primary category (cleaned, lower-case) | `items_clean.category` | string |
| `item_age_years` | item | Years between release year and the as-of year | `as_of.year - release_year` | int |
| `ui_category_affinity` | user-item | Share of the user's training interactions that fall in the item's category | `n(user, category(item)) / n(user)` | float |
| `ui_in_preferred_category` | user-item | 1 if the item's category equals the user's preferred category | `item_category == user_preferred_category` | int |
| `ui_baseline_rating` | user-item | Baseline rating estimate = user mean + item mean - global mean (clipped to 1-5) | `user_avg + item_bayes - global_mean` | float |
