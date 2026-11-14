# Data Validation Report

Policy: `error` rules quarantine the record into `data/errors/<table>_errors.csv`; `warning` rules keep the record, coerce the bad value to NA and log it (fixed in preparation).

## Overall

| Table | Rows in | Rows valid | Rows rejected | % rejected |
|---|---:|---:|---:|---:|
| users | 1521 | 1497 | 24 | 1.58 |
| items | 606 | 600 | 6 | 0.99 |
| interactions | 40020 | 38205 | 1815 | 4.54 |

## Rule-level results

| Table | Rule | Column | Severity | Rows failed | % failed | Description |
|---|---|---|---|---:|---:|---|
| users | U01_user_id_not_null | user_id | error | 3 | 0.197 | user_id must be present |
| users | U02_user_id_unique | user_id | error | 21 | 1.381 | duplicate user_id (first occurrence kept) |
| users | U03_age_missing | age | warning | 30 | 1.972 | age missing (imputed later) |
| users | U04_age_valid | age | warning | 15 | 0.986 | age must be numeric in [10, 100] |
| users | U05_country_missing | country | warning | 22 | 1.446 | country missing (filled later) |
| users | U06_gender_standard | gender | warning | 232 | 15.253 | gender not in ['M', 'F', 'Other'] (standardised later) |
| users | U07_signup_date_parseable | signup_date | warning | 8 | 0.526 | signup_date must be a valid date |
| items | I01_item_id_not_null | item_id | error | 0 | 0.0 | item_id must be present |
| items | I02_item_id_unique | item_id | error | 6 | 0.99 | duplicate item_id (first occurrence kept) |
| items | I03_title_missing | title | warning | 6 | 0.99 | title missing |
| items | I04_category_missing | category | warning | 12 | 1.98 | category missing (set to 'unknown') |
| items | I05_release_year_valid | release_year | warning | 6 | 0.99 | release_year must be numeric in [1900, 2024] |
| interactions | X01_user_id_not_null | user_id | error | 79 | 0.197 | user_id must be present |
| interactions | X01_item_id_not_null | item_id | error | 0 | 0.0 | item_id must be present |
| interactions | X01_rating_not_null | rating | error | 197 | 0.492 | rating must be present |
| interactions | X01_timestamp_not_null | timestamp | error | 119 | 0.297 | timestamp must be present |
| interactions | X02_rating_numeric | rating | error | 118 | 0.295 | rating must be numeric (data type) |
| interactions | X03_rating_range | rating | error | 237 | 0.592 | rating must be in [1, 5] |
| interactions | X04_timestamp_parseable | timestamp | error | 118 | 0.295 | timestamp must be a valid datetime |
| interactions | X05_timestamp_in_range | timestamp | error | 199 | 0.497 | timestamp must be within [2022-01-01, 2024-07-01] (no future / ancient dates) |
| interactions | X06_exact_duplicate | all | error | 394 | 0.985 | exact duplicate row (first occurrence kept) |
| interactions | X07_user_referential_integrity | user_id | error | 205 | 0.512 | user_id not found in validated users table |
| interactions | X08_item_referential_integrity | item_id | error | 158 | 0.395 | item_id not found in validated items table |
