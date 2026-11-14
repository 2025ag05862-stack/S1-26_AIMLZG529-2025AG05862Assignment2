"""Validation stage: staging -> validated tables + quarantined error datasets + report."""
import json

import pandas as pd

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger
from src.validation import validators as V

log = get_logger("validation")


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    proc, errs, reports = path_for(cfg, "processed"), path_for(cfg, "errors"), path_for(cfg, "reports")
    v = cfg["validation"]
    ref_date = cfg["reference_date"]
    users = pd.read_parquet(proc / "staging_users.parquet")
    items = pd.read_parquet(proc / "staging_items.parquet")
    inter = pd.read_parquet(proc / "staging_interactions.parquet")

    ok_u, err_u, sum_u = V.validate_users(users, v)
    ok_i, err_i, sum_i = V.validate_items(items, v, pd.Timestamp(ref_date).year)
    # referential integrity must be checked against the *validated* parents
    ok_x, err_x, sum_x = V.validate_interactions(inter, v, ref_date, set(ok_u.user_id), set(ok_i.item_id))

    for name, df in [("users", ok_u), ("items", ok_i), ("interactions", ok_x)]:
        df.to_parquet(proc / f"validated_{name}.parquet", index=False)
    for name, df in [("users", err_u), ("items", err_i), ("interactions", err_x)]:
        df.to_csv(errs / f"{name}_errors.csv", index=False)

    summary = pd.DataFrame(sum_u + sum_i + sum_x)
    summary.to_csv(reports / "validation_summary.csv", index=False)
    totals = {}
    for name, raw, ok in [("users", users, ok_u), ("items", items, ok_i), ("interactions", inter, ok_x)]:
        totals[name] = {"rows_in": len(raw), "rows_valid": len(ok), "rows_rejected": len(raw) - len(ok),
                        "pct_rejected": round(100 * (len(raw) - len(ok)) / len(raw), 2)}
    (reports / "validation_report.json").write_text(json.dumps(totals, indent=2))
    _write_markdown(summary, totals, reports / "validation_report.md")
    for t, s in totals.items():
        log.info("%-12s in=%d valid=%d rejected=%d (%.2f%%)", t, s["rows_in"], s["rows_valid"], s["rows_rejected"], s["pct_rejected"])
    return totals


def _write_markdown(summary: pd.DataFrame, totals: dict, path):
    lines = ["# Data Validation Report", "",
             "Policy: `error` rules quarantine the record into `data/errors/<table>_errors.csv`; "
             "`warning` rules keep the record, coerce the bad value to NA and log it (fixed in preparation).", "",
             "## Overall", "", "| Table | Rows in | Rows valid | Rows rejected | % rejected |", "|---|---:|---:|---:|---:|"]
    for t, s in totals.items():
        lines.append(f"| {t} | {s['rows_in']} | {s['rows_valid']} | {s['rows_rejected']} | {s['pct_rejected']} |")
    lines += ["", "## Rule-level results", "",
              "| Table | Rule | Column | Severity | Rows failed | % failed | Description |", "|---|---|---|---|---:|---:|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.table} | {r.rule} | {r.column} | {r.severity} | {r.rows_failed} | {r.pct_failed} | {r.description} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
