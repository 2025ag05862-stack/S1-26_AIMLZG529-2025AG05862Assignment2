"""Data ingestion: read raw CSVs *as strings* (so bad values survive untouched),
check the expected schema, and write immutable staging files + a manifest.

Raw files in data/raw are never modified. If they do not exist, the synthetic
generator is called, so a fresh clone runs end-to-end with one command.
"""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger
from src.ingestion import generate_synthetic

log = get_logger("ingestion")

EXPECTED_COLUMNS = {
    "users": ["user_id", "age", "gender", "country", "signup_date"],
    "items": ["item_id", "title", "category", "genres", "release_year"],
    "interactions": ["user_id", "item_id", "rating", "timestamp"],
}


def _md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cfg=None, force_regenerate: bool = False) -> dict:
    cfg = cfg or load_config()
    raw, processed, reports = path_for(cfg, "raw"), path_for(cfg, "processed"), path_for(cfg, "reports")
    if force_regenerate or not all((raw / f"{t}.csv").exists() for t in EXPECTED_COLUMNS):
        log.info("raw CSVs missing (or regeneration forced) -> generating synthetic data")
        generate_synthetic.run(cfg)

    manifest = {"ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": "synthetic (src/ingestion/generate_synthetic.py)", "files": {}}
    for table, cols in EXPECTED_COLUMNS.items():
        src = raw / f"{table}.csv"
        # dtype=str + only empty string = NA: nothing is coerced or silently fixed at this stage
        df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[""])
        missing = [c for c in cols if c not in df.columns]
        if missing:                                    # schema drift is a hard failure
            raise ValueError(f"{src.name}: missing required columns {missing}")
        df = df[cols]
        df.to_parquet(processed / f"staging_{table}.parquet", index=False)
        manifest["files"][table] = {"rows": int(len(df)), "columns": cols, "md5": _md5(src),
                                    "bytes": src.stat().st_size}
        log.info("ingested %-12s rows=%d", table, len(df))
    (reports / "ingestion_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    run()
