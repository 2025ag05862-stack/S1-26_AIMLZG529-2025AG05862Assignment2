"""Lightweight workflow runner (no external services needed).

* resolves task order from DEPENDENCIES (topological sort)
* retries each task, logs duration/status, skips downstream tasks if an upstream one fails
* writes reports/pipeline_run.json and a run-graph image (reports/pipeline_run_graph.png)

    python -m src.pipeline.runner            # run everything
    python -m src.pipeline.runner --fresh    # wipe generated artefacts + regenerate raw data first
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from graphlib import TopologicalSorter

from src.common.config import ROOT, load_config, path_for
from src.common.logging_utils import get_logger
from src.pipeline.graph import DEPENDENCIES
from src.pipeline.tasks import TASKS

log = get_logger("runner")


def _fresh(cfg):
    """Remove every generated artefact (raw data too) so the run starts from nothing."""
    for key in ["raw", "processed", "features", "errors", "models"]:
        d = path_for(cfg, key)
        for p in d.iterdir():
            if p.name != ".gitkeep":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    for f in [ROOT / "mlflow.db"]:
        f.unlink(missing_ok=True)
    for d in ("mlartifacts", "mlruns"):                   # MLflow artefact stores
        shutil.rmtree(ROOT / d, ignore_errors=True)
    log.info("fresh start: generated artefacts removed")


def run_pipeline(config_path=None, fresh=False, retries=1) -> dict:
    cfg = load_config(config_path)
    reports = path_for(cfg, "reports")
    if fresh:
        _fresh(cfg)
    order = list(TopologicalSorter(DEPENDENCIES).static_order())
    log.info("execution order: %s", " -> ".join(order))
    status, record = {}, {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tasks": {}}
    for name in order:
        if any(status.get(u) != "success" for u in DEPENDENCIES[name]):
            status[name] = "upstream_failed"
            record["tasks"][name] = {"status": "upstream_failed", "upstream": DEPENDENCIES[name]}
            log.error("SKIP %s (upstream failed)", name)
            continue
        for attempt in range(1, retries + 2):
            t0 = time.time()
            try:
                log.info("START %s (attempt %d)", name, attempt)
                TASKS[name](cfg)
                status[name] = "success"
                record["tasks"][name] = {"status": "success", "seconds": round(time.time() - t0, 2), "attempts": attempt}
                log.info("DONE  %s in %.2fs", name, time.time() - t0)
                break
            except Exception as exc:                      # noqa: BLE001 - we want to log anything
                status[name] = "failed"
                record["tasks"][name] = {"status": "failed", "seconds": round(time.time() - t0, 2),
                                         "attempts": attempt, "error": repr(exc)}
                log.exception("FAIL  %s: %s", name, exc)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["success"] = all(s == "success" for s in status.values())
    (reports / "pipeline_run.json").write_text(json.dumps(record, indent=2))
    try:
        from src.reporting.make_run_graph import draw
        draw(record, reports / "pipeline_run_graph.png")
    except Exception as exc:                              # plotting must never fail the pipeline
        log.warning("run graph not drawn: %s", exc)
    log.info("PIPELINE %s", "SUCCEEDED" if record["success"] else "FAILED")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fresh", action="store_true", help="delete generated data/models and regenerate from scratch")
    a = ap.parse_args()
    sys.exit(0 if run_pipeline(a.config, a.fresh)["success"] else 1)
