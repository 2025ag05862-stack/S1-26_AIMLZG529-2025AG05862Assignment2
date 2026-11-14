"""Task registry: task name -> callable(cfg). Dependencies live in graph.py."""
from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.pipeline.graph import DEPENDENCIES  # noqa: F401  (re-exported for convenience)
from src.preparation import run_preparation
from src.training import train
from src.transformation import derived
from src.validation import run_validation

TASKS = {
    "ingest": ingest.run,
    "validate": run_validation.run,
    "prepare": run_preparation.run,
    "transform": derived.run,
    "build_features": feature_store.materialize,
    "train": train.run,
    "evaluate": evaluate.run,
}
