"""The workflow graph (pure data, no heavy imports) - shared by the runner and the Airflow DAG.

task name -> upstream tasks.  `transform` and `build_features` are independent branches
that both depend on `prepare`; `train` needs both (matrix + feature store)."""
DEPENDENCIES = {
    "ingest": [],
    "validate": ["ingest"],
    "prepare": ["validate"],
    "transform": ["prepare"],
    "build_features": ["prepare"],
    "train": ["transform", "build_features"],
    "evaluate": ["train"],
}
