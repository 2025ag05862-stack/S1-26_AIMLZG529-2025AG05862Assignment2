"""Apache Airflow DAG for the recommendation pipeline.

The dependency graph comes from src/pipeline/graph.py (the SAME graph the local runner
executes). Each Airflow task calls `python -m src.pipeline.run_task <task>` through
BashOperator, using the interpreter given by the PIPELINE_PYTHON environment variable
(default: the interpreter running Airflow). This keeps Airflow's pinned dependencies and
the ML stack (pandas / mlflow / scipy ...) free to live in different virtualenvs.

Point Airflow's dags_folder at this directory (see README, "Run with Airflow").
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG

try:                                                   # Airflow 3.x
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:                                    # Airflow 2.x
    from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline.graph import DEPENDENCIES  # noqa: E402  (pure data; no heavy imports)

PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)

with DAG(
    dag_id="recommendation_pipeline",
    description="ingest -> validate -> prepare -> (transform | build_features) -> train -> evaluate",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dm4ml", "retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["recommender", "dm4ml"],
) as dag:
    operators = {
        name: BashOperator(task_id=name, bash_command=f'cd "{PROJECT_ROOT}" && "{PYTHON}" -m src.pipeline.run_task {name}')
        for name in DEPENDENCIES
    }
    for task, upstream in DEPENDENCIES.items():
        for u in upstream:
            operators[u] >> operators[task]
