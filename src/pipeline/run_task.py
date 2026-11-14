"""Run ONE pipeline task:  python -m src.pipeline.run_task <task_name>
Used by the Airflow DAG so Airflow and the ML libraries can live in separate environments."""
import sys

from src.common.config import load_config
from src.pipeline.tasks import TASKS

if __name__ == "__main__":
    name = sys.argv[1]
    if name not in TASKS:
        sys.exit(f"unknown task '{name}'. choose from: {list(TASKS)}")
    TASKS[name](load_config())
