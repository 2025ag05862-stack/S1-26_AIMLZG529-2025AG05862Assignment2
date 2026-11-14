"""Draws the pipeline architecture diagram -> reports/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.common.config import load_config, path_for

BOXES = [  # (x, y, w, h, title, detail, colour)
    (0.2, 6.2, 2.6, 1.3, "1. Ingestion", "src/ingestion\nCSV / synthetic -> staging\n+ manifest (row counts, md5)", "#dbe9f6"),
    (3.4, 6.2, 2.6, 1.3, "2. Validation", "src/validation\n23 rules (types, ranges,\nduplicates, referential)", "#dbe9f6"),
    (6.6, 6.2, 2.6, 1.3, "3. Preparation", "src/preparation\nclean + standardise,\ntime-based train/test split", "#dbe9f6"),
    (9.8, 7.1, 2.8, 1.2, "4. Transformation", "src/transformation\nactivity summary, popularity,\nuser-item matrix", "#e3f1df"),
    (9.8, 5.2, 2.8, 1.2, "5. Feature management", "src/features\nuser / item / interaction features\n(Parquet feature store)", "#e3f1df"),
    (9.8, 3.1, 2.8, 1.3, "6. Model training", "src/training  +  MLflow\npopularity | SVD MF | hybrid\n(tuned on validation slice)", "#fbe9cf"),
    (6.6, 3.1, 2.6, 1.3, "7. Evaluation", "src/evaluation\nP/R/NDCG@10, coverage,\ndiversity, RMSE, MAE", "#fbe9cf"),
    (3.4, 3.1, 2.6, 1.3, "Reports", "reports/\nmodel_comparison, validation,\nfeature catalog, run graph", "#eee"),
]
STORES = [
    (0.2, 4.4, 2.6, 0.9, "data/raw  (CSV)", "#f6f6f6"),
    (3.4, 4.7, 2.6, 0.9, "data/errors  (quarantine)", "#fde0e0"),
    (6.6, 4.7, 2.6, 0.9, "data/processed (Parquet)", "#f6f6f6"),
    (9.8, 1.3, 2.8, 0.9, "models/  +  MLflow runs", "#f6f6f6"),
    (13.3, 5.2, 2.4, 1.2, "data/features (Parquet)", "#f6f6f6"),
    (13.3, 7.1, 2.4, 1.2, "data/processed:\nactivity, popularity,\nuser-item matrix (.npz)", "#f6f6f6"),
]


def _box(ax, x, y, w, h, title, detail, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.2, detail, ha="center", va="center", fontsize=7.6)


def _arrow(ax, a, b, **kw):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#444", **kw))


def draw(out_path):
    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    ax.set_xlim(0, 16); ax.set_ylim(0.8, 9.0); ax.axis("off")
    for b in BOXES:
        _box(ax, *b)
    for x, y, w, h, t, c in STORES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="#888", lw=1, ls="--"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5, style="italic")
    # main flow
    _arrow(ax, (2.8, 6.85), (3.4, 6.85)); _arrow(ax, (6.0, 6.85), (6.6, 6.85))
    _arrow(ax, (9.2, 6.85), (9.8, 7.6)); _arrow(ax, (9.2, 6.7), (9.8, 5.9))
    _arrow(ax, (11.2, 5.2), (11.2, 4.4))
    ax.plot([12.6, 13.0, 13.0], [7.7, 7.7, 3.75], color="#444", lw=1.5)          # transformation -> training (routed around)
    ax.plot([13.0, 12.6], [3.75, 3.75], color="#444", lw=1.5); _arrow(ax, (12.7, 3.75), (12.6, 3.75))
    _arrow(ax, (9.8, 3.75), (9.2, 3.75)); _arrow(ax, (6.6, 3.75), (6.0, 3.75))
    # data stores
    _arrow(ax, (1.5, 5.3), (1.5, 6.2)); _arrow(ax, (4.7, 6.2), (4.7, 5.6)); _arrow(ax, (7.9, 6.2), (7.9, 5.6))
    _arrow(ax, (11.2, 3.1), (11.2, 2.2))
    _arrow(ax, (12.6, 5.8), (13.3, 5.8)); _arrow(ax, (12.6, 7.7), (13.3, 7.7))
    ax.text(7.8, 0.95, "Orchestration: src/pipeline (dependency-aware runner)  and  dags/recommendation_pipeline_dag.py (Apache Airflow) "
            "- both execute the same task graph", ha="center", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round", fc="#f3e8ff", ec="#7a4fb5"))
    ax.set_title("End-to-End Data Management Pipeline for a Recommendation System", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(cfg=None):
    cfg = cfg or load_config()
    draw(path_for(cfg, "reports") / "architecture.png")


if __name__ == "__main__":
    run()
