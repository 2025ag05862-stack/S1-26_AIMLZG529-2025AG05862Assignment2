"""Draws the workflow graph coloured by task status (a static view of a run, Airflow-graph style)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.pipeline.graph import DEPENDENCIES

POS = {"ingest": (0, 1), "validate": (1, 1), "prepare": (2, 1), "transform": (3, 1.6),
       "build_features": (3, 0.4), "train": (4, 1), "evaluate": (5, 1)}
COLORS = {"success": "#2e9e5b", "failed": "#d64545", "upstream_failed": "#e0a030"}


def draw(record: dict, out_path):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(-0.6, 5.7); ax.set_ylim(-0.3, 2.2); ax.axis("off")
    for task, ups in DEPENDENCIES.items():
        for u in ups:
            ax.annotate("", xy=(POS[task][0] - 0.38, POS[task][1]), xytext=(POS[u][0] + 0.38, POS[u][1]),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6))
    for task, (x, y) in POS.items():
        info = record["tasks"].get(task, {})
        st = info.get("status", "unknown")
        ax.add_patch(FancyBboxPatch((x - 0.38, y - 0.22), 0.76, 0.44, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=COLORS.get(st, "#999"), ec="black", lw=1.2))
        ax.text(x, y + 0.04, task, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
        ax.text(x, y - 0.12, f"{st}" + (f"  {info['seconds']}s" if info.get("seconds") not in (None, "") else ""), ha="center", va="center", color="white", fontsize=7.5)
    ax.set_title(f"Pipeline run  |  started {record['started_utc']}  |  "
                 f"{'ALL TASKS SUCCEEDED' if record['success'] else 'RUN FAILED'}", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
