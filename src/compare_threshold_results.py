"""여러 threshold 분석 JSON의 F1 추천 결과를 동일 형식으로 비교한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .data import PROJECT_ROOT


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "threshold_analysis"
METRIC_COLUMNS = [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "TP",
    "FP",
    "TN",
    "FN",
    "predicted_positive_count",
    "predicted_positive_ratio",
]


def compare_result_files(result_specs: list[str]) -> pd.DataFrame:
    """MODEL=JSON 형식 결과들의 best_f1 행을 하나의 표로 합친다."""
    rows: list[dict] = []
    for spec in result_specs:
        if "=" not in spec:
            raise ValueError(f"--result는 MODEL=JSON 형식이어야 합니다: {spec}")
        model_name, path_text = spec.split("=", 1)
        path = Path(path_text)
        if not model_name or not path.is_file():
            raise FileNotFoundError(f"모델명 또는 결과 JSON을 확인하세요: {spec}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        best_f1 = payload.get("selected", {}).get("best_f1")
        if not best_f1:
            raise ValueError(f"best_f1 결과가 없습니다: {path}")
        row = {
            "model": model_name,
            **{column: best_f1[column] for column in METRIC_COLUMNS},
            "threshold_search_min": payload["threshold_grid"]["requested_min"],
            "threshold_search_max": payload["threshold_grid"]["requested_max"],
            "threshold_search_step": payload["threshold_grid"]["requested_step"],
            "source_json": str(path.resolve()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["f1_score", "pr_auc", "model"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def save_comparison_plot(table: pd.DataFrame, output_path: Path) -> None:
    """모델별 추천 지점의 핵심 지표와 threshold를 한 장으로 비교한다."""
    models = table["model"].tolist()
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    width = 0.24
    for offset, metric, label, color in (
        (-width, "precision", "Precision", "tab:blue"),
        (0.0, "recall", "Recall", "tab:orange"),
        (width, "f1_score", "F1-score", "tab:green"),
    ):
        bars = axes[0].bar(x + offset, table[metric], width, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].set_title("Metrics at Recommended Threshold")
    axes[0].set_ylabel("Metric value")
    axes[0].set_xticks(x, models, rotation=18, ha="right")
    metric_max = float(table[["precision", "recall", "f1_score"]].max().max())
    axes[0].set_ylim(0, max(0.2, metric_max * 1.25))
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)

    pr_bars = axes[1].bar(models, table["pr_auc"], color="tab:purple")
    axes[1].bar_label(pr_bars, fmt="%.4f", padding=3)
    axes[1].set_title("OOF PR-AUC")
    axes[1].set_ylabel("PR-AUC")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylim(0, max(0.11, float(table["pr_auc"].max()) * 1.2))
    axes[1].grid(axis="y", alpha=0.25)

    threshold_bars = axes[2].bar(models, table["threshold"], color="tab:red")
    axes[2].bar_label(threshold_bars, fmt="%.3f", padding=3)
    axes[2].set_title("Recommended Model-specific Threshold")
    axes[2].set_ylabel("Threshold")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Binary Classifier Threshold Comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_comparison(table: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_threshold_comparison.csv"
    json_path = output_dir / "model_threshold_comparison.json"
    plot_path = output_dir / "model_threshold_comparison.png"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_comparison_plot(table, plot_path)
    payload = {
        "selection_basis": "highest F1 score from each model's refined threshold grid",
        "recommended_model": table.iloc[0]["model"],
        "recommended_threshold": float(table.iloc[0]["threshold"]),
        "models": table.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="MODEL=selected_thresholds.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        table = compare_result_files(args.result)
        csv_path, json_path, plot_path = save_comparison(table, args.output_dir)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(table.to_string(index=False))
    print(f"\nCSV: {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    print(f"PNG: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
