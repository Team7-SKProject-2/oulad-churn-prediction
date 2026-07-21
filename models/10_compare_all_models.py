"""CatBoost 124·108, GRU, TCN의 OOF 성능을 한 기준으로 비교한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "demo_1"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "final_model_comparison"
TARGET = "target_next_week_withdrawn"
KEYS = ["code_module", "code_presentation", "id_student", "prediction_week", TARGET]
TOP_FRACTION = 0.20

OOF_SPECS = {
    "CatBoost 124": (
        "catboost_weekly_next_week_oof_predictions.csv",
        "catboost_oof_probability",
    ),
    "CatBoost 108": (
        "catboost_reduced_feature_oof_predictions.csv",
        "catboost_reduced_oof_probability",
    ),
    "GRU 4-week": (
        "gru_weekly_next_week_oof_predictions.csv",
        "gru_oof_probability",
    ),
    "TCN 4-week": (
        "tcn_weekly_next_week_oof_predictions.csv",
        "tcn_oof_probability",
    ),
}


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            (probability >= lower) & (probability <= upper)
            if index == bins - 1
            else (probability >= lower) & (probability < upper)
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(probability[mask].mean())
            )
    return float(ece)


def top_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float, set[int]]:
    top_k = max(1, int(np.ceil(len(target) * TOP_FRACTION)))
    selected = np.argsort(-probability, kind="stable")[:top_k]
    true_positive = int(target[selected].sum())
    return (
        float(true_positive / target.sum()),
        float(true_positive / top_k),
        {int(index) for index in selected if target[index] == 1},
    )


def load_oof() -> tuple[pd.DataFrame, dict[str, str]]:
    merged: pd.DataFrame | None = None
    probability_columns: dict[str, str] = {}
    for model, (filename, probability_column) in OOF_SPECS.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"OOF 파일이 없습니다: {path}")
        current = pd.read_csv(path, usecols=[*KEYS, probability_column])
        renamed = f"probability_{len(probability_columns)}"
        current = current.rename(columns={probability_column: renamed})
        probability_columns[model] = renamed
        merged = (
            current
            if merged is None
            else merged.merge(current, on=KEYS, how="inner", validate="one_to_one")
        )
    if merged is None:
        raise RuntimeError("OOF 데이터가 비어 있습니다.")
    if len(merged) != 895_005:
        raise ValueError(f"비교 행 수가 다릅니다: {len(merged):,}")
    return merged, probability_columns


def calculate_comparison(
    data: pd.DataFrame,
    probability_columns: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    target = data[TARGET].to_numpy(dtype=np.int8)
    rows = []
    true_positive_sets: dict[str, set[int]] = {}
    for model, column in probability_columns.items():
        probability = data[column].to_numpy(float)
        recall, precision, true_positive = top_metrics(target, probability)
        true_positive_sets[model] = true_positive
        rows.append(
            {
                "model": model,
                "rows": len(target),
                "target_count": int(target.sum()),
                "target_rate": float(target.mean()),
                "pr_auc": float(average_precision_score(target, probability)),
                "recall_at_top_20pct": recall,
                "precision_at_top_20pct": precision,
                "brier_score": float(brier_score_loss(target, probability)),
                "ece_10bin": expected_calibration_error(target, probability),
            }
        )
    return pd.DataFrame(rows), true_positive_sets


def make_metric_figure(metrics: pd.DataFrame) -> None:
    colors = ["#214E7A", "#4C84B5", "#D97925", "#8A5AA9"]
    columns = [
        ("pr_auc", "PR-AUC", 1.0),
        ("recall_at_top_20pct", "Recall@Top20%", 100.0),
        ("precision_at_top_20pct", "Precision@Top20%", 100.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    labels = ["CatBoost\n124", "CatBoost\n108", "GRU\n4-week", "TCN\n4-week"]
    for axis, (column, title, multiplier) in zip(axes, columns):
        values = metrics[column].to_numpy() * multiplier
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.22)
        axis.grid(axis="y", alpha=0.25)
        number_format = "{:.4f}" if multiplier == 1.0 else "{:.2f}%"
        axis.bar_label(
            bars,
            labels=[number_format.format(value) for value in values],
            padding=3,
            fontsize=9,
        )
    fig.suptitle("OOF model comparison: next-week withdrawal prediction")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_oof_metric_comparison.png", dpi=180)
    plt.close(fig)


def make_correlation_figure(
    data: pd.DataFrame,
    probability_columns: dict[str, str],
) -> pd.DataFrame:
    renamed = data[list(probability_columns.values())].rename(
        columns={column: model for model, column in probability_columns.items()}
    )
    correlation = renamed.corr(method="spearman")
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(correlation, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if correlation.iloc[row, column] > 0.72 else "black",
            )
    axis.set_title("Spearman correlation of OOF risk ranking")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_oof_rank_correlation.png", dpi=180)
    plt.close(fig)
    return correlation


def make_risk_decile_figure(
    data: pd.DataFrame,
    probability_columns: dict[str, str],
) -> pd.DataFrame:
    rows = []
    target = data[TARGET]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    colors = ["#214E7A", "#4C84B5", "#D97925", "#8A5AA9"]
    for (model, column), color in zip(probability_columns.items(), colors):
        decile = pd.qcut(
            data[column].rank(method="first"),
            10,
            labels=range(1, 11),
        ).astype(int)
        observed = target.groupby(decile).mean() * 100
        axis.plot(
            observed.index,
            observed.values,
            marker="o",
            label=model,
            color=color,
        )
        for risk_decile, rate in observed.items():
            rows.append(
                {
                    "model": model,
                    "risk_decile": int(risk_decile),
                    "observed_withdrawal_rate_pct": float(rate),
                }
            )
    axis.set_xlabel("Risk decile (10 = highest risk)")
    axis.set_ylabel("Observed next-week withdrawal rate (%)")
    axis.set_title("Observed withdrawal rate by OOF risk decile")
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_oof_risk_deciles.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data, probability_columns = load_oof()
    metrics, true_positive_sets = calculate_comparison(data, probability_columns)
    metrics.to_csv(
        MODEL_DIR / "final_model_comparison_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlation = make_correlation_figure(data, probability_columns)
    correlation.to_csv(
        MODEL_DIR / "final_model_rank_correlations.csv",
        encoding="utf-8-sig",
    )
    deciles = make_risk_decile_figure(data, probability_columns)
    deciles.to_csv(
        MODEL_DIR / "final_model_risk_deciles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_metric_figure(metrics)

    complement = pd.DataFrame(
        {
            "deep_model": ["GRU 4-week", "TCN 4-week"],
            "true_positives_not_in_catboost_108_top20": [
                len(true_positive_sets["GRU 4-week"] - true_positive_sets["CatBoost 108"]),
                len(true_positive_sets["TCN 4-week"] - true_positive_sets["CatBoost 108"]),
            ],
            "true_positives_shared_with_catboost_108_top20": [
                len(true_positive_sets["GRU 4-week"] & true_positive_sets["CatBoost 108"]),
                len(true_positive_sets["TCN 4-week"] & true_positive_sets["CatBoost 108"]),
            ],
        }
    )
    complement.to_csv(
        MODEL_DIR / "deep_model_catboost_complement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("===== 최종 OOF 모델 비교 =====")
    print(metrics.to_string(index=False))
    print("\n===== CatBoost 108 대비 딥러닝 추가 포착 =====")
    print(complement.to_string(index=False))
    print("\n===== 위험순위 Spearman 상관 =====")
    print(correlation.round(4).to_string())
    print("\n그래프:", FIGURE_DIR)


if __name__ == "__main__":
    main()
