"""CatBoost Enhanced와 GRU OOF 결과를 같은 기준으로 비교한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "demo_1"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "demo1_gru"

TARGET = "target_next_week_withdrawn"
KEYS = [
    "code_module",
    "code_presentation",
    "id_student",
    "prediction_week",
    TARGET,
]
TOP_FRACTION = 0.20


def precision_recall_at_top_fraction(
    target: np.ndarray,
    probability: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> tuple[float, float, set[int]]:
    top_k = max(1, int(np.ceil(len(target) * fraction)))
    selected = np.argsort(-probability, kind="stable")[:top_k]
    precision = float(target[selected].mean())
    recall = float(target[selected].sum() / target.sum())
    return precision, recall, set(selected.tolist())


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if mask.any():
            result += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(probability[mask].mean())
            )
    return float(result)


def model_metrics(
    name: str,
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[dict[str, float | int | str], set[int]]:
    precision, recall, selected = precision_recall_at_top_fraction(
        target, probability
    )
    return (
        {
            "model": name,
            "rows": len(target),
            "target_count": int(target.sum()),
            "target_rate": float(target.mean()),
            "pr_auc": float(average_precision_score(target, probability)),
            "recall_at_top_20pct": recall,
            "precision_at_top_20pct": precision,
            "brier_score": float(brier_score_loss(target, probability)),
            "ece_10bin": expected_calibration_error(target, probability),
        },
        selected,
    )


def load_oof() -> pd.DataFrame:
    catboost_path = MODEL_DIR / "catboost_weekly_next_week_oof_predictions.csv"
    gru_path = MODEL_DIR / "gru_weekly_next_week_oof_predictions.csv"
    for path in (catboost_path, gru_path):
        if not path.is_file():
            raise FileNotFoundError(f"OOF 파일이 없습니다: {path}")

    catboost = pd.read_csv(catboost_path)
    gru = pd.read_csv(gru_path)
    merged = catboost.merge(
        gru,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_catboost", "_gru"),
    )
    if len(merged) != len(catboost) or len(merged) != len(gru):
        raise ValueError("CatBoost와 GRU OOF 행이 완전히 일치하지 않습니다.")
    if merged[KEYS[:-1]].duplicated().any():
        raise ValueError("비교 데이터에 복합키 중복이 있습니다.")
    return merged


def make_metric_figure(metrics: pd.DataFrame) -> None:
    names = ["CatBoost\nEnhanced", "GRU\n4-week"]
    colors = ["#2F6690", "#E07A2D"]
    columns = [
        ("pr_auc", "PR-AUC"),
        ("recall_at_top_20pct", "Recall@Top20%"),
        ("precision_at_top_20pct", "Precision@Top20%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for axis, (column, title) in zip(axes, columns):
        values = metrics[column].to_numpy()
        bars = axis.bar(names, values, color=colors)
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.25)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
    fig.suptitle("CatBoost Enhanced vs GRU: next-week withdrawal prediction")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "catboost_gru_metric_comparison.png", dpi=180)
    plt.close(fig)


def make_calibration_figure(
    target: np.ndarray,
    catboost_probability: np.ndarray,
    gru_probability: np.ndarray,
) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    for name, probability, color in (
        ("CatBoost Enhanced", catboost_probability, "#2F6690"),
        ("GRU", gru_probability, "#E07A2D"),
    ):
        observed, predicted = calibration_curve(
            target,
            probability,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", color=color, label=name)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed next-week withdrawal rate")
    axis.set_title("OOF probability calibration")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "catboost_gru_calibration.png", dpi=180)
    plt.close(fig)


def make_overlap_figure(overlap: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#6C8EBF", "#2F6690", "#E07A2D", "#B8B8B8"]
    bars = axis.bar(overlap["category"], overlap["positive_count"], color=colors)
    axis.set_ylabel("Actual next-week withdrawals")
    axis.set_title("True positives captured in each Top20% risk group")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, padding=3)
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "catboost_gru_true_positive_overlap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_oof()
    target = data[TARGET].to_numpy(dtype=np.int8)
    catboost_probability = data["catboost_oof_probability"].to_numpy(float)
    gru_probability = data["gru_oof_probability"].to_numpy(float)

    catboost_metrics, catboost_top = model_metrics(
        "CatBoost Enhanced (124 features)", target, catboost_probability
    )
    gru_metrics, gru_top = model_metrics(
        "GRU (recent 4 weeks, 11 features)", target, gru_probability
    )
    metrics = pd.DataFrame([catboost_metrics, gru_metrics])
    metrics.to_csv(
        MODEL_DIR / "catboost_gru_comparison_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    catboost_positive = {index for index in catboost_top if target[index] == 1}
    gru_positive = {index for index in gru_top if target[index] == 1}
    all_positive = set(np.flatnonzero(target).tolist())
    overlap = pd.DataFrame(
        {
            "category": [
                "Both",
                "CatBoost only",
                "GRU only",
                "Missed by both",
            ],
            "positive_count": [
                len(catboost_positive & gru_positive),
                len(catboost_positive - gru_positive),
                len(gru_positive - catboost_positive),
                len(all_positive - (catboost_positive | gru_positive)),
            ],
        }
    )
    overlap.to_csv(
        MODEL_DIR / "catboost_gru_top20_overlap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    catboost_rank = pd.Series(catboost_probability).rank(pct=True).to_numpy()
    gru_rank = pd.Series(gru_probability).rank(pct=True).to_numpy()
    blend_rows = []
    for catboost_weight in np.arange(0.0, 1.01, 0.1):
        blended_rank = (
            catboost_weight * catboost_rank
            + (1.0 - catboost_weight) * gru_rank
        )
        row, _ = model_metrics(
            f"rank_blend_catboost_{catboost_weight:.1f}", target, blended_rank
        )
        row["catboost_weight"] = catboost_weight
        row["gru_weight"] = 1.0 - catboost_weight
        blend_rows.append(row)
    blend = pd.DataFrame(blend_rows)
    blend.to_csv(
        MODEL_DIR / "catboost_gru_rank_blend_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_metric_figure(metrics)
    make_calibration_figure(target, catboost_probability, gru_probability)
    make_overlap_figure(overlap)

    print("===== CatBoost Enhanced vs GRU =====")
    print(metrics.to_string(index=False))
    print("\n===== Top20% 실제 이탈자 포착 관계 =====")
    print(overlap.to_string(index=False))
    print("\n예측확률 Pearson 상관:", round(float(np.corrcoef(catboost_probability, gru_probability)[0, 1]), 4))
    print("예측순위 Spearman 상관:", round(float(pd.Series(catboost_probability).corr(pd.Series(gru_probability), method="spearman")), 4))
    print("\n결론: 순위 혼합은 CatBoost 단독 PR-AUC를 넘는지 진단 CSV에서 확인합니다.")
    print("그래프:", FIGURE_DIR)


if __name__ == "__main__":
    main()
