"""각 모델의 OOF 예측에서 F1-score가 최대인 분류 임계값을 정확히 찾습니다.

양성 클래스의 모든 고유 OOF 확률을 최적 임계값 후보로 사용합니다. 실행 옵션으로
지정하는 간격 임계값은 F1 변화 곡선을 그릴 때만 사용합니다. 모델별 OOF 파일은
복합키를 기준으로 정렬한 뒤 비교하므로 파일의 행 순서는 결과에 영향을 주지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .data import PROJECT_ROOT, require_columns
from .evaluate import compare_thresholds, generate_thresholds


matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402


TARGET_COLUMN = "target_next_week_withdrawn"
KEY_COLUMNS = ["code_module", "code_presentation", "id_student", "prediction_week"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "threshold_analysis"
MODEL_OUTPUT_SLUGS = {
    "XGBoost weighted": "xgboost",
}
DEFAULT_MODEL_SPECS = (
    # CatBoost: 학습 과정에서 저장한 OOF 양성 클래스 확률을 사용합니다.
    (
        "CatBoost",
        PROJECT_ROOT / "models" / "ML" / "catboost_weekly_next_week_oof_predictions.csv",
        "catboost_oof_probability",
    ),
    # 가중 XGBoost: scale_pos_weight를 적용해 생성한 OOF 확률을 사용합니다.
    (
        "XGBoost weighted",
        PROJECT_ROOT / "models" / "ML" / "xgboost_weekly_next_week_oof_predictions.csv",
        "xgboost_scaled_oof_probability",
    ),
    # ElasticNet: 로지스틱 ElasticNet 모델에서 저장한 OOF 양성 클래스 확률을 사용합니다.
    (
        "ElasticNet",
        PROJECT_ROOT
        / "models"
        / "ML"
        / "elasticnet_logistic_weekly_next_week_oof_predictions.csv",
        "elasticnet_logistic_oof_probability",
    ),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    probability_column: str


def parse_model_spec(value: str) -> ModelSpec:
    """'모델 이름|OOF CSV 경로|양성 확률 컬럼' 형식의 옵션을 해석합니다."""
    parts = value.split("|", maxsplit=2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--model은 '이름|OOF_CSV_경로|양성확률_컬럼' 형식이어야 합니다."
        )
    return ModelSpec(parts[0].strip(), Path(parts[1].strip()), parts[2].strip())


def default_model_specs() -> list[ModelSpec]:
    return [ModelSpec(name, path, column) for name, path, column in DEFAULT_MODEL_SPECS]


def load_aligned_oof(
    specs: list[ModelSpec],
    target_column: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    if len(specs) < 2:
        raise ValueError("비교할 OOF 모델을 두 개 이상 지정해야 합니다.")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("모델 이름은 중복될 수 없습니다.")

    aligned: pd.DataFrame | None = None
    internal_columns: dict[str, str] = {}
    sources: list[dict[str, Any]] = []

    for index, spec in enumerate(specs):
        if not spec.path.is_file():
            raise FileNotFoundError(f"{spec.name} OOF 파일이 없습니다: {spec.path}")
        selected_columns = [*KEY_COLUMNS, target_column, spec.probability_column]
        frame = pd.read_csv(spec.path, usecols=selected_columns)
        require_columns(frame, selected_columns, spec.path.name)
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{spec.path.name}에 복합키 중복 행이 있습니다.")
        labels = pd.to_numeric(frame[target_column], errors="raise")
        probabilities = pd.to_numeric(frame[spec.probability_column], errors="raise")
        if not np.isin(labels, [0, 1]).all() or labels.nunique() != 2:
            raise ValueError(f"{spec.path.name}의 {target_column}에는 0과 1이 모두 있어야 합니다.")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{spec.path.name}의 {spec.probability_column}이 유효한 확률이 아닙니다.")

        internal_probability = f"_probability_{index}"
        internal_target = f"_target_{index}"
        candidate = frame.rename(
            columns={
                target_column: internal_target,
                spec.probability_column: internal_probability,
            }
        )
        if aligned is None:
            aligned = candidate
            aligned = aligned.rename(columns={internal_target: target_column})
        else:
            aligned = aligned.merge(
                candidate,
                on=KEY_COLUMNS,
                how="inner",
                sort=False,
                validate="one_to_one",
            )
            if len(aligned) != len(frame):
                raise ValueError(f"{spec.name} OOF의 키가 다른 모델 OOF와 일치하지 않습니다.")
            if not aligned[target_column].eq(aligned[internal_target]).all():
                raise ValueError(f"{spec.name} OOF의 라벨이 다른 모델 OOF와 일치하지 않습니다.")
            aligned = aligned.drop(columns=[internal_target])

        internal_columns[spec.name] = internal_probability
        sources.append(
            {
                "model": spec.name,
                "oof_path": str(spec.path.resolve()),
                "probability_column": spec.probability_column,
                "rows": len(frame),
            }
        )

    if aligned is None:
        raise RuntimeError("OOF 데이터를 불러오지 못했습니다.")
    if len(aligned) == 0:
        raise ValueError("모델 간 공통 OOF 행이 없습니다.")

    y_true = aligned[target_column].to_numpy(dtype=np.int8)
    probability_by_model = {
        name: aligned[column].to_numpy(dtype=float)
        for name, column in internal_columns.items()
    }
    return y_true, probability_by_model, sources


def exact_f1_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float, int]:
    """모든 고유 OOF 확률 중 F1-score가 가장 높은 임계값을 반환합니다.

    예측 기준은 항상 ``probability >= threshold``입니다. 부동소수점 허용 오차
    범위에서 최대 F1-score가 같은 임계값이 여러 개면 0.5에 가까운 값을 우선하고,
    그 거리도 같으면 더 높은 임계값을 선택합니다.
    """
    labels = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.size != scores.size or labels.size == 0:
        raise ValueError("라벨과 확률은 길이가 같은 비어 있지 않은 1차원 배열이어야 합니다.")
    if not np.isin(labels, [0, 1]).all() or labels.sum() == 0:
        raise ValueError("라벨에는 양성(1)이 하나 이상 있어야 합니다.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("확률은 NaN 없이 0과 1 사이여야 합니다.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_predicted = np.arange(1, labels.size + 1, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    candidate_thresholds = sorted_scores[group_ends]
    tp = cumulative_tp[group_ends]
    predicted_positive = cumulative_predicted[group_ends]
    fp = predicted_positive - tp
    fn = int(labels.sum()) - tp
    denominator = 2 * tp + fp + fn
    f1_values = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    maximum_f1 = float(f1_values.max())
    tied = np.flatnonzero(np.isclose(f1_values, maximum_f1, rtol=0, atol=1e-12))
    tied_thresholds = candidate_thresholds[tied]
    best_threshold = min(tied_thresholds, key=lambda value: (abs(value - 0.5), -value))
    return float(best_threshold), maximum_f1, int(candidate_thresholds.size)


def evaluate_models(
    y_true: np.ndarray,
    probability_by_model: dict[str, np.ndarray],
    grid_thresholds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, Any]] = []
    grid_frames: list[pd.DataFrame] = []
    for model_name, probabilities in probability_by_model.items():
        exact_threshold, exact_f1, unique_count = exact_f1_optimal_threshold(
            y_true, probabilities
        )
        exact_metrics = compare_thresholds(
            y_true,
            probabilities,
            np.asarray([exact_threshold], dtype=float),
        ).iloc[0].to_dict()
        if not np.isclose(exact_metrics["f1_score"], exact_f1, rtol=0, atol=1e-12):
            raise RuntimeError(f"{model_name}의 exact F1 계산과 지표 계산 결과가 다릅니다.")
        selected_rows.append(
            {
                "model": model_name,
                "search_mode": "all_unique_oof_probabilities",
                "unique_probability_candidates": unique_count,
                **exact_metrics,
            }
        )

        grid = compare_thresholds(y_true, probabilities, grid_thresholds)
        grid.insert(0, "model", model_name)
        grid_frames.append(grid)

    selected = pd.DataFrame(selected_rows).sort_values(
        ["f1_score", "threshold"], ascending=[False, False], ignore_index=True
    )
    grid_table = pd.concat(grid_frames, ignore_index=True)
    return selected, grid_table


def _korean_font() -> font_manager.FontProperties | None:
    for family in ("Malgun Gothic", "NanumGothic"):
        try:
            return font_manager.FontProperties(
                fname=font_manager.findfont(family, fallback_to_default=False)
            )
        except ValueError:
            continue
    return None


def model_slug(model_name: str) -> str:
    if model_name in MODEL_OUTPUT_SLUGS:
        return MODEL_OUTPUT_SLUGS[model_name]
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return slug or "model"


def save_model_metrics_table(
    model_name: str,
    result: pd.Series,
    output_path: Path,
) -> None:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    metric_rows = [
        ("F1 최적 임계값", f"{result['threshold']:.9f}"),
        ("Accuracy", f"{result['accuracy']:.4%}"),
        ("Precision", f"{result['precision']:.4%}"),
        ("Recall", f"{result['recall']:.4%}"),
        ("Specificity", f"{result['specificity']:.4%}"),
        ("F1-score", f"{result['f1_score']:.4%}"),
        ("ROC-AUC", f"{result['roc_auc']:.6f}"),
        ("PR-AUC", f"{result['pr_auc']:.6f}"),
        ("TP / FP", f"{int(result['TP']):,} / {int(result['FP']):,}"),
        ("TN / FN", f"{int(result['TN']):,} / {int(result['FN']):,}"),
        ("양성 예측 개수", f"{int(result['predicted_positive_count']):,}"),
        ("양성 예측 비율", f"{result['predicted_positive_ratio']:.4%}"),
    ]
    figure, axis = plt.subplots(figsize=(8, 7.5))
    axis.axis("off")
    axis.set_title(f"{model_name} F1 최적 임계값 평가표", pad=16, **korean_text)
    table = axis.table(
        cellText=metric_rows,
        colLabels=["평가 지표", "값"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.08, 0.15, 0.84, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.35)
    if korean_font is not None:
        for cell in table.get_celld().values():
            cell.get_text().set_fontproperties(korean_font)
    axis.text(
        0.08,
        0.08,
        "모든 고유 OOF 확률을 후보로 검색한 정확한 F1 최대 지점입니다.",
        transform=axis.transAxes,
        fontsize=9.5,
        **korean_text,
    )
    axis.text(
        0.08,
        0.04,
        "ROC-AUC와 PR-AUC는 임계값 적용 전의 원래 OOF 확률로 계산합니다.",
        transform=axis.transAxes,
        fontsize=9.5,
        **korean_text,
    )
    # 표는 tight_layout의 자동 배치에 맡기면 긴 모델명에서 셀 텍스트가
    # 밀릴 수 있으므로 고정 여백을 사용해 모든 모델에서 같은 위치를 유지합니다.
    figure.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.06)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def save_model_threshold_curve(
    model_name: str,
    result: pd.Series,
    grid: pd.DataFrame,
    output_path: Path,
) -> None:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for column, label, color in (
        ("precision", "Precision", "tab:blue"),
        ("recall", "Recall", "tab:orange"),
        ("f1_score", "F1-score", "tab:green"),
    ):
        axis.plot(grid["threshold"], grid[column], color=color, linewidth=2, label=label)
    axis.axvline(
        result["threshold"],
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"정확한 F1 최적 임계값 ({result['threshold']:.6f})",
    )
    axis.scatter(
        result["threshold"],
        result["f1_score"],
        color="tab:red",
        s=75,
        zorder=3,
    )
    axis.annotate(
        f"F1={result['f1_score']:.4f}",
        (result["threshold"], result["f1_score"]),
        xytext=(8, 9),
        textcoords="offset points",
    )
    axis.set_title(f"{model_name}: 임계값별 Precision·Recall·F1", **korean_text)
    axis.set_xlabel("분류 임계값", **korean_text)
    axis.set_ylabel("평가 지표 값", **korean_text)
    axis.set_xlim(0, 1)
    metric_maximum = float(grid[["precision", "recall", "f1_score"]].max().max())
    axis.set_ylim(0, min(1.02, max(0.1, metric_maximum * 1.15)))
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", prop=korean_font)
    figure.text(
        0.5,
        0.01,
        "곡선은 실행 옵션의 간격 그리드이고, 빨간 점선은 모든 고유 OOF 확률에서 찾은 정확한 최적값입니다.",
        ha="center",
        fontsize=9,
        **korean_text,
    )
    figure.subplots_adjust(bottom=0.13, left=0.10, right=0.97, top=0.90)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_model_roc_curve(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    result: pd.Series,
    output_path: Path,
) -> None:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    operating_fpr = 1 - float(result["specificity"])
    operating_tpr = float(result["recall"])
    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.plot(fpr, tpr, color="tab:blue", linewidth=2.5, label="OOF ROC 곡선")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="무작위 분류기")
    axis.scatter(
        operating_fpr,
        operating_tpr,
        color="tab:red",
        s=75,
        zorder=3,
        label=f"F1 최적 임계값 ({result['threshold']:.6f})",
    )
    axis.annotate(
        f"TPR={operating_tpr:.4f}\nFPR={operating_fpr:.4f}",
        (operating_fpr, operating_tpr),
        xytext=(0.20, 0.15),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    axis.text(
        0.97,
        0.96,
        f"ROC-AUC = {result['roc_auc']:.4f}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=11,
    )
    axis.set_title(f"{model_name} OOF ROC Curve", **korean_text)
    axis.set_xlabel("위양성률 (1 - 특이도)", **korean_text)
    axis.set_ylabel("진양성률 (재현율)", **korean_text)
    axis.set_xlim(0, 1.01)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", prop=korean_font)
    figure.text(
        0.5,
        0.01,
        "ROC Curve와 ROC-AUC는 임계값과 무관하며, 빨간 점은 F1 최적 임계값의 운영 지점입니다.",
        ha="center",
        fontsize=9,
        **korean_text,
    )
    figure.subplots_adjust(bottom=0.13, left=0.11, right=0.97, top=0.90)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_per_model_outputs(
    selected: pd.DataFrame,
    grid_table: pd.DataFrame,
    y_true: np.ndarray,
    probability_by_model: dict[str, np.ndarray],
    sources: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    source_by_model = {source["model"]: source for source in sources}
    files: dict[str, dict[str, str]] = {}
    for _, result in selected.iterrows():
        model_name = str(result["model"])
        slug = model_slug(model_name)
        model_dir = output_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        model_grid = grid_table.loc[grid_table["model"] == model_name].copy()

        metrics_csv = model_dir / f"{slug}_optimal_f1_metrics.csv"
        grid_csv = model_dir / f"{slug}_threshold_metrics.csv"
        json_path = model_dir / f"{slug}_optimal_f1_summary.json"
        table_png = model_dir / f"{slug}_metrics_table.png"
        threshold_png = model_dir / f"{slug}_threshold_curve.png"
        roc_png = model_dir / f"{slug}_roc_curve.png"

        pd.DataFrame([result.to_dict()]).to_csv(
            metrics_csv, index=False, encoding="utf-8-sig"
        )
        model_grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
        save_model_metrics_table(model_name, result, table_png)
        save_model_threshold_curve(model_name, result, model_grid, threshold_png)
        save_model_roc_curve(
            model_name,
            y_true,
            probability_by_model[model_name],
            result,
            roc_png,
        )

        model_summary = {
            "documentation": {
                "purpose_ko": f"{model_name}의 정확한 OOF F1 최적 임계값과 평가 지표입니다.",
                "prediction_rule": (
                    f"pred = (positive_probability >= {result['threshold']:.9f}).astype(int)"
                ),
                "search_ko": "모든 고유 OOF 양성 확률을 후보 임계값으로 검색했습니다.",
                "auc_note_ko": "ROC-AUC와 PR-AUC는 임계값 적용 전 원래 OOF 확률로 계산합니다.",
            },
            "source": source_by_model[model_name],
            "selected": _json_value(result.to_dict()),
            "files": {
                "metrics_csv": str(metrics_csv.resolve()),
                "threshold_metrics_csv": str(grid_csv.resolve()),
                "metrics_table_png": str(table_png.resolve()),
                "threshold_curve_png": str(threshold_png.resolve()),
                "roc_curve_png": str(roc_png.resolve()),
                "summary_json": str(json_path.resolve()),
            },
        }
        model_summary = _json_value(model_summary)
        json_path.write_text(
            json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        files[model_name] = model_summary["files"]
    return files


def save_plot(
    selected: pd.DataFrame,
    grid_table: pd.DataFrame,
    output_path: Path,
) -> None:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    models = selected["model"].tolist()
    color_names = ["tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:red"]
    color_map = {model: color_names[index % len(color_names)] for index, model in enumerate(models)}

    figure, (curve_axis, metric_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 7.8),
        gridspec_kw={"width_ratios": (1.35, 1)},
    )
    annotation_offsets = {
        "CatBoost": (6, 12),
        "XGBoost weighted": (6, 9),
        "ElasticNet": (6, 9),
    }
    for model in models:
        curve = grid_table.loc[grid_table["model"] == model]
        result = selected.loc[selected["model"] == model].iloc[0]
        curve_axis.plot(
            curve["threshold"],
            curve["f1_score"],
            color=color_map[model],
            linewidth=2,
            label=model,
        )
        curve_axis.scatter(
            result["threshold"],
            result["f1_score"],
            color=color_map[model],
            s=65,
            zorder=3,
        )
        curve_axis.annotate(
            f"t={result['threshold']:.6f}",
            (result["threshold"], result["f1_score"]),
            xytext=annotation_offsets.get(model, (6, 9)),
            textcoords="offset points",
            fontsize=8.5,
        )
    curve_axis.set_title("모델별 임계값에 따른 F1-score", **korean_text)
    curve_axis.set_xlabel("분류 임계값", **korean_text)
    curve_axis.set_ylabel("F1-score", **korean_text)
    curve_axis.set_xlim(0, 1)
    maximum = float(selected["f1_score"].max())
    curve_axis.set_ylim(0, min(1, maximum * 1.28))
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(prop=korean_font, loc="upper right")

    x_positions = np.arange(len(models))
    width = 0.24
    for offset, column, label, color in (
        (-width, "precision", "Precision", "tab:blue"),
        (0, "recall", "Recall", "tab:orange"),
        (width, "f1_score", "F1-score", "tab:green"),
    ):
        values = selected.set_index("model").loc[models, column].to_numpy(dtype=float)
        bars = metric_axis.bar(x_positions + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            metric_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + maximum * 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    metric_maximum = float(
        selected[["precision", "recall", "f1_score"]].max().max()
    )
    for index, model in enumerate(models):
        threshold = selected.loc[selected["model"] == model, "threshold"].iloc[0]
        model_maximum = float(
            selected.loc[
                selected["model"] == model,
                ["precision", "recall", "f1_score"],
            ].max(axis=1).iloc[0]
        )
        metric_axis.text(
            index,
            model_maximum + metric_maximum * 0.07,
            f"임계값 {threshold:.6f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            **korean_text,
        )
    metric_axis.set_title("각 모델의 정확한 F1 최적 운영 지점", **korean_text)
    metric_axis.set_ylabel("평가 지표 값", **korean_text)
    metric_axis.set_xticks(x_positions)
    metric_axis.set_xticklabels(models, rotation=18, ha="right", fontproperties=korean_font)
    metric_axis.set_ylim(0, metric_maximum * 1.27)
    metric_axis.grid(axis="y", alpha=0.25)
    metric_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
    )

    figure.suptitle(
        "OOF 기반 모델별 F1 최대 성능 비교",
        fontsize=15,
        y=0.98,
        **korean_text,
    )
    figure.text(
        0.5,
        0.012,
        (
            "점과 막대의 임계값은 모든 고유 OOF 확률을 검색한 정확한 최적값입니다. "
            "왼쪽 곡선은 가독성을 위한 실행 옵션의 간격 그리드입니다."
        ),
        ha="center",
        fontsize=9,
        **korean_text,
    )
    figure.subplots_adjust(top=0.89, bottom=0.25, left=0.07, right=0.98, wspace=0.22)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = args.model if args.model else default_model_specs()
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step은 0보다 커야 합니다.")
    y_true, probabilities, sources = load_aligned_oof(specs, args.target_column)
    grid_thresholds = generate_thresholds(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    selected, grid_table = evaluate_models(y_true, probabilities, grid_thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "model_optimal_f1_metrics.csv"
    grid_path = args.output_dir / "model_threshold_grid_metrics.csv"
    json_path = args.output_dir / "model_optimal_f1_summary.json"
    plot_path = args.output_dir / "model_optimal_f1_comparison.png"
    selected.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    grid_table.to_csv(grid_path, index=False, encoding="utf-8-sig")
    save_plot(selected, grid_table, plot_path)
    per_model_files = save_per_model_outputs(
        selected,
        grid_table,
        y_true,
        probabilities,
        sources,
        args.output_dir,
    )

    winner = selected.iloc[0]
    summary = {
        "documentation": {
            "purpose_ko": "각 모델이 OOF 검증 데이터에서 낼 수 있는 F1-score 최대 성능을 비교합니다.",
            "prediction_rule": "pred = (positive_probability >= threshold).astype(int)",
            "exact_search_ko": (
                "각 모델의 모든 고유 OOF 양성 확률을 후보 임계값으로 검색했습니다. "
                "따라서 PNG의 간격 그리드보다 정확한 최적 임계값입니다."
            ),
            "tie_rule_ko": (
                "F1-score 동점이면 0.5에 가까운 임계값을 우선하고, 그래도 같으면 "
                "더 높은 임계값을 선택합니다."
            ),
            "notes_ko": [
                "임계값은 OOF 검증 예측으로 선택했으며 테스트 데이터로 선택하지 않았습니다.",
                "ROC-AUC와 PR-AUC는 이진 예측이 아니라 원래 OOF 확률로 계산합니다.",
                "모델별 최적 임계값 비교는 각 모델의 최대 운영 성능을 비교하지만 양성 예측량은 서로 다를 수 있습니다.",
                "XGBoost weighted는 scale_pos_weight를 사용한 클래스 가중 모델이며 Platt Scaling 모델이 아닙니다.",
            ],
        },
        "evaluation": {
            "rows": int(y_true.size),
            "positive_count": int(y_true.sum()),
            "positive_ratio": float(y_true.mean()),
            "target_column": args.target_column,
            "key_columns": KEY_COLUMNS,
            "source_models": sources,
            "plot_grid": {
                "minimum": args.threshold_min,
                "maximum": args.threshold_max,
                "step": args.threshold_step,
                "includes_0_5": True,
            },
        },
        "recommended": {
            "criterion": "maximum OOF F1-score",
            "model": winner["model"],
            "threshold": float(winner["threshold"]),
            "f1_score": float(winner["f1_score"]),
        },
        "ranking": [
            _json_value(row.to_dict()) for _, row in selected.iterrows()
        ],
        "files": {
            "optimal_metrics_csv": str(comparison_path.resolve()),
            "threshold_grid_csv": str(grid_path.resolve()),
            "summary_json": str(json_path.resolve()),
            "comparison_png": str(plot_path.resolve()),
            "reproducible_python_script": str(Path(__file__).resolve()),
            "per_model": per_model_files,
        },
    }
    summary = _json_value(summary)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        help=(
            "반복 지정: '이름|OOF_CSV_경로|양성확률_컬럼'. "
            "생략하면 CatBoost, 가중 XGBoost, ElasticNet을 비교합니다."
        ),
    )
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-min", type=float, default=0.005)
    parser.add_argument("--threshold-max", type=float, default=0.995)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        selected, summary = run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print("\n=== 모델별 정확한 F1 최적 임계값 ===")
    print(selected.to_string(index=False))
    recommended = summary["recommended"]
    print(
        "\n추천: "
        f"{recommended['model']}, threshold={recommended['threshold']:.9f}, "
        f"F1={recommended['f1_score']:.6f}"
    )
    print(f"결과 저장 완료: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
