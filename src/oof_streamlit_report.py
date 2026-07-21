"""모델별 OOF 평가표와 ROC Figure를 만드는 공통 Streamlit 백엔드입니다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .compare_model_optimal_thresholds import exact_f1_optimal_threshold
from .data import require_columns
from .evaluate import compare_thresholds


matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


@dataclass(frozen=True)
class OOFModelConfig:
    model_name: str
    display_name: str
    file_prefix: str
    oof_path: Path
    output_dir: Path
    target_column: str
    probability_column: str
    output_interpretation: str


@dataclass(frozen=True)
class OOFModelReport:
    config: OOFModelConfig
    y_true: np.ndarray
    probabilities: np.ndarray
    metrics: dict[str, Any]
    metrics_frame: pd.DataFrame
    display_table: pd.DataFrame
    source: dict[str, Any]
    search_mode: str
    selection_method: str


def _korean_font() -> font_manager.FontProperties | None:
    for family in ("Malgun Gothic", "NanumGothic"):
        try:
            return font_manager.FontProperties(
                fname=font_manager.findfont(family, fallback_to_default=False)
            )
        except ValueError:
            continue
    return None


def load_oof(
    config: OOFModelConfig,
    oof_path: Path | None = None,
    target_column: str | None = None,
    probability_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = oof_path or config.oof_path
    target = target_column or config.target_column
    probability = probability_column or config.probability_column
    if not path.is_file():
        raise FileNotFoundError(f"{config.display_name} OOF 파일이 없습니다: {path}")

    columns = [target, probability]
    frame = pd.read_csv(path, usecols=columns)
    require_columns(frame, columns, path.name)
    labels = pd.to_numeric(frame[target], errors="raise").to_numpy()
    probabilities = pd.to_numeric(frame[probability], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError(f"{target}에는 음성(0)과 양성(1)이 모두 있어야 합니다.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{probability}에 NaN 또는 무한대가 있습니다.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{probability}은 0과 1 사이의 확률이어야 합니다.")

    labels = labels.astype(np.int8)
    source = {
        "model": config.model_name,
        "oof_path": str(path.resolve()),
        "target_column": target,
        "probability_column": probability,
        "rows": int(labels.size),
        "positive_label": 1,
        "positive_count": int(labels.sum()),
        "positive_ratio": float(labels.mean()),
        "output_interpretation": config.output_interpretation,
    }
    return labels, probabilities, source


def _display_table(
    metrics: dict[str, Any], threshold_is_f1_optimal: bool
) -> pd.DataFrame:
    threshold_label = "F1 최적 임계값" if threshold_is_f1_optimal else "분류 임계값"
    rows = [
        (threshold_label, f"{metrics['threshold']:.9f}"),
        ("Accuracy", f"{metrics['accuracy']:.4%}"),
        ("Precision", f"{metrics['precision']:.4%}"),
        ("Recall", f"{metrics['recall']:.4%}"),
        ("Specificity", f"{metrics['specificity']:.4%}"),
        ("F1-score", f"{metrics['f1_score']:.4%}"),
        ("ROC-AUC", f"{metrics['roc_auc']:.6f}"),
        ("PR-AUC", f"{metrics['pr_auc']:.6f}"),
        ("TP", f"{int(metrics['TP']):,}"),
        ("FP", f"{int(metrics['FP']):,}"),
        ("TN", f"{int(metrics['TN']):,}"),
        ("FN", f"{int(metrics['FN']):,}"),
        ("양성 예측 개수", f"{int(metrics['predicted_positive_count']):,}"),
        ("양성 예측 비율", f"{metrics['predicted_positive_ratio']:.4%}"),
    ]
    return pd.DataFrame(rows, columns=["평가 지표", "값"])


def evaluate_oof_model(
    config: OOFModelConfig,
    oof_path: Path | None = None,
    threshold: float | None = None,
    target_column: str | None = None,
    probability_column: str | None = None,
) -> OOFModelReport:
    labels, probabilities, source = load_oof(
        config,
        oof_path,
        target_column,
        probability_column,
    )
    if threshold is None:
        selected_threshold, _, candidate_count = exact_f1_optimal_threshold(
            labels, probabilities
        )
        search_mode = "all_unique_oof_probabilities"
        selection_method = "모든 고유 OOF 확률에서 F1-score 최대값 선택"
    else:
        if not 0 <= threshold <= 1:
            raise ValueError("임계값은 0과 1 사이여야 합니다.")
        selected_threshold = float(threshold)
        candidate_count = None
        search_mode = "user_supplied_threshold"
        selection_method = "사용자가 지정한 임계값"

    metrics = compare_thresholds(
        labels,
        probabilities,
        np.asarray([selected_threshold], dtype=float),
    ).iloc[0].to_dict()
    metrics_frame = pd.DataFrame(
        [
            {
                "model": config.model_name,
                "search_mode": search_mode,
                "selection_method": selection_method,
                "unique_probability_candidates": candidate_count,
                **metrics,
            }
        ]
    )
    return OOFModelReport(
        config=config,
        y_true=labels,
        probabilities=probabilities,
        metrics=metrics,
        metrics_frame=metrics_frame,
        display_table=_display_table(metrics, threshold is None),
        source=source,
        search_mode=search_mode,
        selection_method=selection_method,
    )


def create_metrics_table_figure(report: OOFModelReport) -> Figure:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    figure, axis = plt.subplots(figsize=(8, 8.2))
    axis.axis("off")
    axis.set_title(
        f"{report.config.display_name} F1 최적 임계값 평가표",
        pad=16,
        **korean_text,
    )
    table = axis.table(
        cellText=report.display_table.values,
        colLabels=report.display_table.columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0.08, 0.10, 0.84, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.25)
    if korean_font is not None:
        for cell in table.get_celld().values():
            cell.get_text().set_fontproperties(korean_font)
    axis.text(
        0.08,
        0.035,
        "ROC-AUC와 PR-AUC는 임계값 적용 전의 원래 OOF 확률로 계산합니다.",
        transform=axis.transAxes,
        fontsize=9,
        **korean_text,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.91, bottom=0.04)
    return figure


def create_roc_figure(report: OOFModelReport) -> Figure:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    fpr, tpr, _ = roc_curve(report.y_true, report.probabilities)
    operating_fpr = 1 - float(report.metrics["specificity"])
    operating_tpr = float(report.metrics["recall"])
    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.plot(
        fpr,
        tpr,
        color="tab:blue",
        linewidth=2.5,
        label=f"{report.config.display_name} OOF ROC",
    )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="무작위 분류기")
    axis.scatter(
        operating_fpr,
        operating_tpr,
        color="tab:red",
        marker="o",
        s=75,
        zorder=3,
        label=f"선택 임계값 ({report.metrics['threshold']:.6f})",
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
        f"ROC-AUC = {report.metrics['roc_auc']:.4f}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=11,
    )
    axis.set_title(f"{report.config.display_name} OOF ROC 곡선", **korean_text)
    axis.set_xlabel("위양성률 (1 - 특이도)", **korean_text)
    axis.set_ylabel("진양성률 (재현율)", **korean_text)
    axis.set_xlim(0, 1.01)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", prop=korean_font)
    figure.subplots_adjust(bottom=0.13, left=0.11, right=0.97, top=0.90)
    return figure


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


def save_oof_report(
    report: OOFModelReport,
    output_dir: Path,
    script_path: Path,
    evaluator_name: str,
    roc_function_name: str = "create_roc_figure",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.config.file_prefix
    metrics_csv = output_dir / f"{prefix}_optimal_f1_metrics.csv"
    summary_json = output_dir / f"{prefix}_optimal_f1_summary.json"
    metrics_png = output_dir / f"{prefix}_metrics_table.png"
    roc_png = output_dir / f"{prefix}_roc_curve.png"

    report.metrics_frame.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    metrics_figure = create_metrics_table_figure(report)
    metrics_figure.savefig(metrics_png, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(metrics_figure)
    roc_figure = create_roc_figure(report)
    roc_figure.savefig(roc_png, dpi=180)
    plt.close(roc_figure)

    files = {
        "metrics_csv": str(metrics_csv.resolve()),
        "metrics_table_png": str(metrics_png.resolve()),
        "roc_curve_png": str(roc_png.resolve()),
        "summary_json": str(summary_json.resolve()),
        "reproducible_python_script": str(script_path.resolve()),
    }
    optional_files = {
        "threshold_metrics_csv": output_dir / f"{prefix}_threshold_metrics.csv",
        "threshold_curve_png": output_dir / f"{prefix}_threshold_curve.png",
    }
    files.update(
        {
            name: str(path.resolve())
            for name, path in optional_files.items()
            if path.is_file()
        }
    )
    payload = {
        "documentation": {
            "purpose_ko": (
                f"Streamlit에서 표시할 {report.config.display_name} OOF 평가표와 ROC 곡선입니다."
            ),
            "streamlit_table": f"st.dataframe({evaluator_name}().display_table, hide_index=True)",
            "streamlit_roc": (
                f"report = {evaluator_name}(); st.pyplot({roc_function_name}(report))"
            ),
            "prediction_rule": "pred = (positive_probability >= threshold).astype(int)",
            "auc_note_ko": "ROC-AUC와 PR-AUC는 임계값 적용 전 원래 OOF 확률로 계산합니다.",
        },
        "source": report.source,
        "search_mode": report.search_mode,
        "selection_method": report.selection_method,
        "selected": _json_value(report.metrics),
        "files": files,
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return files
