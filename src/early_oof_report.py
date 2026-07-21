"""주차 범위를 제한한 OOF 임계값 평가와 시각화 공통 기능입니다."""

from __future__ import annotations

import argparse
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
from .evaluate import compare_thresholds, generate_thresholds
from .oof_streamlit_report import OOFModelConfig


matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


DEFAULT_START_WEEK = 1
DEFAULT_END_WEEK = 10
DEFAULT_WEEK_COLUMN = "prediction_week"


@dataclass(frozen=True)
class EarlyOOFReport:
    """Streamlit 표와 그래프에 필요한 주차 부분집단 OOF 평가 결과입니다."""

    config: OOFModelConfig
    y_true: np.ndarray
    probabilities: np.ndarray
    metrics: dict[str, Any]
    metrics_frame: pd.DataFrame
    threshold_frame: pd.DataFrame
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


def _validate_week_range(start_week: int, end_week: int) -> None:
    if start_week < 1:
        raise ValueError("시작 주차는 1 이상이어야 합니다.")
    if end_week < start_week:
        raise ValueError("종료 주차는 시작 주차 이상이어야 합니다.")


def load_early_oof(
    config: OOFModelConfig,
    oof_path: Path | None = None,
    *,
    start_week: int = DEFAULT_START_WEEK,
    end_week: int = DEFAULT_END_WEEK,
    week_column: str = DEFAULT_WEEK_COLUMN,
    target_column: str | None = None,
    probability_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """OOF에서 지정 주차만 골라 정답과 양성 확률을 검증해 불러옵니다."""
    _validate_week_range(start_week, end_week)
    path = oof_path or config.oof_path
    target = target_column or config.target_column
    probability = probability_column or config.probability_column
    if not path.is_file():
        raise FileNotFoundError(f"{config.display_name} OOF 파일이 없습니다: {path}")

    columns = [week_column, target, probability]
    frame = pd.read_csv(path, usecols=columns)
    require_columns(frame, columns, path.name)
    source_rows = int(len(frame))
    weeks = pd.to_numeric(frame[week_column], errors="raise")
    subset = frame.loc[weeks.between(start_week, end_week)].copy()
    if subset.empty:
        raise ValueError(f"{start_week}~{end_week}주차 OOF 데이터가 없습니다.")

    labels = pd.to_numeric(subset[target], errors="raise").to_numpy()
    probabilities = pd.to_numeric(
        subset[probability], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError(
            f"{start_week}~{end_week}주차의 {target}에는 음성(0)과 양성(1)이 모두 있어야 합니다."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{probability}에 NaN 또는 무한대가 있습니다.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{probability}은 0과 1 사이의 확률이어야 합니다.")

    labels = labels.astype(np.int8)
    source = {
        "model": config.model_name,
        "oof_path": str(path.resolve()),
        "week_column": week_column,
        "start_week": int(start_week),
        "end_week": int(end_week),
        "source_rows": source_rows,
        "rows": int(labels.size),
        "target_column": target,
        "probability_column": probability,
        "positive_label": 1,
        "positive_count": int(labels.sum()),
        "positive_ratio": float(labels.mean()),
        "output_interpretation": config.output_interpretation,
    }
    return labels, probabilities, source


def _display_table(metrics: dict[str, Any], source: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("평가 주차", f"{source['start_week']}~{source['end_week']}주차"),
        ("평가 행", f"{source['rows']:,}"),
        ("실제 이탈", f"{source['positive_count']:,} ({source['positive_ratio']:.4%})"),
        ("F1 최적 임계값", f"{metrics['threshold']:.9f}"),
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


def evaluate_early_oof_model(
    config: OOFModelConfig,
    oof_path: Path | None = None,
    threshold: float | None = None,
    *,
    start_week: int = DEFAULT_START_WEEK,
    end_week: int = DEFAULT_END_WEEK,
    week_column: str = DEFAULT_WEEK_COLUMN,
    target_column: str | None = None,
    probability_column: str | None = None,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.05,
) -> EarlyOOFReport:
    labels, probabilities, source = load_early_oof(
        config,
        oof_path,
        start_week=start_week,
        end_week=end_week,
        week_column=week_column,
        target_column=target_column,
        probability_column=probability_column,
    )
    if threshold is None:
        selected_threshold, _, candidate_count = exact_f1_optimal_threshold(
            labels, probabilities
        )
        search_mode = "all_unique_early_oof_probabilities"
        selection_method = (
            f"{start_week}~{end_week}주차의 모든 고유 OOF 확률에서 F1-score 최대값 선택; "
            "동점이면 0.5에 가까운 값, 이후 높은 임계값 우선"
        )
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
    grid_thresholds = generate_thresholds(
        threshold_min,
        threshold_max,
        threshold_step,
    )
    grid_thresholds = np.unique(
        np.r_[grid_thresholds, selected_threshold]
    )
    threshold_frame = compare_thresholds(labels, probabilities, grid_thresholds)
    metrics_frame = pd.DataFrame(
        [
            {
                "model": config.model_name,
                "start_week": start_week,
                "end_week": end_week,
                "rows": source["rows"],
                "positive_count": source["positive_count"],
                "positive_ratio": source["positive_ratio"],
                "search_mode": search_mode,
                "selection_method": selection_method,
                "unique_probability_candidates": candidate_count,
                **metrics,
            }
        ]
    )
    return EarlyOOFReport(
        config=config,
        y_true=labels,
        probabilities=probabilities,
        metrics=metrics,
        metrics_frame=metrics_frame,
        threshold_frame=threshold_frame,
        display_table=_display_table(metrics, source),
        source=source,
        search_mode=search_mode,
        selection_method=selection_method,
    )


def create_metrics_table_figure(report: EarlyOOFReport) -> Figure:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    start_week = report.source["start_week"]
    end_week = report.source["end_week"]
    figure, axis = plt.subplots(figsize=(8, 9.2))
    axis.axis("off")
    axis.set_title(
        f"{report.config.display_name} {start_week}~{end_week}주차 OOF 평가표",
        pad=16,
        **korean_text,
    )
    table = axis.table(
        cellText=report.display_table.values,
        colLabels=report.display_table.columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0.08, 0.09, 0.84, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.3)
    table.scale(1, 1.2)
    if korean_font is not None:
        for cell in table.get_celld().values():
            cell.get_text().set_fontproperties(korean_font)
    axis.text(
        0.08,
        0.025,
        f"ROC-AUC와 PR-AUC는 임계값 적용 전의 {start_week}~{end_week}주차 OOF 확률로 계산합니다.",
        transform=axis.transAxes,
        fontsize=9,
        **korean_text,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03)
    return figure


def create_threshold_curve_figure(report: EarlyOOFReport) -> Figure:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    start_week = report.source["start_week"]
    end_week = report.source["end_week"]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for column, label, color in (
        ("precision", "Precision", "tab:blue"),
        ("recall", "Recall", "tab:orange"),
        ("f1_score", "F1-score", "tab:green"),
    ):
        axis.plot(
            report.threshold_frame["threshold"],
            report.threshold_frame[column],
            linewidth=2,
            color=color,
            label=label,
        )
    threshold = float(report.metrics["threshold"])
    axis.axvline(
        threshold,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"F1 최적 임계값 ({threshold:.6f})",
    )
    axis.scatter(
        threshold,
        report.metrics["f1_score"],
        color="tab:red",
        s=75,
        zorder=3,
    )
    axis.set_title(
        f"{report.config.display_name} {start_week}~{end_week}주차 임계값별 지표",
        **korean_text,
    )
    axis.set_xlabel("분류 임계값", **korean_text)
    axis.set_ylabel("평가 지표 값", **korean_text)
    axis.set_xlim(0, 1)
    maximum = float(
        report.threshold_frame[["precision", "recall", "f1_score"]].max().max()
    )
    axis.set_ylim(0, min(1.02, max(0.1, maximum * 1.15)))
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", prop=korean_font)
    figure.text(
        0.5,
        0.01,
        "곡선은 실행 옵션의 후보 그리드이며, 빨간 점선은 모든 고유 OOF 확률에서 찾은 F1 최적값입니다.",
        ha="center",
        fontsize=9,
        **korean_text,
    )
    figure.subplots_adjust(bottom=0.13, left=0.10, right=0.97, top=0.90)
    return figure


def create_roc_figure(report: EarlyOOFReport) -> Figure:
    korean_font = _korean_font()
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}
    start_week = report.source["start_week"]
    end_week = report.source["end_week"]
    fpr, tpr, _ = roc_curve(report.y_true, report.probabilities)
    operating_fpr = 1 - float(report.metrics["specificity"])
    operating_tpr = float(report.metrics["recall"])
    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.plot(
        fpr,
        tpr,
        color="tab:blue",
        linewidth=2.5,
        label=f"{start_week}~{end_week}주차 OOF ROC",
    )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="무작위 분류기")
    axis.scatter(
        operating_fpr,
        operating_tpr,
        color="tab:red",
        s=75,
        zorder=3,
        label=f"F1 최적 임계값 ({report.metrics['threshold']:.6f})",
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
    axis.set_title(
        f"{report.config.display_name} {start_week}~{end_week}주차 OOF ROC Curve",
        **korean_text,
    )
    axis.set_xlabel("위양성률 (1 - 특이도)", **korean_text)
    axis.set_ylabel("진양성률 (재현율)", **korean_text)
    axis.set_xlim(0, 1.01)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", prop=korean_font)
    figure.text(
        0.5,
        0.01,
        "ROC Curve와 ROC-AUC는 분류 임계값과 무관하며, 빨간 점은 선택 임계값의 운영 지점입니다.",
        ha="center",
        fontsize=9,
        **korean_text,
    )
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


def save_early_report(
    report: EarlyOOFReport,
    output_dir: Path,
    script_path: Path,
    evaluator_name: str,
) -> dict[str, str]:
    """CSV·JSON과 평가표·임계값·ROC PNG를 모델별 폴더에 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.config.file_prefix
    metrics_csv = output_dir / f"{prefix}_optimal_f1_metrics.csv"
    threshold_csv = output_dir / f"{prefix}_threshold_metrics.csv"
    summary_json = output_dir / f"{prefix}_optimal_f1_summary.json"
    metrics_png = output_dir / f"{prefix}_metrics_table.png"
    threshold_png = output_dir / f"{prefix}_threshold_curve.png"
    roc_png = output_dir / f"{prefix}_roc_curve.png"

    report.metrics_frame.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    report.threshold_frame.to_csv(threshold_csv, index=False, encoding="utf-8-sig")
    figures = (
        (create_metrics_table_figure(report), metrics_png),
        (create_threshold_curve_figure(report), threshold_png),
        (create_roc_figure(report), roc_png),
    )
    for figure, path in figures:
        figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)

    files = {
        "metrics_csv": str(metrics_csv.resolve()),
        "threshold_metrics_csv": str(threshold_csv.resolve()),
        "summary_json": str(summary_json.resolve()),
        "metrics_table_png": str(metrics_png.resolve()),
        "threshold_curve_png": str(threshold_png.resolve()),
        "roc_curve_png": str(roc_png.resolve()),
        "reproducible_python_script": str(script_path.resolve()),
    }
    payload = {
        "documentation": {
            "purpose_ko": (
                f"{report.source['start_week']}~{report.source['end_week']}주차 OOF 부분집단의 "
                f"{report.config.display_name} 임계값 평가 결과입니다."
            ),
            "scope_note_ko": "신규 학생 추론이 아니라 기존 OOF 검증행의 주차 부분집단 분석입니다.",
            "streamlit_table": (
                f"st.dataframe({evaluator_name}().display_table, hide_index=True)"
            ),
            "streamlit_roc": (
                f"report = {evaluator_name}(); st.pyplot(create_roc_figure(report))"
            ),
            "prediction_rule": "pred = (positive_probability >= threshold).astype(int)",
            "auc_note_ko": "ROC-AUC와 PR-AUC는 임계값 적용 전의 주차 부분집단 OOF 확률로 계산합니다.",
        },
        "source": report.source,
        "search_mode": report.search_mode,
        "selection_method": report.selection_method,
        "selected": _json_value(report.metrics),
        "threshold_grid": {
            "candidate_count": int(len(report.threshold_frame)),
            "includes_default_0_5": bool(
                np.isclose(report.threshold_frame["threshold"], 0.5).any()
            ),
            "includes_exact_selected_threshold": bool(
                np.isclose(
                    report.threshold_frame["threshold"], report.metrics["threshold"]
                ).any()
            ),
        },
        "files": files,
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return files


def run_early_cli(
    config: OOFModelConfig,
    script_path: Path,
    evaluator_name: str,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-path", type=Path, default=config.oof_path)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--start-week", type=int, default=DEFAULT_START_WEEK)
    parser.add_argument("--end-week", type=int, default=DEFAULT_END_WEEK)
    parser.add_argument("--week-column", default=DEFAULT_WEEK_COLUMN)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--target-column", default=config.target_column)
    parser.add_argument("--probability-column", default=config.probability_column)
    args = parser.parse_args()
    try:
        report = evaluate_early_oof_model(
            config,
            args.oof_path,
            args.threshold,
            start_week=args.start_week,
            end_week=args.end_week,
            week_column=args.week_column,
            target_column=args.target_column,
            probability_column=args.probability_column,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        files = save_early_report(
            report,
            args.output_dir,
            script_path,
            evaluator_name,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"\n=== {config.display_name} {args.start_week}~{args.end_week}주차 OOF 평가 ==="
    )
    print(report.display_table.to_string(index=False))
    print(f"\n결과 저장 완료: {args.output_dir.resolve()}")
    for name, path in files.items():
        print(f"- {name}: {path}")
