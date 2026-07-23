"""1~10주차 전용 모델이 공유하는 데이터 필터링·OOF·임계값 저장 기능."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


# 이 파일을 직접 실행하는 모델 스크립트에서도 src 패키지를 찾을 수 있게 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .common_weekly_metrics import (
        ID_COL,
        N_SPLITS,
        RANDOM_STATE,
        SORT_COLUMNS,
        TARGET_COL,
        PreparedWeeklyData,
        calculate_metrics,
        load_and_prepare_weekly_data,
        resolve_data_path,
        validate_oof,
    )
except ImportError:  # ``python models/early_*.py`` 직접 실행 지원
    from common_weekly_metrics import (  # type: ignore
        ID_COL,
        N_SPLITS,
        RANDOM_STATE,
        SORT_COLUMNS,
        TARGET_COL,
        PreparedWeeklyData,
        calculate_metrics,
        load_and_prepare_weekly_data,
        resolve_data_path,
        validate_oof,
    )

from src.compare_model_optimal_thresholds import exact_f1_optimal_threshold
from src.evaluate import compare_thresholds, generate_thresholds


DEFAULT_START_WEEK = 1
DEFAULT_END_WEEK = 10
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ML"
DEFAULT_EVAL_PARAMS_DIR = PROJECT_ROOT / "data" / "eval_params"
DEFAULT_EVAL_RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"
# 사용자가 지정한 기존 폴더명 철자를 그대로 유지한다.
DEFAULT_AUC_GRAPHS_DIR = PROJECT_ROOT / "data" / "auc_grahps"


@dataclass(frozen=True)
class EarlyModelConfig:
    """모델별 출력 이름과 실제 학습 조건을 명시한다."""

    model_name: str
    file_prefix: str
    probability_column: str
    hyperparameters: dict[str, Any]
    probability_interpretation: str


def subset_prepared_data(
    prepared: PreparedWeeklyData,
    start_week: int = DEFAULT_START_WEEK,
    end_week: int = DEFAULT_END_WEEK,
) -> PreparedWeeklyData:
    """검증 완료된 주간 데이터에서 지정 주차만 행·Feature 순서를 맞춰 추출한다."""
    if start_week < 1 or end_week < start_week:
        raise ValueError("주차 범위는 1 <= start_week <= end_week여야 합니다.")

    weeks = pd.to_numeric(prepared.data["prediction_week"], errors="coerce")
    if weeks.isna().any():
        raise ValueError("prediction_week에 숫자로 변환할 수 없는 값이 있습니다.")
    mask = weeks.between(start_week, end_week).to_numpy()
    if not mask.any():
        raise ValueError(f"{start_week}~{end_week}주차 학습 데이터가 없습니다.")

    data = prepared.data.loc[mask].reset_index(drop=True)
    features = prepared.features.loc[mask].reset_index(drop=True)
    target = prepared.target[mask]
    groups = prepared.groups[mask]
    if set(np.unique(target)) != {0, 1}:
        raise ValueError(
            f"{start_week}~{end_week}주차 Target에는 음성(0)과 양성(1)이 모두 있어야 합니다."
        )
    if np.unique(groups).size < N_SPLITS:
        raise ValueError(f"학생 수가 {N_SPLITS}-Fold 검증에 부족합니다.")

    profile = dict(prepared.profile)
    profile.update(
        {
            "source_rows": int(prepared.profile["rows"]),
            "rows": int(len(data)),
            "target_count": int(target.sum()),
            "target_rate": float(target.mean()),
            "student_count": int(np.unique(groups).size),
            "start_week": int(start_week),
            "end_week": int(end_week),
            "training_scope": "early_weeks_only",
        }
    )
    return PreparedWeeklyData(
        data=data,
        features=features,
        target=target,
        groups=groups,
        categorical=prepared.categorical,
        numeric=prepared.numeric,
        profile=profile,
    )


def load_early_weekly_data(
    data_path: str | Path,
    *,
    start_week: int = DEFAULT_START_WEEK,
    end_week: int = DEFAULT_END_WEEK,
    max_rows: int | None = None,
) -> PreparedWeeklyData:
    """원본 CSV를 검증한 다음 모델 학습 전에 주차 범위를 제한한다."""
    prepared = load_and_prepare_weekly_data(data_path, max_rows=max_rows)
    early = subset_prepared_data(prepared, start_week, end_week)
    early.profile["smoke_test"] = max_rows is not None
    early.profile["smoke_rows_requested"] = max_rows
    return early


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """네 모델이 동일하게 사용하는 실행 옵션을 추가한다."""
    parser.add_argument(
        "--data-path",
        type=Path,
        help="학습 CSV. 생략하면 OULAD_DATA_PATH 또는 models/ML/used_data를 확인합니다.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--eval-params-dir", type=Path, default=DEFAULT_EVAL_PARAMS_DIR
    )
    parser.add_argument(
        "--eval-results-dir", type=Path, default=DEFAULT_EVAL_RESULTS_DIR
    )
    parser.add_argument(
        "--auc-graphs-dir", type=Path, default=DEFAULT_AUC_GRAPHS_DIR
    )
    parser.add_argument("--start-week", type=int, default=DEFAULT_START_WEEK)
    parser.add_argument("--end-week", type=int, default=DEFAULT_END_WEEK)
    parser.add_argument(
        "--threshold",
        type=float,
        help="고정 임계값. 생략하면 Early OOF의 모든 고유 확률에서 F1 최적값을 찾습니다.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--smoke-rows",
        type=int,
        help="개발 확인용 CSV 선두 행 수. 이 옵션을 사용한 결과는 최종 성능으로 사용하지 않습니다.",
    )


def resolved_data_path(cli_path: Path | None) -> Path:
    """공통 탐색 규칙으로 실제 학습 CSV 위치를 결정한다."""
    return resolve_data_path(cli_path)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def build_auc_curve_frame(
    *,
    model_name: str,
    target: np.ndarray,
    probability: np.ndarray,
    selected_threshold: float,
    roc_auc: float,
    pr_auc: float,
    start_week: int,
    end_week: int,
) -> pd.DataFrame:
    """ROC와 Precision-Recall 곡선을 다시 그릴 수 있는 정규화 좌표를 만든다."""
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        target, probability, pos_label=1
    )
    precision, recall, pr_thresholds = precision_recall_curve(
        target, probability, pos_label=1
    )
    # PR curve는 좌표가 threshold보다 하나 많으므로 마지막 좌표에 NaN을 붙인다.
    pr_threshold_values = np.append(pr_thresholds, np.nan)
    roc_threshold_values = np.where(
        np.isfinite(roc_thresholds), roc_thresholds, np.nan
    )

    def curve_rows(
        curve_type: str,
        x: np.ndarray,
        y: np.ndarray,
        thresholds: np.ndarray,
        x_name: str,
        y_name: str,
        score_name: str,
        score_value: float,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": model_name,
                "curve_type": curve_type,
                "point_index": np.arange(len(x), dtype=int),
                "x": x.astype(float),
                "y": y.astype(float),
                "threshold": thresholds.astype(float),
                "x_name": x_name,
                "y_name": y_name,
                "score_name": score_name,
                "score_value": float(score_value),
                "selected_threshold": float(selected_threshold),
                "positive_label": 1,
                "start_week": int(start_week),
                "end_week": int(end_week),
            }
        )

    return pd.concat(
        [
            curve_rows(
                "roc",
                false_positive_rate,
                true_positive_rate,
                roc_threshold_values,
                "false_positive_rate",
                "true_positive_rate",
                "roc_auc",
                roc_auc,
            ),
            curve_rows(
                "precision_recall",
                recall,
                precision,
                pr_threshold_values,
                "recall",
                "precision",
                "pr_auc",
                pr_auc,
            ),
        ],
        ignore_index=True,
    )


def finalize_early_oof(
    *,
    config: EarlyModelConfig,
    prepared: PreparedWeeklyData,
    probabilities: np.ndarray,
    fold_assignment: np.ndarray,
    fold_hash: str,
    fold_rows: list[dict[str, Any]],
    data_path: Path,
    output_dir: Path,
    eval_params_dir: Path = DEFAULT_EVAL_PARAMS_DIR,
    eval_results_dir: Path = DEFAULT_EVAL_RESULTS_DIR,
    auc_graphs_dir: Path = DEFAULT_AUC_GRAPHS_DIR,
    threshold: float | None = None,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.05,
    additional_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """OOF를 검증하고 임계값·평가지표·재현 조건을 한 번에 저장한다."""
    probability = validate_oof(probabilities, fold_assignment, len(prepared.data))
    if threshold is None:
        selected_threshold, _, unique_candidates = exact_f1_optimal_threshold(
            prepared.target, probability
        )
        search_mode = "all_unique_early_oof_probabilities"
        selection_method = (
            "Early OOF의 모든 고유 양성 확률에서 F1-score 최대; "
            "동점이면 0.5에 가까운 값, 이후 높은 임계값 우선"
        )
    else:
        if not 0 <= threshold <= 1:
            raise ValueError("--threshold는 0과 1 사이여야 합니다.")
        selected_threshold = float(threshold)
        unique_candidates = 0
        search_mode = "user_supplied_threshold"
        selection_method = "사용자가 지정한 임계값"

    selected = compare_thresholds(
        prepared.target, probability, np.asarray([selected_threshold], dtype=float)
    ).iloc[0].to_dict()
    grid = generate_thresholds(threshold_min, threshold_max, threshold_step)
    grid = np.asarray(sorted({*grid.tolist(), float(selected_threshold)}), dtype=float)
    threshold_frame = compare_thresholds(prepared.target, probability, grid)
    ranking_metrics = calculate_metrics(prepared.target, probability)

    metrics_row: dict[str, Any] = {
        "model": config.model_name,
        "training_scope": "early_weeks_only",
        "start_week": prepared.profile["start_week"],
        "end_week": prepared.profile["end_week"],
        "rows": prepared.profile["rows"],
        "student_count": prepared.profile["student_count"],
        "target_count": prepared.profile["target_count"],
        "target_rate": prepared.profile["target_rate"],
        "feature_count": prepared.profile["feature_count"],
        "n_splits": N_SPLITS,
        "group_column": ID_COL,
        "random_state": RANDOM_STATE,
        "fold_assignment_hash": fold_hash,
        "search_mode": search_mode,
        "unique_probability_candidates": unique_candidates,
        **selected,
        **ranking_metrics,
    }
    metrics_frame = pd.DataFrame([metrics_row])
    fold_frame = pd.DataFrame(fold_rows)
    oof_frame = prepared.data[[*SORT_COLUMNS, TARGET_COL]].copy()
    oof_frame["fold"] = fold_assignment
    oof_frame[config.probability_column] = probability

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    params_destination = Path(eval_params_dir)
    results_destination = Path(eval_results_dir)
    curves_destination = Path(auc_graphs_dir)
    params_destination.mkdir(parents=True, exist_ok=True)
    results_destination.mkdir(parents=True, exist_ok=True)
    curves_destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {
        "metrics_csv": destination / f"{config.file_prefix}_metrics.csv",
        "fold_metrics_csv": destination / f"{config.file_prefix}_fold_metrics.csv",
        "oof_predictions_csv": destination / f"{config.file_prefix}_oof_predictions.csv",
        "threshold_metrics_csv": destination / f"{config.file_prefix}_threshold_metrics.csv",
        "summary_json": destination / f"{config.file_prefix}_summary.json",
        "eval_params_json": params_destination / f"{config.file_prefix}_params.json",
        "eval_results_json": results_destination / f"{config.file_prefix}_results.json",
        "auc_curves_parquet": curves_destination
        / f"{config.file_prefix}_auc_curves.parquet",
    }
    metrics_frame.to_csv(files["metrics_csv"], index=False, encoding="utf-8-sig")
    fold_frame.to_csv(files["fold_metrics_csv"], index=False, encoding="utf-8-sig")
    oof_frame.to_csv(files["oof_predictions_csv"], index=False, encoding="utf-8-sig")
    threshold_frame.to_csv(
        files["threshold_metrics_csv"], index=False, encoding="utf-8-sig"
    )
    for suffix, frame in (additional_frames or {}).items():
        path = destination / f"{config.file_prefix}_{suffix}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        files[f"{suffix}_csv"] = path

    curve_frame = build_auc_curve_frame(
        model_name=config.model_name,
        target=prepared.target,
        probability=probability,
        selected_threshold=float(selected_threshold),
        roc_auc=float(selected["roc_auc"]),
        pr_auc=float(selected["pr_auc"]),
        start_week=int(prepared.profile["start_week"]),
        end_week=int(prepared.profile["end_week"]),
    )
    curve_frame.to_parquet(files["auc_curves_parquet"], index=False)

    params_payload = {
        "schema_version": 1,
        "model": config.model_name,
        "file_prefix": config.file_prefix,
        "training_scope": "early_weeks_only",
        "training_data": {
            "path": str(data_path.resolve()),
            "start_week": prepared.profile["start_week"],
            "end_week": prepared.profile["end_week"],
            "rows": prepared.profile["rows"],
            "feature_count": prepared.profile["feature_count"],
            "target_column": TARGET_COL,
            "positive_label": 1,
            "smoke_test": prepared.profile.get("smoke_test", False),
            "smoke_rows_requested": prepared.profile.get("smoke_rows_requested"),
        },
        "validation": {
            "method": "GroupKFold",
            "group_column": ID_COL,
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "fold_assignment_hash": fold_hash,
        },
        "hyperparameters": config.hyperparameters,
        "probability": {
            "column": config.probability_column,
            "interpretation_ko": config.probability_interpretation,
            "sigmoid_applied_after_predict_proba": False,
        },
        "threshold_policy": {
            "search_mode": search_mode,
            "selection_method": selection_method,
            "grid_min": threshold_min,
            "grid_max": threshold_max,
            "grid_step": threshold_step,
        },
    }
    files["eval_params_json"].write_text(
        json.dumps(_json_value(params_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    count_columns = ("TP", "FP", "TN", "FN", "predicted_positive_count")
    selected_metrics = {
        key: int(value) if key in count_columns else float(value)
        for key, value in selected.items()
    }
    tn = int(selected_metrics["TN"])
    fp = int(selected_metrics["FP"])
    fn = int(selected_metrics["FN"])
    tp = int(selected_metrics["TP"])
    results_payload = {
        "schema_version": 1,
        "model": config.model_name,
        "training_scope": "early_weeks_only",
        "evaluation_data": {
            "type": "OOF validation",
            "start_week": prepared.profile["start_week"],
            "end_week": prepared.profile["end_week"],
            "rows": prepared.profile["rows"],
            "target_count": prepared.profile["target_count"],
            "target_rate": prepared.profile["target_rate"],
            "smoke_test": prepared.profile.get("smoke_test", False),
        },
        "selected_threshold": float(selected_threshold),
        "prediction_rule": (
            f"pred = (positive_probability >= {selected_threshold:.9f}).astype(int)"
        ),
        "metrics": selected_metrics,
        "confusion_matrix": {
            "labels": [0, 1],
            "layout": "[[TN, FP], [FN, TP]]",
            "matrix": [[tn, fp], [fn, tp]],
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
        "ranking_metrics_note_ko": (
            "ROC-AUC와 PR-AUC는 임계값 예측이 아니라 원래 OOF 양성 확률로 계산했습니다."
        ),
        "auc_curves_parquet": str(files["auc_curves_parquet"].resolve()),
    }
    files["eval_results_json"].write_text(
        json.dumps(_json_value(results_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = {
        "documentation": {
            "purpose_ko": (
                f"{prepared.profile['start_week']}~{prepared.profile['end_week']}주차 행만으로 "
                f"{config.model_name}을 새로 학습한 학생 단위 OOF 결과입니다."
            ),
            "validation_ko": (
                f"id_student 기준 {N_SPLITS}-Fold GroupKFold이며 각 행은 검증에 한 번만 포함됩니다."
            ),
            "threshold_ko": selection_method,
            "prediction_rule": (
                f"pred = (positive_probability >= {selected_threshold:.9f}).astype(int)"
            ),
            "probability_ko": config.probability_interpretation,
            "test_data_note_ko": "임계값 선택에는 별도 테스트 데이터를 사용하지 않았습니다.",
            "smoke_test_note_ko": (
                "smoke_test=true인 결과는 개발 확인용이며 최종 성능으로 사용하면 안 됩니다."
            ),
        },
        "training_data": {
            "path": str(data_path.resolve()),
            **prepared.profile,
            "target_column": TARGET_COL,
            "positive_label": 1,
        },
        "model": {
            "name": config.model_name,
            "hyperparameters": config.hyperparameters,
        },
        "threshold_selection": {
            "search_mode": search_mode,
            "selection_method": selection_method,
            "selected": selected,
            "grid_min": threshold_min,
            "grid_max": threshold_max,
            "grid_step": threshold_step,
            "grid_includes_0_5": bool(np.isclose(grid, 0.5).any()),
        },
        "ranking_metrics": ranking_metrics,
        "files": {name: str(path.resolve()) for name, path in files.items()},
    }
    files["summary_json"].write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "metrics": metrics_frame,
        "fold_metrics": fold_frame,
        "oof": oof_frame,
        "threshold_metrics": threshold_frame,
        "summary": payload,
        "files": {name: str(path.resolve()) for name, path in files.items()},
    }


def print_saved_result(result: dict[str, Any]) -> None:
    """모델 실행 종료 시 핵심 성능과 생성 파일을 표시한다."""
    print("\n=== Early 모델 OOF 결과 ===")
    print(result["metrics"].to_string(index=False))
    print("\n=== 생성 파일 ===")
    for name, path in result["files"].items():
        print(f"- {name}: {path}")
