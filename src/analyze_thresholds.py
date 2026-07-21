"""저장된 모델과 validation split으로 이진 분류 threshold를 비교한다."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import GroupKFold

from .data import PROJECT_ROOT, TARGET_COLUMN, require_columns
from .evaluate import compare_thresholds, generate_thresholds, select_thresholds


matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "feature_schema.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "threshold_analysis"
OOF_KEY_COLUMNS = ["code_module", "code_presentation", "id_student", "prediction_week"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"JSON 파일을 읽을 수 없습니다: {path} ({exc})") from exc


def load_joblib_model(model_path: Path):
    """기존 train_ml.py가 저장한 scikit-learn joblib Pipeline을 불러온다."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"학습 완료 모델이 없습니다: {model_path}. "
            "먼저 `python -m src.train_ml <split 포함 모델링 CSV>`로 생성하거나 "
            "--model-path에 실제 joblib 경로를 지정하세요."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 경로가 파일이 아닙니다: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"joblib 모델을 로드하지 못했습니다: {model_path} ({exc})") from exc
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "현재 프로젝트 모델 계약은 predict_proba를 제공하는 scikit-learn "
            "분류기/Pipeline입니다. logits 전용 모델로 확인할 근거가 없으므로 "
            "임의로 sigmoid를 적용하지 않습니다."
        )
    return model


def resolve_features(model, schema: dict) -> list[str]:
    """학습 시 저장한 schema를 우선하고 모델의 입력 컬럼명을 보조로 사용한다."""
    features = schema.get("features") or []
    if not features and hasattr(model, "feature_names_in_"):
        features = model.feature_names_in_.tolist()
    if not features:
        raise ValueError(
            "모델 입력 feature를 확인할 수 없습니다. feature_schema.json의 features가 "
            "비어 있고 모델에도 feature_names_in_가 없습니다."
        )
    return [str(feature) for feature in features]


def load_validation_data(
    data_path: Path,
    features: list[str],
    target_column: str,
    split_column: str,
    validation_value: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """명시적인 validation 행만 로드하며 test/전체 데이터 사용을 방지한다."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"검증 데이터 파일이 없습니다: {data_path}. "
            "학습에 사용한 split 포함 CSV를 --data-path로 지정하세요."
        )
    if not data_path.is_file():
        raise FileNotFoundError(f"검증 데이터 경로가 파일이 아닙니다: {data_path}")
    frame = pd.read_csv(data_path)
    require_columns(frame, [target_column, split_column, *features], data_path.name)
    validation = frame.loc[frame[split_column] == validation_value].copy()
    if validation.empty:
        available = sorted(frame[split_column].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"{split_column} == {validation_value!r}인 검증 행이 없습니다. "
            f"현재 split 값: {available}"
        )

    raw_labels = pd.to_numeric(validation[target_column], errors="raise").to_numpy()
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError(f"{target_column}은 양성=1, 음성=0이어야 합니다.")
    if np.unique(raw_labels).size != 2:
        raise ValueError("검증 데이터에 양성(1)과 음성(0)이 모두 있어야 합니다.")
    return validation[features], raw_labels.astype(int)


def load_oof_probabilities(
    data_path: Path,
    target_column: str,
    probability_column: str,
    fold_column: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """각 행이 검증 fold에서 예측된 OOF 정답과 확률을 불러온다."""
    if not data_path.is_file():
        raise FileNotFoundError(f"OOF 검증 예측 파일이 없습니다: {data_path}")
    header = pd.read_csv(data_path, nrows=0).columns
    fold_is_stored = fold_column in header
    required = [target_column, probability_column]
    selected = [*required, fold_column] if fold_is_stored else required
    frame = pd.read_csv(data_path, usecols=selected)
    require_columns(frame, selected, data_path.name)

    labels = pd.to_numeric(frame[target_column], errors="raise").to_numpy()
    probabilities = pd.to_numeric(frame[probability_column], errors="raise").to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError(f"{target_column}에는 양성(1)과 음성(0)이 모두 있어야 합니다.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{probability_column}은 NaN이 없는 0~1 확률이어야 합니다.")
    if fold_is_stored:
        if frame[fold_column].isna().any() or frame[fold_column].nunique() < 2:
            raise ValueError(
                f"{fold_column}에는 각 행이 검증된 두 개 이상의 OOF fold가 있어야 합니다."
            )
        fold_counts = {
            str(key): int(value)
            for key, value in frame[fold_column].value_counts().sort_index().items()
        }
        fold_metadata = {
            "fold_column": fold_column,
            "fold_count": int(frame[fold_column].nunique()),
            "fold_rows": fold_counts,
            "fold_assignment": "stored_in_oof",
        }
    else:
        fold_metadata = {
            "fold_column": None,
            "fold_count": None,
            "fold_rows": None,
            "fold_assignment": "not_stored",
        }
    return labels.astype(int), probabilities, fold_metadata


def reconstruct_group_folds(
    data_path: Path,
    group_column: str = "id_student",
    n_splits: int = 3,
) -> dict:
    """저장되지 않은 fold를 기존 GroupKFold 규칙으로 재구성한다."""
    if n_splits < 2:
        raise ValueError("n-splits는 2 이상이어야 합니다.")
    groups = pd.read_csv(data_path, usecols=[group_column])[group_column]
    if groups.isna().any():
        raise ValueError(f"OOF 데이터의 {group_column}에 결측값이 있습니다.")
    assignments = np.zeros(len(groups), dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(groups), dtype=np.uint8)
    for fold, (_, validation_index) in enumerate(
        splitter.split(placeholder, groups=groups.to_numpy()), start=1
    ):
        assignments[validation_index] = fold
    if not np.isin(assignments, np.arange(1, n_splits + 1)).all():
        raise ValueError("모든 OOF 행에 검증 fold를 재구성하지 못했습니다.")
    group_fold_counts = pd.DataFrame(
        {group_column: groups.to_numpy(), "fold": assignments}
    ).groupby(group_column)["fold"].nunique()
    if not group_fold_counts.eq(1).all():
        raise ValueError("동일 학생이 둘 이상의 재구성 검증 fold에 포함되었습니다.")
    values, counts = np.unique(assignments, return_counts=True)
    return {
        "fold_column": None,
        "fold_count": n_splits,
        "fold_rows": {str(key): int(value) for key, value in zip(values, counts)},
        "fold_assignment": f"reconstructed GroupKFold by {group_column}",
    }


def verify_oof_source(
    source_data_path: Path,
    oof_data_path: Path,
    target_column: str,
) -> dict:
    """OOF의 복합키·정답이 제공 원본 CSV와 정확히 같은지 확인한다."""
    if not source_data_path.is_file():
        raise FileNotFoundError(f"OOF 원본 데이터 파일이 없습니다: {source_data_path}")
    columns = [*OOF_KEY_COLUMNS, target_column]
    source = pd.read_csv(source_data_path, usecols=columns)
    oof = pd.read_csv(oof_data_path, usecols=columns)
    require_columns(source, columns, source_data_path.name)
    require_columns(oof, columns, oof_data_path.name)
    source = source.sort_values(OOF_KEY_COLUMNS, kind="mergesort", ignore_index=True)
    oof = oof.sort_values(OOF_KEY_COLUMNS, kind="mergesort", ignore_index=True)
    if source.duplicated(OOF_KEY_COLUMNS).any() or oof.duplicated(OOF_KEY_COLUMNS).any():
        raise ValueError("원본 또는 OOF 데이터에 학생·과목·운영회차·주차 키 중복이 있습니다.")
    if not source.equals(oof):
        raise ValueError("OOF 복합키·정답이 제공 원본 CSV와 일치하지 않습니다.")
    return {
        "source_data_path": str(source_data_path.resolve()),
        "source_rows": len(source),
        "key_columns": OOF_KEY_COLUMNS,
        "key_and_target_match": True,
    }


def predict_positive_probabilities(
    model,
    features: pd.DataFrame,
    batch_size: int,
    positive_label: int = 1,
) -> np.ndarray:
    """predict_proba 결과에서 클래스 1의 확률을 배치 단위로 수집한다."""
    if batch_size <= 0:
        raise ValueError("batch-size는 1 이상이어야 합니다.")

    # scikit-learn Pipeline에는 train/eval 모드나 gradient tape가 없다.
    # predict_proba는 학습을 수행하지 않는 추론 API이며 이미 확률을 반환하므로
    # sigmoid를 다시 적용하지 않는다.
    classes = np.asarray(getattr(model, "classes_", []))
    matches = np.flatnonzero(classes == positive_label)
    if matches.size != 1:
        raise ValueError(
            f"모델 classes_={classes.tolist()}에서 양성 라벨 {positive_label}을 "
            "유일하게 찾을 수 없습니다."
        )
    positive_index = int(matches[0])

    batches: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = features.iloc[start : start + batch_size]
        output = np.asarray(model.predict_proba(batch), dtype=float)
        if output.ndim != 2 or positive_index >= output.shape[1]:
            raise ValueError(f"predict_proba 출력 형태가 올바르지 않습니다: {output.shape}")
        batches.append(output[:, positive_index])
    probabilities = np.concatenate(batches)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("predict_proba가 0~1 범위를 벗어난 값 또는 NaN을 반환했습니다.")
    return probabilities


def _json_value(value: Any) -> Any:
    """numpy 값과 NaN을 표준 JSON 값으로 바꾼다."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def save_threshold_plot(table: pd.DataFrame, output_path: Path) -> None:
    best = select_thresholds(table)["best_f1"]
    best_threshold = float(best["threshold"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric, label, color in (
        ("precision", "Precision", "tab:blue"),
        ("recall", "Recall", "tab:orange"),
        ("f1_score", "F1-score", "tab:green"),
    ):
        ax.plot(table["threshold"], table[metric], marker="o", label=label, color=color)
    ax.axvline(0.5, color="black", linestyle="--", label="Default threshold (0.5)")
    ax.axvline(
        best_threshold,
        color="tab:red",
        linestyle=":",
        linewidth=2,
        label=f"Best F1 threshold ({best_threshold:.3f})",
    )
    ax.set(title="Validation Metrics by Classification Threshold", xlabel="Threshold", ylabel="Metric value")
    ax.set_xlim(0, 1)
    metric_maximum = float(table[["precision", "recall", "f1_score"]].max().max())
    ax.set_ylim(0, min(1.02, max(0.1, metric_maximum * 1.15)))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_precision_recall_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    selected = select_thresholds(table)
    best = selected["best_f1"]
    default = selected["default_0_5"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="tab:blue", label="Precision-Recall curve")
    ax.scatter(best["recall"], best["precision"], color="tab:red", s=70, zorder=3,
               label=f"Best F1 threshold ({best['threshold']:.3f})")
    ax.scatter(default["recall"], default["precision"], color="black", marker="x", s=80, zorder=3,
               label="Default threshold (0.5)")
    ax.set(title="Validation Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    table: pd.DataFrame,
    output_path: Path,
    model_name: str = "model",
) -> None:
    """Save an annotated ROC curve and the selected-threshold metric table."""
    korean_font = None
    for family in ("Malgun Gothic", "NanumGothic"):
        try:
            korean_font = font_manager.FontProperties(
                fname=font_manager.findfont(family, fallback_to_default=False)
            )
            break
        except ValueError:
            continue
    korean_text = {"fontproperties": korean_font} if korean_font is not None else {}

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    selected = select_thresholds(table)
    best = selected["best_f1"]
    default = selected["default_0_5"]

    model_label = model_name.replace("_", " ").title()
    figure = plt.figure(figsize=(13, 8))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.75, 1), wspace=0.22)
    curve_axis = figure.add_subplot(grid[0, 0])
    summary_axis = figure.add_subplot(grid[0, 1])

    curve_axis.plot(fpr, tpr, color="tab:blue", linewidth=2.5, label="OOF ROC 곡선")
    curve_axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="무작위 분류기")
    curve_axis.scatter(
        1 - best["specificity"],
        best["recall"],
        color="tab:red",
        s=75,
        zorder=3,
        label=f"F1 최적 임계값 ({best['threshold']:.3f})",
    )
    curve_axis.scatter(
        1 - default["specificity"],
        default["recall"],
        color="black",
        marker="x",
        s=85,
        zorder=3,
        label="기본 임계값 (0.5)",
    )
    curve_axis.annotate(
        (
            f"F1 최적 임계값 {best['threshold']:.3f}\n"
            f"TPR={best['recall']:.4f}, FPR={1 - best['specificity']:.4f}"
        ),
        xy=(1 - best["specificity"], best["recall"]),
        xytext=(0.18, 0.26),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        **korean_text,
    )
    curve_axis.annotate(
        (
            "기본 임계값 0.500\n"
            f"TPR={default['recall']:.4f}, FPR={1 - default['specificity']:.5f}"
        ),
        xy=(1 - default["specificity"], default["recall"]),
        xytext=(0.18, 0.11),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "black"},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        **korean_text,
    )
    curve_axis.text(
        0.97,
        0.96,
        f"ROC-AUC = {best['roc_auc']:.4f}",
        transform=curve_axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=11,
    )
    curve_axis.set(xlim=(0, 1.01), ylim=(0, 1.01))
    curve_axis.set_title(f"{model_label} OOF ROC 곡선", **korean_text)
    curve_axis.set_xlabel("위양성률 (1 - 특이도)", **korean_text)
    curve_axis.set_ylabel("진양성률 (재현율)", **korean_text)
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(loc="lower right", prop=korean_font)

    metric_rows = [
        ("임계값", f"{best['threshold']:.3f}"),
        ("정확도", f"{best['accuracy']:.4%}"),
        ("정밀도", f"{best['precision']:.4%}"),
        ("재현율 / TPR", f"{best['recall']:.4%}"),
        ("특이도", f"{best['specificity']:.4%}"),
        ("F1-score", f"{best['f1_score']:.4%}"),
        ("ROC-AUC", f"{best['roc_auc']:.4f}"),
        ("PR-AUC", f"{best['pr_auc']:.4f}"),
        ("TP / FP", f"{best['TP']:,} / {best['FP']:,}"),
        ("TN / FN", f"{best['TN']:,} / {best['FN']:,}"),
        ("양성 예측 개수", f"{best['predicted_positive_count']:,}"),
        ("양성 예측 비율", f"{best['predicted_positive_ratio']:.4%}"),
    ]
    summary_axis.axis("off")
    summary_axis.set_title("F1 최적 임계값 평가 지표", loc="left", pad=14, **korean_text)
    metric_table = summary_axis.table(
        cellText=metric_rows,
        colLabels=["평가 지표", "값"],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.25, 1, 0.72],
    )
    metric_table.auto_set_font_size(False)
    metric_table.set_fontsize(10)
    metric_table.scale(1, 1.35)
    if korean_font is not None:
        for cell in metric_table.get_celld().values():
            cell.get_text().set_fontproperties(korean_font)
    summary_axis.text(
        0,
        0.18,
        "해석 주석",
        transform=summary_axis.transAxes,
        fontweight="bold",
        verticalalignment="top",
        **korean_text,
    )
    summary_axis.text(
        0,
        0.14,
        (
            "- ROC 곡선과 ROC-AUC는 원래 OOF 확률로 계산합니다.\n"
            "- 임계값을 바꿔도 ROC 곡선과 AUC는 변하지 않습니다.\n"
            "- 임계값은 같은 곡선 위의 운영 지점만 선택합니다.\n"
            "- Platt 확률 보정이 아니라 분류 임계값 조정입니다."
        ),
        transform=summary_axis.transAxes,
        verticalalignment="top",
        fontsize=9.5,
        linespacing=1.45,
        **korean_text,
    )

    figure.suptitle(
        f"{model_label}: OOF 임계값 평가와 해석 주석",
        fontsize=15,
        y=0.98,
        **korean_text,
    )
    figure.subplots_adjust(top=0.9, bottom=0.08, left=0.07, right=0.98)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    probability_column = getattr(args, "probability_column", None)
    source_data_path = getattr(args, "source_data_path", None)
    fold_column = getattr(args, "fold_column", "fold")
    group_column = getattr(args, "group_column", "id_student")
    n_splits = getattr(args, "n_splits", 3)
    if probability_column:
        target_column = args.target_column or "target_next_week_withdrawn"
        y_true, probabilities, fold_metadata = load_oof_probabilities(
            args.data_path, target_column, probability_column, fold_column
        )
        if fold_metadata["fold_count"] is None:
            fold_metadata = reconstruct_group_folds(
                args.data_path, group_column=group_column, n_splits=n_splits
            )
        source_verification = (
            verify_oof_source(source_data_path, args.data_path, target_column)
            if source_data_path
            else None
        )
        evaluation_metadata = {
            "source_mode": "precomputed_oof_probabilities",
            "model_path": None,
            "model_name": probability_column.removesuffix("_oof_probability"),
            "data_path": str(args.data_path.resolve()),
            "source_verification": source_verification,
            "validation_rows": int(y_true.size),
            "feature_count": None,
            "positive_label": 1,
            "positive_count": int(y_true.sum()),
            "positive_ratio": float(y_true.mean()),
            "inference_method": "OOF probabilities produced by predict_proba",
            "output_interpretation": "probability; sigmoid was not applied",
            **fold_metadata,
        }
    else:
        model = load_joblib_model(args.model_path)
        schema = _load_json(args.schema_path)
        features = resolve_features(model, schema)
        target_column = args.target_column or schema.get("target") or TARGET_COLUMN
        validation_features, y_true = load_validation_data(
            args.data_path,
            features,
            target_column,
            args.split_column,
            args.validation_value,
        )
        probabilities = predict_positive_probabilities(
            model, validation_features, args.batch_size, positive_label=1
        )
        evaluation_metadata = {
            "source_mode": "loaded_joblib_model",
            "model_path": str(args.model_path.resolve()),
            "data_path": str(args.data_path.resolve()),
            "schema_path": str(args.schema_path.resolve()),
            "split_filter": {"column": args.split_column, "value": args.validation_value},
            "validation_rows": int(y_true.size),
            "feature_count": len(features),
            "positive_label": 1,
            "positive_count": int(y_true.sum()),
            "positive_ratio": float(y_true.mean()),
            "inference_method": "predict_proba",
            "output_interpretation": "probability; sigmoid was not applied",
            "batch_size": args.batch_size,
        }
    thresholds = generate_thresholds(args.threshold_min, args.threshold_max, args.threshold_step)
    table = compare_thresholds(y_true, probabilities, thresholds)
    selected = select_thresholds(table, args.min_recall, args.min_precision)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "threshold_metrics.csv"
    json_path = args.output_dir / "selected_thresholds.json"
    metrics_plot_path = args.output_dir / "threshold_metrics.png"
    pr_plot_path = args.output_dir / "precision_recall_curve.png"
    roc_plot_path = args.output_dir / "roc_curve_annotated.png"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_threshold_plot(table, metrics_plot_path)
    save_precision_recall_plot(y_true, probabilities, table, pr_plot_path)
    save_roc_curve_plot(
        y_true,
        probabilities,
        table,
        roc_plot_path,
        model_name=evaluation_metadata.get("model_name", args.model_path.stem),
    )

    summary = {
        "documentation": {
            "purpose_ko": "OOF 검증 확률을 이진 예측으로 변환할 분류 임계값별 성능을 비교한 결과입니다.",
            "prediction_rule": "pred = (positive_probability >= threshold).astype(int)",
            "recommended_threshold_ko": (
                f"후보 임계값 중 F1-score가 가장 높은 {selected['best_f1']['threshold']:.3f}을 "
                "추천 임계값으로 선택했습니다."
            ),
            "notes_ko": [
                "이 결과는 Platt Scaling 같은 확률 보정이 아니라 분류 임계값 조정 결과입니다.",
                "ROC-AUC와 PR-AUC는 임계값으로 이진화하기 전의 원래 OOF 양성 확률로 계산합니다.",
                "따라서 ROC-AUC와 PR-AUC는 모든 임계값 행에서 동일하며, 임계값 변경으로 ROC Curve 자체가 바뀌지 않습니다.",
                "양성 비율이 낮은 불균형 데이터이므로 Accuracy만 보지 말고 Precision, Recall, F1-score, PR-AUC를 함께 해석해야 합니다.",
            ],
            "metric_definitions_ko": {
                "precision": "양성으로 예측한 건 중 실제 양성의 비율입니다.",
                "recall": "전체 실제 양성 중 모델이 찾아낸 비율이며 ROC의 TPR과 같습니다.",
                "specificity": "전체 실제 음성 중 모델이 음성으로 정확히 분류한 비율입니다.",
                "f1_score": "Precision과 Recall의 조화평균입니다.",
                "roc_auc": "전체 OOF 확률의 양성·음성 순위 구분 능력을 나타냅니다.",
                "pr_auc": "불균형 데이터에서 Precision과 Recall의 전체 관계를 나타냅니다.",
            },
        },
        "evaluation": {
            **evaluation_metadata,
            "seed": args.seed,
            "auc_note": (
                "ROC-AUC and PR-AUC use the original positive probabilities, so they are "
                "identical across threshold rows."
            ),
        },
        "threshold_grid": {
            "requested_min": args.threshold_min,
            "requested_max": args.threshold_max,
            "requested_step": args.threshold_step,
            "candidates": thresholds.tolist(),
            "default_0_5_always_included": True,
        },
        "selection_rule": (
            "Maximize the criterion; ties prefer the threshold closest to 0.5, "
            "then the higher threshold."
        ),
        "constraints": {"min_recall": args.min_recall, "min_precision": args.min_precision},
        "selected": selected,
        "files": {
            "threshold_metrics_csv": str(csv_path.resolve()),
            "threshold_metrics_plot": str(metrics_plot_path.resolve()),
            "precision_recall_curve": str(pr_plot_path.resolve()),
            "roc_curve_annotated": str(roc_plot_path.resolve()),
            "selected_thresholds_json": str(json_path.resolve()),
        },
    }
    summary = _json_value(summary)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return table, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--data-path",
        "--validation-data-path",
        dest="data_path",
        type=Path,
        required=True,
        help="split 포함 모델링 CSV 또는 OOF 검증 예측 CSV",
    )
    parser.add_argument(
        "--probability-column",
        help="지정하면 모델 대신 OOF CSV의 해당 양성 확률을 사용",
    )
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--group-column", default="id_student")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--source-data-path",
        type=Path,
        help="선택 사항: OOF 키·정답과 대조할 원본 feature CSV",
    )
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--min-precision", type=float)
    parser.add_argument("--target-column", help="기본값은 feature schema의 target 또는 'target'")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--validation-value", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_ratio(value: float | None, name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name}은 0과 1 사이여야 합니다.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _validate_ratio(args.min_recall, "--min-recall")
        _validate_ratio(args.min_precision, "--min-precision")
        table, summary = run_analysis(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print("\n=== Validation threshold metrics ===")
    print(table.to_string(index=False))
    print("\n=== Selected thresholds ===")
    for name, result in summary["selected"].items():
        if result is None:
            print(f"{name}: no threshold satisfies the condition")
        else:
            print(f"{name}: threshold={result['threshold']:.3f}")
    print(f"\n결과 저장 완료: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
