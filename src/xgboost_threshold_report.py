"""가중 XGBoost OOF 평가표와 ROC 곡선을 Streamlit용으로 생성합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import PROJECT_ROOT
from .oof_streamlit_report import (
    OOFModelConfig,
    OOFModelReport,
    create_metrics_table_figure,
    create_roc_figure,
    evaluate_oof_model,
    load_oof,
    save_oof_report,
)


CONFIG = OOFModelConfig(
    model_name="XGBoost weighted",
    display_name="가중 XGBoost",
    file_prefix="xgboost",
    oof_path=(
        PROJECT_ROOT
        / "models"
        / "ML"
        / "xgboost_weekly_next_week_oof_predictions.csv"
    ),
    output_dir=PROJECT_ROOT / "outputs" / "threshold_analysis" / "xgboost",
    target_column="target_next_week_withdrawn",
    probability_column="xgboost_scaled_oof_probability",
    output_interpretation=(
        "scale_pos_weight를 적용한 predict_proba 양성 확률; sigmoid와 Platt Scaling 미적용"
    ),
)
XGBoostReport = OOFModelReport


def load_xgboost_oof(
    oof_path: Path = CONFIG.oof_path,
    target_column: str = CONFIG.target_column,
    probability_column: str = CONFIG.probability_column,
):
    return load_oof(CONFIG, oof_path, target_column, probability_column)


def evaluate_xgboost(
    oof_path: Path = CONFIG.oof_path,
    threshold: float | None = None,
    target_column: str = CONFIG.target_column,
    probability_column: str = CONFIG.probability_column,
) -> XGBoostReport:
    return evaluate_oof_model(
        CONFIG,
        oof_path,
        threshold,
        target_column,
        probability_column,
    )


def save_report(
    report: XGBoostReport,
    output_dir: Path = CONFIG.output_dir,
) -> dict[str, str]:
    return save_oof_report(
        report,
        output_dir,
        Path(__file__),
        "evaluate_xgboost",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-path", type=Path, default=CONFIG.oof_path)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--target-column", default=CONFIG.target_column)
    parser.add_argument("--probability-column", default=CONFIG.probability_column)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = evaluate_xgboost(
            args.oof_path,
            args.threshold,
            args.target_column,
            args.probability_column,
        )
        files = save_report(report, args.output_dir)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print("\n=== 가중 XGBoost 임계값 평가표 ===")
    print(report.display_table.to_string(index=False))
    print(f"\n결과 저장 완료: {args.output_dir.resolve()}")
    for name, path in files.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
