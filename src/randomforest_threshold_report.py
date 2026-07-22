"""RandomForest OOF 평가표와 ROC 곡선을 생성합니다.

이 모듈은 명령줄에서 독립 실행할 수 있으며, Streamlit에서도 평가 함수와
Matplotlib Figure 생성 함수를 그대로 불러와 사용할 수 있습니다.
"""

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


# RandomForest 학습 스크립트가 생성하는 OOF 파일과 열 이름을 기본값으로 쓴다.
CONFIG = OOFModelConfig(
    model_name="RandomForest",
    display_name="RandomForest",
    file_prefix="randomforest",
    oof_path=(
        PROJECT_ROOT
        / "models"
        / "ML"
        / "randomforest_weekly_next_week_oof_predictions.csv"
    ),
    output_dir=PROJECT_ROOT / "outputs" / "threshold_analysis" / "randomforest",
    target_column="target_next_week_withdrawn",
    probability_column="randomforest_oof_probability",
    output_interpretation=(
        "class_weight='balanced_subsample'을 적용한 RandomForest의 predict_proba "
        "양성 확률이며 sigmoid를 추가 적용하지 않음"
    ),
)
RandomForestReport = OOFModelReport


def load_randomforest_oof(
    oof_path: Path = CONFIG.oof_path,
    target_column: str = CONFIG.target_column,
    probability_column: str = CONFIG.probability_column,
):
    """RandomForest OOF 라벨과 양성 클래스 확률을 불러옵니다."""
    return load_oof(CONFIG, oof_path, target_column, probability_column)


def evaluate_randomforest(
    oof_path: Path = CONFIG.oof_path,
    threshold: float | None = None,
    target_column: str = CONFIG.target_column,
    probability_column: str = CONFIG.probability_column,
) -> RandomForestReport:
    """지정 임계값 또는 전체 OOF 기준 F1 최적 임계값에서 평가합니다."""
    return evaluate_oof_model(
        CONFIG,
        oof_path,
        threshold,
        target_column,
        probability_column,
    )


def save_report(
    report: RandomForestReport,
    output_dir: Path = CONFIG.output_dir,
) -> dict[str, str]:
    """평가지표 CSV·JSON과 표·ROC PNG를 저장합니다."""
    return save_oof_report(
        report,
        output_dir,
        Path(__file__),
        "evaluate_randomforest",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-path", type=Path, default=CONFIG.oof_path)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument(
        "--threshold",
        type=float,
        help="생략하면 전체 OOF 확률 후보 중 F1-score가 가장 높은 값을 선택합니다.",
    )
    parser.add_argument("--target-column", default=CONFIG.target_column)
    parser.add_argument("--probability-column", default=CONFIG.probability_column)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = evaluate_randomforest(
            args.oof_path,
            args.threshold,
            args.target_column,
            args.probability_column,
        )
        files = save_report(report, args.output_dir)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print("\n=== RandomForest 최적 임계값 평가 ===")
    print(report.display_table.to_string(index=False))
    print(f"\n결과 저장 완료: {args.output_dir.resolve()}")
    for name, path in files.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
