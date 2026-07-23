"""CatBoost 1~10주차 OOF 부분집단 평가 결과를 생성합니다."""

from pathlib import Path

from .data import PROJECT_ROOT
from .early_oof_report import (
    EarlyOOFReport,
    create_metrics_table_figure,
    create_roc_figure,
    create_threshold_curve_figure,
    evaluate_early_oof_model,
    run_early_cli,
    save_early_report,
)
from .oof_streamlit_report import OOFModelConfig


CONFIG = OOFModelConfig(
    model_name="CatBoost",
    display_name="CatBoost",
    file_prefix="early_catboost",
    oof_path=PROJECT_ROOT / "models" / "ML" / "catboost_weekly_next_week_oof_predictions.csv",
    output_dir=PROJECT_ROOT / "outputs" / "threshold_analysis" / "early_catboot",
    target_column="target_next_week_withdrawn",
    probability_column="catboost_oof_probability",
    output_interpretation="CatBoost predict_proba 양성 확률이며 sigmoid를 추가 적용하지 않음",
)


def evaluate_early_catboost(oof_path: Path = CONFIG.oof_path, **kwargs) -> EarlyOOFReport:
    return evaluate_early_oof_model(CONFIG, oof_path, **kwargs)


def save_report(report: EarlyOOFReport, output_dir: Path = CONFIG.output_dir) -> dict[str, str]:
    return save_early_report(report, output_dir, Path(__file__), "evaluate_early_catboost")


def main() -> None:
    run_early_cli(CONFIG, Path(__file__), "evaluate_early_catboost")


if __name__ == "__main__":
    main()
