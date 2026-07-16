"""머신러닝 인수인계용 Snapshot의 무결성과 누수 위험을 점검한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CUTOFF_WEEKS = (1, 2, 4)
KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
TARGET_COLUMN = "target"
REQUIRED_FEATURES = {
    "cum_total_clicks",
    "current_total_clicks",
    "current_no_activity",
    "weeks_since_last_activity",
    "assessment_missing_due_rate",
    "assessment_late_rate",
    "any_known_mean_score",
}
FORBIDDEN_COLUMNS = {
    "final_result",
    "date_unregistration",
    "unregister_yn",
    "pre_course_unregister_yn",
    "unregister_week",
    "split",
}
FORBIDDEN_SUFFIXES = ("_vs_course_median", "_course_percentile")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_handoff_snapshot(
    frame: pd.DataFrame,
    cutoff_week: int,
    expected_columns: list[str] | None = None,
) -> None:
    """한 주차 Snapshot이 ML 인수인계 기준을 충족하는지 확인한다."""
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{cutoff_week}주차 중복 컬럼: {duplicated}")

    required = {*KEY_COLUMNS, TARGET_COLUMN, "cutoff_week", *REQUIRED_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{cutoff_week}주차 필수 컬럼 누락: {missing}")

    if expected_columns is not None and frame.columns.tolist() != expected_columns:
        raise ValueError(f"{cutoff_week}주차 컬럼 또는 순서가 다른 주차와 다릅니다.")

    if frame.empty:
        raise ValueError(f"{cutoff_week}주차 Snapshot이 비어 있습니다.")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{cutoff_week}주차 학생-강좌 키가 중복됐습니다.")
    if frame.isna().any().any():
        raise ValueError(f"{cutoff_week}주차 Snapshot에 결측치가 있습니다.")
    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{cutoff_week}주차 target이 0/1이 아닙니다.")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"{cutoff_week}주차 target 한쪽 클래스가 없습니다.")
    if not frame["cutoff_week"].eq(cutoff_week).all():
        raise ValueError(f"{cutoff_week}주차 cutoff_week 값이 일치하지 않습니다.")

    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    forbidden.extend(
        column
        for column in frame.columns
        if column.endswith(FORBIDDEN_SUFFIXES)
    )
    if forbidden:
        raise ValueError(f"{cutoff_week}주차 누수 위험 컬럼: {sorted(set(forbidden))}")

    numeric = frame.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{cutoff_week}주차 수치형 컬럼에 무한값이 있습니다.")


def build_manifest() -> dict[str, object]:
    """세 Snapshot을 검증하고 재현성 확인용 Manifest를 만든다."""
    files: dict[str, object] = {}
    expected_columns: list[str] | None = None

    for cutoff_week in CUTOFF_WEEKS:
        path = PROCESSED_DIR / f"model_snapshot_week_{cutoff_week}.csv"
        if not path.exists():
            raise FileNotFoundError(f"최종 Snapshot이 없습니다: {path}")

        frame = pd.read_csv(path)
        validate_handoff_snapshot(frame, cutoff_week, expected_columns)
        if expected_columns is None:
            expected_columns = frame.columns.tolist()

        target_counts = frame[TARGET_COLUMN].value_counts().sort_index()
        files[path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
            "columns": frame.shape[1],
            "target_0": int(target_counts.get(0, 0)),
            "target_1": int(target_counts.get(1, 0)),
        }

    return {
        "analysis_unit": KEY_COLUMNS,
        "target": "final_result == 'Withdrawn'",
        "candidate_weeks": list(CUTOFF_WEEKS),
        "ml_exclude_columns": ["id_student", "target"],
        "feature_columns": expected_columns,
        "dtypes": {
            column: str(dtype)
            for column, dtype in frame.dtypes.items()
        },
        "files": files,
    }


def main() -> None:
    manifest = build_manifest()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "preprocessing_manifest.json"
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = pd.DataFrame.from_dict(manifest["files"], orient="index")
    print(summary[["rows", "columns", "target_0", "target_1", "sha256"]])
    print(f"\n전처리 인수인계 검증 통과: {output_path}")


if __name__ == "__main__":
    main()
