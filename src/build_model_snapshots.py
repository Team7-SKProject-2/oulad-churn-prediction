from pathlib import Path

import pandas as pd

from src.assessment_features import (
    KEY_COLS,
    build_assessment_features,
    prepare_assessment_events
)
from src.build_vle_snapshots import build_and_save_vle_snapshots


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CUTOFF_WEEKS = [1, 2, 4]

LEAKAGE_COLUMNS = {
    "final_result",
    "date_unregistration",
    "unregister_yn",
    "pre_course_unregister_yn",
    "unregister_week"
}

DISTRIBUTION_DERIVED_SUFFIXES = (
    "_vs_course_median",
    "_course_percentile",
)


def validate_snapshot(
    snapshot,
    original_row_count,
    cutoff_week
):
    """최종 머신러닝 Snapshot을 검증한다."""

    if len(snapshot) != original_row_count:
        raise ValueError(
            f"{cutoff_week}주차 Snapshot 행 수가 변경됐습니다."
        )

    if snapshot.duplicated(KEY_COLS).any():
        raise ValueError(
            f"{cutoff_week}주차 학생-강좌 키가 중복됐습니다."
        )

    missing_count = int(
        snapshot.isna().sum().sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"{cutoff_week}주차 Snapshot에 "
            f"{missing_count}개의 결측치가 있습니다."
        )

    found_leakage = sorted(
        LEAKAGE_COLUMNS.intersection(snapshot.columns)
    )

    if found_leakage:
        raise ValueError(
            f"{cutoff_week}주차 누수 컬럼 발견: "
            f"{found_leakage}"
        )

    distribution_derived = sorted(
        column
        for column in snapshot.columns
        if column.endswith(DISTRIBUTION_DERIVED_SUFFIXES)
    )
    if distribution_derived:
        raise ValueError(
            "전체 코호트 분포로 계산된 누수 위험 컬럼 발견: "
            f"{distribution_derived}"
        )

    if "target" not in snapshot.columns:
        raise ValueError(
            f"{cutoff_week}주차 Snapshot에 target이 없습니다."
        )

    invalid_target = (
        ~snapshot["target"].isin([0, 1])
    ).sum()

    if invalid_target:
        raise ValueError(
            f"{cutoff_week}주차 target에 "
            f"잘못된 값이 있습니다."
        )


def main():
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("VLE 1·2·4주차 Snapshot 생성")
    build_and_save_vle_snapshots()

    assessments_path = RAW_DIR / "assessments.csv"
    student_assessment_path = (
        RAW_DIR / "studentAssessment.csv"
    )

    if not assessments_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {assessments_path}"
        )

    if not student_assessment_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {student_assessment_path}"
        )

    print("평가 원본 데이터 로드")

    assessments = pd.read_csv(
        assessments_path,
        na_values=["?"]
    )

    student_assessment = pd.read_csv(
        student_assessment_path,
        na_values=["?"]
    )

    print(
        f"assessments: {assessments.shape}"
    )

    print(
        "studentAssessment:",
        student_assessment.shape
    )

    assessment_events = prepare_assessment_events(
        assessments=assessments,
        student_assessment=student_assessment
    )

    print(
        "평가정보 병합 완료:",
        assessment_events.shape
    )

    results = []

    for cutoff_week in CUTOFF_WEEKS:
        input_path = (
            INTERIM_DIR
            / f"vle_snapshot_week_{cutoff_week}.csv"
        )

        output_path = (
            PROCESSED_DIR
            / f"model_snapshot_week_{cutoff_week}.csv"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"VLE Snapshot이 없습니다: {input_path}"
            )

        print(
            f"\n{cutoff_week}주차 Snapshot 생성 시작"
        )

        vle_snapshot = pd.read_csv(input_path)

        final_snapshot, _ = build_assessment_features(
            snapshot=vle_snapshot,
            cutoff_week=cutoff_week,
            assessments=assessments,
            assessment_events=assessment_events
        )

        validate_snapshot(
            snapshot=final_snapshot,
            original_row_count=len(vle_snapshot),
            cutoff_week=cutoff_week
        )

        final_snapshot.to_csv(
            output_path,
            index=False
        )

        file_size_mb = round(
            output_path.stat().st_size / 1024**2,
            2
        )

        results.append({
            "cutoff_week": cutoff_week,
            "row_count": len(final_snapshot),
            "column_count": final_snapshot.shape[1],
            "target_1_count": int(
                final_snapshot["target"].sum()
            ),
            "file_size_mb": file_size_mb
        })

        print(
            f"{cutoff_week}주차 저장 완료: "
            f"{final_snapshot.shape}"
        )

    result_df = pd.DataFrame(results)

    print("\n최종 생성 결과")
    print(
        result_df.to_string(index=False)
    )

    print(
        "\n모든 머신러닝용 Snapshot 생성 완료"
    )


if __name__ == "__main__":
    main()
