"""원본 데이터 로드, 스키마 검증, 공통 코호트 생성을 담당한다."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
TARGET_COLUMN = "target"

EDUCATION_CODES = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}

IMD_BAND_CODES = {
    "0-10%": 0,
    "10-20": 1,
    "20-30%": 2,
    "30-40%": 3,
    "40-50%": 4,
    "50-60%": 5,
    "60-70%": 6,
    "70-80%": 7,
    "80-90%": 8,
    "90-100%": 9,
}

AGE_BAND_CODES = {
    "0-35": "Y",
    "35-55": "M",
    "55<=": "S",
}

RAW_SCHEMAS: dict[str, set[str]] = {
    "courses": {"code_module", "code_presentation", "module_presentation_length"},
    "assessments": {
        "code_module",
        "code_presentation",
        "id_assessment",
        "assessment_type",
        "date",
        "weight",
    },
    "studentInfo": {*KEY_COLUMNS, "final_result"},
    "studentRegistration": {*KEY_COLUMNS, "date_registration", "date_unregistration"},
    "studentAssessment": {"id_assessment", "id_student", "date_submitted", "score"},
    "vle": {"code_module", "code_presentation", "id_site", "activity_type"},
    "studentVle": {*KEY_COLUMNS, "id_site", "date", "sum_click"},
}


def raw_path(table_name: str) -> Path:
    """원본 CSV의 프로젝트 상대경로를 반환한다."""
    return RAW_DIR / f"{table_name}.csv"


def require_columns(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    """필수 컬럼이 없으면 이해하기 쉬운 오류를 발생시킨다."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name}에 필요한 컬럼이 없습니다: {missing}")


def normalize_known_columns(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """배포본에서 깨진 것으로 확인된 컬럼명을 안전하게 보정한다."""
    if table_name == "vle" and "id_site" not in frame.columns:
        candidates = [column for column in frame.columns if "id_site" in column]
        if len(candidates) != 1:
            raise ValueError(
                "vle의 id_site 컬럼을 한 개로 특정할 수 없습니다: "
                f"{frame.columns.tolist()}"
            )
        frame = frame.rename(columns={candidates[0]: "id_site"})
    return frame


def load_raw_table(table_name: str, **read_csv_kwargs) -> pd.DataFrame:
    """data/raw에서 원본 테이블을 읽고 스키마를 확인한다."""
    path = raw_path(table_name)
    if not path.exists():
        raise FileNotFoundError(f"원본 파일이 없습니다: {path}")
    frame = pd.read_csv(path, **read_csv_kwargs)
    frame = normalize_known_columns(frame, table_name)
    required = RAW_SCHEMAS.get(table_name)
    if required:
        require_columns(frame, required, table_name)
    return frame


def check_raw_files() -> list[str]:
    """대용량 파일 전체를 읽지 않고 파일 존재 여부와 헤더를 검사한다."""
    messages: list[str] = []
    for table_name, required in RAW_SCHEMAS.items():
        path = raw_path(table_name)
        if not path.exists():
            messages.append(f"[누락] {path.name}")
            continue
        header = pd.read_csv(path, nrows=0)
        header = normalize_known_columns(header, table_name)
        require_columns(header, required, table_name)
        size_mb = path.stat().st_size / (1024 * 1024)
        messages.append(f"[정상] {path.name}: {size_mb:.1f} MB")
    return messages


def build_cohort() -> pd.DataFrame:
    """학생-과목-회차 기준 정정 코호트와 이탈 Target을 만든다."""
    student_info = load_raw_table("studentInfo")
    registration = load_raw_table("studentRegistration")

    duplicate_count = int(student_info.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(f"studentInfo 공통 키 중복: {duplicate_count}행")

    cohort = student_info.merge(
        registration,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    for column in ("date_registration", "date_unregistration"):
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")

    cohort[TARGET_COLUMN] = cohort["final_result"].eq("Withdrawn").astype("int8")
    cohort["highest_education_cd"] = cohort["highest_education"].map(
        EDUCATION_CODES
    )
    if cohort["highest_education_cd"].isna().any():
        unknown = sorted(
            cohort.loc[
                cohort["highest_education_cd"].isna(), "highest_education"
            ].dropna().unique()
        )
        raise ValueError(f"정의되지 않은 학력 구간: {unknown}")

    cohort["imd_band_missing"] = cohort["imd_band"].eq("?").astype("int8")
    cohort["imd_band"] = cohort["imd_band"].replace("?", "Unknown")
    cohort["imd_band_cd"] = (
        cohort["imd_band"].map(IMD_BAND_CODES).fillna(-1).astype("int8")
    )
    cohort["age_band_cd"] = cohort["age_band"].map(AGE_BAND_CODES)
    if cohort["age_band_cd"].isna().any():
        unknown = sorted(
            cohort.loc[cohort["age_band_cd"].isna(), "age_band"]
            .dropna()
            .unique()
        )
        raise ValueError(f"정의되지 않은 연령 구간: {unknown}")

    cohort["date_registration_missing"] = (
        cohort["date_registration"].isna().astype("int8")
    )
    cohort["unregister_yn"] = cohort["date_unregistration"].notna().map(
        {True: "Y", False: "N"}
    )
    cohort["pre_course_unregister_yn"] = cohort["date_unregistration"].lt(0).map(
        {True: "Y", False: "N"}
    )
    cohort["unregister_week"] = pd.Series(pd.NA, index=cohort.index, dtype="Int64")
    after_start = cohort["date_unregistration"].ge(0)
    cohort.loc[after_start, "unregister_week"] = (
        cohort.loc[after_start, "date_unregistration"].floordiv(7) + 1
    ).astype("Int64")

    ordered_columns = [
        *KEY_COLUMNS,
        "gender",
        "region",
        "highest_education",
        "highest_education_cd",
        "imd_band",
        "imd_band_cd",
        "imd_band_missing",
        "age_band",
        "age_band_cd",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
        TARGET_COLUMN,
        "date_registration",
        "date_registration_missing",
        "date_unregistration",
        "unregister_yn",
        "pre_course_unregister_yn",
        "unregister_week",
    ]
    cohort = cohort[ordered_columns]

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(INTERIM_DIR / "cohort_base.csv", index=False)
    cohort.to_csv(
        INTERIM_DIR / "student_registration_merged_corrected.csv",
        index=False,
    )
    return cohort


def assign_group_splits(
    frame: pd.DataFrame,
    group_column: str = "id_student",
    random_state: int = 42,
) -> pd.DataFrame:
    """동일 학생이 섞이지 않도록 train/validation/test 라벨을 부여한다."""
    require_columns(frame, [group_column], "modeling_data")
    result = frame.copy()

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(result, groups=result[group_column]))
    train_val = result.iloc[train_val_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx_local, val_idx_local = next(
        inner.split(train_val, groups=train_val[group_column])
    )

    result["split"] = ""
    result.iloc[test_idx, result.columns.get_loc("split")] = "test"
    result.iloc[
        train_val_idx[train_idx_local], result.columns.get_loc("split")
    ] = "train"
    result.iloc[
        train_val_idx[val_idx_local], result.columns.get_loc("split")
    ] = "validation"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="원본 파일과 헤더 확인")
    parser.add_argument(
        "--build-cohort", action="store_true", help="공통 코호트 CSV 생성"
    )
    args = parser.parse_args()

    if not args.check and not args.build_cohort:
        parser.print_help()
        return
    if args.check:
        print("\n".join(check_raw_files()))
    if args.build_cohort:
        cohort = build_cohort()
        print(f"코호트 생성 완료: {len(cohort):,}행")


if __name__ == "__main__":
    main()
