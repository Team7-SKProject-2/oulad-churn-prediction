from pathlib import Path
import gc

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STUDENT_VLE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "studentVle.csv"
)
VLE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "vle.csv"
)
INTERIM_DIR = (
    PROJECT_ROOT / "data" / "interim"
)

CHUNK_SIZE = 500_000

WEEKLY_KEYS = [
    "code_module",
    "code_presentation",
    "id_student",
    "week_index",
]

PRE_COURSE_KEYS = [
    "code_module",
    "code_presentation",
    "id_student",
]

MERGE_KEYS = [
    "code_module",
    "code_presentation",
    "id_site",
]

CORE_ACTIVITY_TYPES = [
    "forumng",
    "quiz",
    "oucontent",
    "resource",
]

ACTIVITY_FEATURES = [
    "forumng_clicks",
    "quiz_clicks",
    "oucontent_clicks",
    "resource_clicks",
    "other_clicks",
]


def load_vle_metadata() -> pd.DataFrame:
    """vle.csv를 읽고 컬럼명과 결측치를 정리한다."""

    vle = pd.read_csv(
        VLE_PATH,
        na_values=["?"],
    )

    if "id_site" not in vle.columns:
        candidates = [
            col for col in vle.columns
            if "id_site" in col
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"id_site 컬럼을 찾을 수 없습니다: {vle.columns.tolist()}"
            )

        vle = vle.rename(
            columns={candidates[0]: "id_site"}
        )

    for col in ["week_from", "week_to"]:
        vle[col] = pd.to_numeric(
            vle[col],
            errors="coerce",
        ).astype("Int64")

    required_columns = (
        MERGE_KEYS + ["activity_type"]
    )

    if vle[required_columns].isna().sum().sum() != 0:
        raise ValueError(
            "VLE 병합 핵심 컬럼에 결측치가 있습니다."
        )

    if vle.duplicated(MERGE_KEYS).sum() != 0:
        raise ValueError(
            "VLE 병합 키에 중복이 있습니다."
        )

    return vle


def aggregate_weekly_base():
    """학생·강좌·주차별 기본 클릭 수를 집계한다."""

    weekly_parts = []
    pre_course_parts = []
    total_input_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            STUDENT_VLE_PATH,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        total_input_rows += len(chunk)

        pre_course = chunk[
            chunk["date"] < 0
        ].copy()

        after_start = chunk[
            chunk["date"] >= 0
        ].copy()

        after_start["week_index"] = (
            after_start["date"] // 7
        ) + 1

        weekly_part = (
            after_start
            .groupby(
                WEEKLY_KEYS,
                as_index=False,
                sort=False,
            )
            .agg(
                total_clicks=("sum_click", "sum"),
                interaction_rows=("sum_click", "size"),
            )
        )

        pre_course_part = (
            pre_course
            .groupby(
                PRE_COURSE_KEYS,
                as_index=False,
                sort=False,
            )
            .agg(
                pre_course_clicks=("sum_click", "sum"),
                pre_course_interaction_rows=(
                    "sum_click",
                    "size",
                ),
            )
        )

        weekly_parts.append(weekly_part)
        pre_course_parts.append(pre_course_part)

        if chunk_number % 5 == 0:
            print(
                f"[기본 집계] {chunk_number}개 청크 완료"
            )

    weekly_base = (
        pd.concat(
            weekly_parts,
            ignore_index=True,
        )
        .groupby(
            WEEKLY_KEYS,
            as_index=False,
            sort=False,
        )
        .agg(
            total_clicks=("total_clicks", "sum"),
            interaction_rows=("interaction_rows", "sum"),
        )
    )

    pre_course_base = (
        pd.concat(
            pre_course_parts,
            ignore_index=True,
        )
        .groupby(
            PRE_COURSE_KEYS,
            as_index=False,
            sort=False,
        )
        .agg(
            pre_course_clicks=(
                "pre_course_clicks",
                "sum",
            ),
            pre_course_interaction_rows=(
                "pre_course_interaction_rows",
                "sum",
            ),
        )
    )

    del weekly_parts
    del pre_course_parts
    gc.collect()

    return (
        weekly_base,
        pre_course_base,
        total_input_rows,
    )


def aggregate_active_days() -> pd.DataFrame:
    """학생·강좌·주차별 활동일 수를 계산한다."""

    day_parts = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            STUDENT_VLE_PATH,
            usecols=[
                "code_module",
                "code_presentation",
                "id_student",
                "date",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk = chunk[
            chunk["date"] >= 0
        ].copy()

        chunk["week_index"] = (
            chunk["date"] // 7
        ) + 1

        day_parts.append(
            chunk[
                WEEKLY_KEYS + ["date"]
            ].drop_duplicates()
        )

        if chunk_number % 5 == 0:
            print(
                f"[활동일 집계] {chunk_number}개 청크 완료"
            )

    unique_days = (
        pd.concat(
            day_parts,
            ignore_index=True,
        )
        .drop_duplicates(
            WEEKLY_KEYS + ["date"]
        )
    )

    active_days = (
        unique_days
        .groupby(
            WEEKLY_KEYS,
            as_index=False,
            sort=False,
        )
        .agg(
            active_days=("date", "nunique")
        )
    )

    del day_parts
    del unique_days
    gc.collect()

    return active_days


def aggregate_unique_sites() -> pd.DataFrame:
    """학생·강좌·주차별 고유 콘텐츠 수를 계산한다."""

    site_parts = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            STUDENT_VLE_PATH,
            usecols=[
                "code_module",
                "code_presentation",
                "id_student",
                "id_site",
                "date",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk = chunk[
            chunk["date"] >= 0
        ].copy()

        chunk["week_index"] = (
            chunk["date"] // 7
        ) + 1

        site_parts.append(
            chunk[
                WEEKLY_KEYS + ["id_site"]
            ].drop_duplicates()
        )

        if chunk_number % 5 == 0:
            print(
                f"[콘텐츠 집계] {chunk_number}개 청크 완료"
            )

    unique_sites = (
        pd.concat(
            site_parts,
            ignore_index=True,
        )
        .drop_duplicates(
            WEEKLY_KEYS + ["id_site"]
        )
    )

    weekly_unique_sites = (
        unique_sites
        .groupby(
            WEEKLY_KEYS,
            as_index=False,
            sort=False,
        )
        .agg(
            unique_sites=("id_site", "nunique")
        )
    )

    del site_parts
    del unique_sites
    gc.collect()

    return weekly_unique_sites


def aggregate_activity_types(
    vle: pd.DataFrame,
):
    """학생·강좌·주차별 활동 유형 클릭 수를 계산한다."""

    type_keys = WEEKLY_KEYS + ["activity_type"]
    activity_parts = []
    unmatched_rows = 0

    vle_lookup = vle[
        MERGE_KEYS + ["activity_type"]
    ].copy()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            STUDENT_VLE_PATH,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk = chunk[
            chunk["date"] >= 0
        ].copy()

        chunk["week_index"] = (
            chunk["date"] // 7
        ) + 1

        chunk = chunk.merge(
            vle_lookup,
            on=MERGE_KEYS,
            how="left",
            validate="many_to_one",
        )

        unmatched_rows += (
            chunk["activity_type"].isna().sum()
        )

        activity_parts.append(
            chunk
            .groupby(
                type_keys,
                as_index=False,
                sort=False,
            )
            .agg(
                type_clicks=("sum_click", "sum")
            )
        )

        if chunk_number % 5 == 0:
            print(
                f"[활동 유형 집계] {chunk_number}개 청크 완료"
            )

    activity_type_totals = (
        pd.concat(
            activity_parts,
            ignore_index=True,
        )
        .groupby(
            type_keys,
            as_index=False,
            sort=False,
        )
        .agg(
            type_clicks=("type_clicks", "sum")
        )
    )

    activity_type_count = (
        activity_type_totals
        .groupby(
            WEEKLY_KEYS,
            as_index=False,
            sort=False,
        )
        .agg(
            activity_type_count=(
                "activity_type",
                "nunique",
            )
        )
    )

    activity_type_totals["activity_group"] = (
        activity_type_totals["activity_type"]
        .where(
            activity_type_totals[
                "activity_type"
            ].isin(CORE_ACTIVITY_TYPES),
            "other",
        )
    )

    activity_group_totals = (
        activity_type_totals
        .groupby(
            WEEKLY_KEYS + ["activity_group"],
            as_index=False,
            sort=False,
        )
        .agg(
            activity_clicks=("type_clicks", "sum")
        )
    )

    activity_wide = (
        activity_group_totals
        .pivot(
            index=WEEKLY_KEYS,
            columns="activity_group",
            values="activity_clicks",
        )
        .fillna(0)
        .reset_index()
    )

    activity_wide.columns.name = None

    activity_wide = activity_wide.rename(
        columns={
            "forumng": "forumng_clicks",
            "quiz": "quiz_clicks",
            "oucontent": "oucontent_clicks",
            "resource": "resource_clicks",
            "other": "other_clicks",
        }
    )

    for col in ACTIVITY_FEATURES:
        if col not in activity_wide.columns:
            activity_wide[col] = 0

    total_type_clicks = (
        activity_type_totals["type_clicks"].sum()
    )

    del activity_parts
    del activity_group_totals
    del activity_type_totals
    gc.collect()

    return (
        activity_type_count,
        activity_wide[
            WEEKLY_KEYS + ACTIVITY_FEATURES
        ],
        unmatched_rows,
        total_type_clicks,
    )


def build_vle_features() -> None:
    """전체 VLE 전처리를 실행하고 결과를 저장한다."""

    if not STUDENT_VLE_PATH.exists():
        raise FileNotFoundError(STUDENT_VLE_PATH)

    if not VLE_PATH.exists():
        raise FileNotFoundError(VLE_PATH)

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1. VLE 메타데이터 정리")
    vle = load_vle_metadata()

    print("2. 주차별 기본 클릭 집계")
    (
        weekly_features,
        pre_course_features,
        total_input_rows,
    ) = aggregate_weekly_base()

    print("3. 활동일 수 집계")
    active_days = aggregate_active_days()

    weekly_features = weekly_features.merge(
        active_days,
        on=WEEKLY_KEYS,
        how="left",
        validate="one_to_one",
    )

    del active_days
    gc.collect()

    print("4. 고유 콘텐츠 수 집계")
    unique_sites = aggregate_unique_sites()

    weekly_features = weekly_features.merge(
        unique_sites,
        on=WEEKLY_KEYS,
        how="left",
        validate="one_to_one",
    )

    del unique_sites
    gc.collect()

    weekly_features[
        "avg_clicks_per_active_day"
    ] = (
        weekly_features["total_clicks"]
        / weekly_features["active_days"]
    )

    weekly_features[
        "avg_clicks_per_site"
    ] = (
        weekly_features["total_clicks"]
        / weekly_features["unique_sites"]
    )

    print("5. 활동 유형별 클릭 집계")
    (
        activity_type_count,
        activity_wide,
        unmatched_rows,
        total_type_clicks,
    ) = aggregate_activity_types(vle)

    weekly_features = (
        weekly_features
        .merge(
            activity_type_count,
            on=WEEKLY_KEYS,
            how="left",
            validate="one_to_one",
        )
        .merge(
            activity_wide,
            on=WEEKLY_KEYS,
            how="left",
            validate="one_to_one",
        )
    )

    integer_features = [
        "total_clicks",
        "interaction_rows",
        "active_days",
        "unique_sites",
        "activity_type_count",
    ] + ACTIVITY_FEATURES

    weekly_features[integer_features] = (
        weekly_features[integer_features]
        .fillna(0)
        .astype("int64")
    )

    weekly_features = (
        weekly_features
        .sort_values(WEEKLY_KEYS)
        .reset_index(drop=True)
    )

    pre_course_features = (
        pre_course_features
        .sort_values(PRE_COURSE_KEYS)
        .reset_index(drop=True)
    )

    accounted_rows = (
        weekly_features["interaction_rows"].sum()
        + pre_course_features[
            "pre_course_interaction_rows"
        ].sum()
    )

    activity_sum = weekly_features[
        ACTIVITY_FEATURES
    ].sum(axis=1)

    if accounted_rows != total_input_rows:
        raise ValueError(
            "원본 행 수와 집계 행 수가 일치하지 않습니다."
        )

    if unmatched_rows != 0:
        raise ValueError(
            f"활동 유형 연결 실패: {unmatched_rows}행"
        )

    if total_type_clicks != weekly_features[
        "total_clicks"
    ].sum():
        raise ValueError(
            "활동 유형 클릭 합계가 일치하지 않습니다."
        )

    if (
        activity_sum
        != weekly_features["total_clicks"]
    ).any():
        raise ValueError(
            "행별 활동 유형 클릭 합계가 일치하지 않습니다."
        )

    if weekly_features.isna().sum().sum() != 0:
        raise ValueError(
            "최종 주차별 데이터에 결측치가 있습니다."
        )

    if weekly_features.duplicated(
        WEEKLY_KEYS
    ).sum() != 0:
        raise ValueError(
            "최종 주차별 데이터에 중복 키가 있습니다."
        )

    weekly_output = (
        INTERIM_DIR / "vle_weekly_features.csv"
    )
    pre_course_output = (
        INTERIM_DIR / "vle_pre_course_features.csv"
    )
    metadata_output = (
        INTERIM_DIR / "vle_metadata_clean.csv"
    )

    weekly_features.to_csv(
        weekly_output,
        index=False,
    )
    pre_course_features.to_csv(
        pre_course_output,
        index=False,
    )
    vle.to_csv(
        metadata_output,
        index=False,
    )

    print("\n전처리 완료")
    print(
        "주차별 집계:",
        f"{len(weekly_features):,}행",
    )
    print(
        "개강 전 집계:",
        f"{len(pre_course_features):,}행",
    )
    print(
        "원본 반영:",
        f"{accounted_rows:,}행",
    )
    print(
        "주차별 파일:",
        weekly_output,
    )
    print(
        "개강 전 파일:",
        pre_course_output,
    )
    print(
        "VLE 메타데이터:",
        metadata_output,
    )


if __name__ == "__main__":
    build_vle_features()