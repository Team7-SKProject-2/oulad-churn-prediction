"""주차별 VLE 집계와 학생 코호트를 1·2·4주차 Snapshot으로 변환한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CUTOFF_WEEKS = (1, 2, 4)
KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
WEEKLY_KEYS = KEY_COLUMNS + ["week_index"]

SUM_FEATURE_RENAME = {
    "total_clicks": "cum_total_clicks",
    "interaction_rows": "cum_interaction_rows",
    "active_days": "cum_active_days",
    "unique_sites": "cum_unique_site_week_count",
    "activity_type_count": "cum_activity_type_week_count",
    "forumng_clicks": "cum_forumng_clicks",
    "quiz_clicks": "cum_quiz_clicks",
    "oucontent_clicks": "cum_oucontent_clicks",
    "resource_clicks": "cum_resource_clicks",
    "other_clicks": "cum_other_clicks",
}

CURRENT_SOURCE_FEATURES = [
    "total_clicks",
    "interaction_rows",
    "active_days",
    "unique_sites",
    "activity_type_count",
    "forumng_clicks",
    "quiz_clicks",
    "oucontent_clicks",
    "resource_clicks",
    "other_clicks",
    "has_vle_record",
]
PREVIOUS_SOURCE_FEATURES = ["total_clicks", "active_days", "unique_sites"]
PRE_COURSE_FEATURES = ["pre_course_clicks", "pre_course_interaction_rows"]
SAFE_BASELINE_COLUMNS = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "imd_band_missing",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "date_registration_missing",
]
LEAKAGE_COLUMNS = {
    "final_result",
    "date_registration",
    "date_unregistration",
    "unregister_yn",
    "pre_course_unregister_yn",
    "unregister_week",
}


def _require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicates = int(frame.duplicated(keys).sum())
    if duplicates:
        raise ValueError(f"{name} 키 중복: {duplicates}행")


def load_inputs(
    interim_dir: Path = INTERIM_DIR,
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "cohort": interim_dir / "student_registration_merged_corrected.csv",
        "weekly": interim_dir / "vle_weekly_features.csv",
        "pre_course": interim_dir / "vle_pre_course_features.csv",
        "courses": raw_dir / "courses.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"필요한 파일이 없습니다: {missing}")

    cohort = pd.read_csv(paths["cohort"], na_values=["?"])
    weekly = pd.read_csv(paths["weekly"])
    pre_course = pd.read_csv(paths["pre_course"])
    courses = pd.read_csv(paths["courses"])
    _require_unique(cohort, KEY_COLUMNS, "cohort")
    _require_unique(weekly, WEEKLY_KEYS, "vle_weekly")
    _require_unique(pre_course, KEY_COLUMNS, "vle_pre_course")
    return cohort, weekly, pre_course, courses


def build_student_week_grid(
    cohort: pd.DataFrame,
    weekly: pd.DataFrame,
    courses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_base = cohort[
        KEY_COLUMNS
        + ["target", "final_result", "date_registration", "date_unregistration"]
    ].copy()
    cohort_base["date_registration"] = pd.to_numeric(
        cohort_base["date_registration"], errors="coerce"
    )
    cohort_base["date_unregistration"] = pd.to_numeric(
        cohort_base["date_unregistration"], errors="coerce"
    )

    week_table = pd.DataFrame({"week_index": range(1, max(CUTOFF_WEEKS) + 1)})
    grid = cohort_base.merge(week_table, how="cross")
    weekly_features = [column for column in weekly.columns if column not in WEEKLY_KEYS]
    grid = grid.merge(
        weekly,
        on=WEEKLY_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    grid["has_vle_record"] = grid["_merge"].eq("both").astype(int)
    grid = grid.drop(columns="_merge")
    grid[weekly_features] = grid[weekly_features].fillna(0)

    course_info = courses[
        ["code_module", "code_presentation", "module_presentation_length"]
    ].drop_duplicates()
    grid = grid.merge(
        course_info,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )
    if grid["module_presentation_length"].isna().any():
        raise ValueError("강좌 기간이 연결되지 않은 행이 있습니다.")

    grid["cutoff_day"] = grid["week_index"] * 7 - 1
    grid["registration_day_for_model"] = grid["date_registration"].fillna(0)
    valid_timing = grid["target"].eq(0) | (
        grid["target"].eq(1)
        & grid["date_unregistration"].notna()
        & grid["date_unregistration"].ge(0)
        & grid["date_unregistration"].lt(grid["module_presentation_length"])
    )
    registered = grid["registration_day_for_model"].le(grid["cutoff_day"])
    still_active = grid["target"].eq(0) | grid["date_unregistration"].gt(
        grid["cutoff_day"]
    )
    eligible = grid.loc[valid_timing & registered & still_active].copy()

    grid["registration_week_for_model"] = np.where(
        grid["registration_day_for_model"].le(0),
        1,
        grid["registration_day_for_model"].floordiv(7) + 1,
    ).astype(int)
    grid["is_exposed_week"] = grid["week_index"].ge(
        grid["registration_week_for_model"]
    )

    _require_unique(grid, WEEKLY_KEYS, "student_week_grid")
    return grid, eligible


def _add_current_and_change_features(
    snapshot: pd.DataFrame,
    grid: pd.DataFrame,
    cutoff_week: int,
) -> pd.DataFrame:
    current = grid.loc[
        grid["week_index"].eq(cutoff_week),
        KEY_COLUMNS + CURRENT_SOURCE_FEATURES,
    ].rename(
        columns={column: f"current_{column}" for column in CURRENT_SOURCE_FEATURES}
    )
    if cutoff_week == 1:
        previous = current[KEY_COLUMNS].copy()
        for column in PREVIOUS_SOURCE_FEATURES:
            previous[f"previous_{column}"] = 0
    else:
        previous = grid.loc[
            grid["week_index"].eq(cutoff_week - 1),
            KEY_COLUMNS + PREVIOUS_SOURCE_FEATURES,
        ].rename(
            columns={
                column: f"previous_{column}" for column in PREVIOUS_SOURCE_FEATURES
            }
        )

    active_history = grid.loc[
        grid["week_index"].le(cutoff_week)
        & grid["is_exposed_week"]
        & grid["has_vle_record"].eq(1)
    ]
    last_activity = active_history.groupby(KEY_COLUMNS, as_index=False).agg(
        last_active_week=("week_index", "max")
    )
    result = (
        snapshot.merge(current, on=KEY_COLUMNS, how="left", validate="one_to_one")
        .merge(previous, on=KEY_COLUMNS, how="left", validate="one_to_one")
        .merge(last_activity, on=KEY_COLUMNS, how="left", validate="one_to_one")
    )
    result["last_active_week"] = result["last_active_week"].fillna(0).astype(int)
    result["weeks_since_last_activity"] = np.where(
        result["last_active_week"].eq(0),
        result["observed_weeks"],
        cutoff_week - result["last_active_week"],
    ).astype(int)
    result["current_no_activity"] = 1 - result["current_has_vle_record"]
    result["click_change"] = (
        result["current_total_clicks"] - result["previous_total_clicks"]
    )
    result["click_change_rate"] = result["click_change"] / (
        result["previous_total_clicks"] + 1
    )
    result["active_days_change"] = (
        result["current_active_days"] - result["previous_active_days"]
    )
    result["unique_sites_change"] = (
        result["current_unique_sites"] - result["previous_unique_sites"]
    )

    for column in [
        "cum_forumng_clicks",
        "cum_quiz_clicks",
        "cum_oucontent_clicks",
        "cum_resource_clicks",
        "cum_other_clicks",
    ]:
        result[column.replace("_clicks", "_share")] = (
            result[column] / result["cum_total_clicks"].replace(0, np.nan)
        ).fillna(0)

    result["log1p_cum_total_clicks"] = np.log1p(result["cum_total_clicks"])
    result["log1p_current_total_clicks"] = np.log1p(
        result["current_total_clicks"]
    )
    result["log1p_pre_course_clicks"] = np.log1p(result["pre_course_clicks"])
    return result


def _add_baseline_and_context_features(
    snapshot: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    baseline = cohort[KEY_COLUMNS + SAFE_BASELINE_COLUMNS].drop_duplicates(KEY_COLUMNS)
    result = snapshot.merge(
        baseline,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    registration_day = result["date_registration"].fillna(0)
    result["registered_after_start"] = registration_day.gt(0).astype(int)
    result["registration_lead_days"] = (-registration_day).clip(lower=0)
    result["late_registration_days"] = registration_day.clip(lower=0)

    return result


def build_vle_snapshots(
    cohort: pd.DataFrame,
    weekly: pd.DataFrame,
    pre_course: pd.DataFrame,
    courses: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    grid, eligible = build_student_week_grid(cohort, weekly, courses)
    snapshots: dict[int, pd.DataFrame] = {}

    for cutoff_week in CUTOFF_WEEKS:
        history = grid.loc[
            grid["week_index"].le(cutoff_week) & grid["is_exposed_week"]
        ].copy()
        cumulative = (
            history.groupby(KEY_COLUMNS, as_index=False)[list(SUM_FEATURE_RENAME)]
            .sum()
            .rename(columns=SUM_FEATURE_RENAME)
        )
        exposure = history.groupby(KEY_COLUMNS, as_index=False).agg(
            observed_weeks=("week_index", "nunique"),
            active_weeks=("has_vle_record", "sum"),
        )
        base = eligible.loc[
            eligible["week_index"].eq(cutoff_week),
            KEY_COLUMNS
            + [
                "target",
                "final_result",
                "date_registration",
                "date_unregistration",
                "module_presentation_length",
            ],
        ].drop_duplicates(KEY_COLUMNS)
        snapshot = (
            base.merge(cumulative, on=KEY_COLUMNS, how="left", validate="one_to_one")
            .merge(exposure, on=KEY_COLUMNS, how="left", validate="one_to_one")
            .merge(pre_course, on=KEY_COLUMNS, how="left", validate="one_to_one")
        )
        snapshot[PRE_COURSE_FEATURES] = snapshot[PRE_COURSE_FEATURES].fillna(0)
        snapshot["inactive_weeks"] = (
            snapshot["observed_weeks"] - snapshot["active_weeks"]
        )
        snapshot["active_week_rate"] = (
            snapshot["active_weeks"] / snapshot["observed_weeks"]
        )
        snapshot["cum_avg_clicks_per_active_day"] = (
            snapshot["cum_total_clicks"]
            / snapshot["cum_active_days"].replace(0, np.nan)
        ).fillna(0)
        snapshot["cum_avg_clicks_per_site_week"] = (
            snapshot["cum_total_clicks"]
            / snapshot["cum_unique_site_week_count"].replace(0, np.nan)
        ).fillna(0)
        snapshot["cutoff_week"] = cutoff_week

        snapshot = _add_current_and_change_features(snapshot, grid, cutoff_week)
        snapshot = _add_baseline_and_context_features(snapshot, cohort)
        snapshot = snapshot.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
        first = KEY_COLUMNS + ["cutoff_week", "target"]
        snapshot = snapshot[first + [c for c in snapshot.columns if c not in first]]

        if len(snapshot) != len(base):
            raise ValueError(f"{cutoff_week}주차 Snapshot 행 수가 변경됐습니다.")
        _require_unique(snapshot, KEY_COLUMNS, f"{cutoff_week}주차 Snapshot")
        if snapshot.isna().sum().sum():
            raise ValueError(f"{cutoff_week}주차 Snapshot에 결측치가 있습니다.")
        if not snapshot["target"].isin([0, 1]).all():
            raise ValueError(f"{cutoff_week}주차 target 값이 0/1이 아닙니다.")
        snapshots[cutoff_week] = snapshot

    return snapshots


def build_and_save_vle_snapshots() -> dict[int, pd.DataFrame]:
    cohort, weekly, pre_course, courses = load_inputs()
    snapshots = build_vle_snapshots(cohort, weekly, pre_course, courses)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for week, frame in snapshots.items():
        path = INTERIM_DIR / f"vle_snapshot_week_{week}.csv"
        frame.to_csv(path, index=False)
        print(f"VLE {week}주차 Snapshot 저장: {frame.shape}")
    return snapshots


def main() -> None:
    build_and_save_vle_snapshots()


if __name__ == "__main__":
    main()
