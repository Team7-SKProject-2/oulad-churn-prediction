"""주차별 학습행동 Feature를 생성한다."""

from __future__ import annotations

import argparse
import re

import pandas as pd

from .data import INTERIM_DIR, KEY_COLUMNS, RAW_DIR, require_columns


CUTOFF_DAYS = {1: 6, 2: 13, 3: 20}
VLE_JOIN_COLUMNS = ["code_module", "code_presentation", "id_site"]


def _safe_feature_name(value: object) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def aggregate_vle_daily(chunksize: int = 1_000_000) -> pd.DataFrame:
    """대용량 studentVle를 읽어 학생-일자-자료유형 단위로 축약한다."""
    student_vle_path = RAW_DIR / "studentVle.csv"
    vle_path = RAW_DIR / "vle.csv"
    if not student_vle_path.exists() or not vle_path.exists():
        raise FileNotFoundError("data/raw에 vle.csv와 studentVle.csv가 필요합니다.")

    vle = pd.read_csv(vle_path)
    require_columns(vle, [*VLE_JOIN_COLUMNS, "activity_type"], "vle")
    lookup = vle[[*VLE_JOIN_COLUMNS, "activity_type"]].drop_duplicates(
        VLE_JOIN_COLUMNS
    )

    partial: list[pd.DataFrame] = []
    required = [*KEY_COLUMNS, "id_site", "date", "sum_click"]
    for chunk in pd.read_csv(student_vle_path, chunksize=chunksize):
        require_columns(chunk, required, "studentVle")
        chunk = chunk.loc[chunk["date"].between(0, max(CUTOFF_DAYS.values()))]
        if chunk.empty:
            continue
        chunk = chunk.merge(lookup, on=VLE_JOIN_COLUMNS, how="left", validate="many_to_one")
        chunk["activity_type"] = chunk["activity_type"].fillna("unknown")
        daily = (
            chunk.groupby([*KEY_COLUMNS, "date", "activity_type"], as_index=False)[
                "sum_click"
            ]
            .sum()
        )
        partial.append(daily)

    if not partial:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "date", "activity_type", "sum_click"])

    combined = pd.concat(partial, ignore_index=True)
    return (
        combined.groupby([*KEY_COLUMNS, "date", "activity_type"], as_index=False)[
            "sum_click"
        ]
        .sum()
    )


def build_vle_snapshots(daily: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """1·2·3주차 누적 VLE Feature를 모든 코호트 학생에 대해 생성한다."""
    require_columns(daily, [*KEY_COLUMNS, "date", "activity_type", "sum_click"], "daily_vle")
    require_columns(cohort, KEY_COLUMNS, "cohort")
    base = cohort[KEY_COLUMNS].drop_duplicates()
    snapshots: list[pd.DataFrame] = []

    for cutoff_week, cutoff_day in CUTOFF_DAYS.items():
        cumulative = daily.loc[daily["date"] <= cutoff_day]
        current_start = cutoff_day - 6
        current = daily.loc[daily["date"].between(current_start, cutoff_day)]

        summary = cumulative.groupby(KEY_COLUMNS, as_index=False).agg(
            cumulative_clicks=("sum_click", "sum"),
            cumulative_active_days=("date", "nunique"),
            last_active_day=("date", "max"),
        )
        recent = current.groupby(KEY_COLUMNS, as_index=False).agg(
            recent_week_clicks=("sum_click", "sum")
        )

        if cumulative.empty:
            type_features = base.copy()
        else:
            type_features = cumulative.pivot_table(
                index=KEY_COLUMNS,
                columns="activity_type",
                values="sum_click",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()
            type_features.columns = [
                column
                if column in KEY_COLUMNS
                else f"clicks_type_{_safe_feature_name(column)}"
                for column in type_features.columns
            ]

        snapshot = base.merge(summary, on=KEY_COLUMNS, how="left")
        snapshot = snapshot.merge(recent, on=KEY_COLUMNS, how="left")
        snapshot = snapshot.merge(type_features, on=KEY_COLUMNS, how="left")
        snapshot["cutoff_week"] = cutoff_week
        snapshot["recent_activity_gap"] = (
            cutoff_day - snapshot["last_active_day"]
        ).fillna(cutoff_day + 1)
        snapshot["last_active_day"] = snapshot["last_active_day"].fillna(-1)

        numeric_features = [
            column
            for column in snapshot.columns
            if column not in [*KEY_COLUMNS, "cutoff_week"]
        ]
        snapshot[numeric_features] = snapshot[numeric_features].fillna(0)
        snapshots.append(snapshot)

    return pd.concat(snapshots, ignore_index=True)


def save_vle_features() -> pd.DataFrame:
    cohort_path = INTERIM_DIR / "cohort_base.csv"
    if not cohort_path.exists():
        raise FileNotFoundError("먼저 python -m src.data --build-cohort를 실행하세요.")
    cohort = pd.read_csv(cohort_path)
    daily = aggregate_vle_daily()
    features = build_vle_snapshots(daily, cohort)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(INTERIM_DIR / "vle_weekly_features.csv", index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-vle", action="store_true", help="VLE Feature 생성")
    args = parser.parse_args()
    if not args.build_vle:
        parser.print_help()
        return
    features = save_vle_features()
    print(f"VLE Feature 생성 완료: {len(features):,}행")


if __name__ == "__main__":
    main()
