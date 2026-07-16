"""주차·과목 EDA와 Streamlit에서 공통으로 사용할 요약 파일을 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CUTOFF_WEEKS = (1, 2, 4)


def build_dropout_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(INTERIM_DIR / "student_registration_merged_corrected.csv")
    courses = pd.read_csv(RAW_DIR / "courses.csv")
    frame = cohort.merge(
        courses,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )
    if frame["module_presentation_length"].isna().any():
        raise ValueError("강좌 길이가 연결되지 않은 수강 사례가 있습니다.")

    for column in ("target", "date_registration", "date_unregistration"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["target"] = frame["target"].fillna(0).astype(int)

    valid_event = (
        frame["target"].eq(1)
        & frame["date_unregistration"].notna()
        & frame["date_unregistration"].ge(0)
        & frame["date_unregistration"].le(frame["module_presentation_length"])
    )
    frame["withdraw_week"] = np.where(
        valid_event,
        np.floor(frame["date_unregistration"] / 7) + 1,
        np.nan,
    )
    max_week = int(np.ceil(frame["module_presentation_length"].max() / 7))

    overall_rows = []
    module_rows = []
    cumulative = 0
    valid_total = int(valid_event.sum())
    for week in range(1, max_week + 1):
        start_day = (week - 1) * 7
        at_risk = (
            frame["module_presentation_length"].ge(start_day)
            & (frame["date_registration"].isna() | frame["date_registration"].le(start_day))
            & (~frame["target"].eq(1) | frame["date_unregistration"].ge(start_day))
        )
        events = valid_event & frame["withdraw_week"].eq(week)
        count = int(events.sum())
        denominator = int(at_risk.sum())
        cumulative += count
        overall_rows.append(
            {
                "week_index": week,
                "at_risk_count": denominator,
                "dropout_count": count,
                "dropout_rate_pct": 100 * count / denominator if denominator else 0,
                "cumulative_dropout_count": cumulative,
                "cumulative_dropout_pct": 100 * cumulative / valid_total if valid_total else 0,
            }
        )

        for module, module_frame in frame.groupby("code_module"):
            module_at_risk = at_risk.loc[module_frame.index]
            module_events = events.loc[module_frame.index]
            module_denominator = int(module_at_risk.sum())
            module_count = int(module_events.sum())
            module_rows.append(
                {
                    "code_module": module,
                    "week_index": week,
                    "at_risk_count": module_denominator,
                    "dropout_count": module_count,
                    "dropout_rate_pct": (
                        100 * module_count / module_denominator
                        if module_denominator
                        else 0
                    ),
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(module_rows)


def build_module_behavior_summary() -> pd.DataFrame:
    rows = []
    for week in CUTOFF_WEEKS:
        path = PROCESSED_DIR / f"model_snapshot_week_{week}.csv"
        frame = pd.read_csv(path)
        grouped = (
            frame.groupby(["code_module", "target"], as_index=False)
            .agg(
                student_course_count=("id_student", "size"),
                median_cumulative_clicks=("cum_total_clicks", "median"),
                median_current_clicks=("current_total_clicks", "median"),
                no_activity_rate_pct=("current_no_activity", lambda s: 100 * s.mean()),
                median_active_days=("current_active_days", "median"),
                median_unique_sites=("current_unique_sites", "median"),
                median_known_score=("any_known_mean_score", "median"),
                mean_missing_due_rate=("assessment_missing_due_rate", "mean"),
                mean_late_rate=("assessment_late_rate", "mean"),
            )
        )
        grouped.insert(0, "cutoff_week", week)
        grouped["target_label"] = grouped["target"].map(
            {0: "Non-withdrawn", 1: "Future withdrawn"}
        )
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    weekly, module_week = build_dropout_summaries()
    behavior = build_module_behavior_summary()

    weekly.to_csv(ARTIFACTS_DIR / "weekly_dropout_summary.csv", index=False)
    module_week.to_csv(
        ARTIFACTS_DIR / "module_week_dropout_summary.csv",
        index=False,
    )
    behavior.to_csv(ARTIFACTS_DIR / "module_behavior_summary.csv", index=False)

    figure_dir = PROJECT_ROOT / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(weekly["week_index"], weekly["dropout_rate_pct"], color="#4C78A8")
    ax.set_title("Weekly dropout rate")
    ax.set_xlabel("Week")
    ax.set_ylabel("Dropout rate (%)")
    fig.tight_layout()
    fig.savefig(figure_dir / "weekly_dropout_rate.png", dpi=180)
    plt.close(fig)

    heatmap = module_week.pivot(
        index="code_module",
        columns="week_index",
        values="dropout_rate_pct",
    )
    fig, ax = plt.subplots(figsize=(18, 5))
    sns.heatmap(heatmap, cmap="OrRd", ax=ax, cbar_kws={"label": "Dropout rate (%)"})
    ax.set_title("Dropout rate by module and week")
    ax.set_xlabel("Week")
    ax.set_ylabel("Module")
    fig.tight_layout()
    fig.savefig(figure_dir / "module_week_dropout_heatmap.png", dpi=180)
    plt.close(fig)

    rules = {
        "current_no_activity": "접속 안내 및 학습 상담",
        "weeks_since_last_activity": "복귀용 짧은 콘텐츠와 일정 제안",
        "click_change_rate": "학습 장애 요인 확인 및 계획 조정",
        "assessment_missing_due_rate": "과제 마감 알림과 제출 지원",
        "assessment_late_rate": "마감 캘린더·리마인더 제공",
        "any_known_mean_score": "보충 자료와 튜터링 연결",
    }
    (ARTIFACTS_DIR / "intervention_rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("EDA·Streamlit 요약 파일 생성 완료")


if __name__ == "__main__":
    main()
