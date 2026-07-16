import numpy as np
import pandas as pd


KEY_COLS = [
    "code_module",
    "code_presentation",
    "id_student"
]


def prepare_assessment_events(
    assessments,
    student_assessment
):
    """학생 평가 결과와 평가 일정 정보를 연결한다."""

    assessment_events = student_assessment.merge(
        assessments,
        on="id_assessment",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    if len(assessment_events) != len(student_assessment):
        raise ValueError(
            "평가정보 병합 과정에서 행 수가 변경됐습니다."
        )

    if assessment_events["_merge"].ne("both").any():
        raise ValueError(
            "평가정보와 연결되지 않은 제출 기록이 있습니다."
        )

    return assessment_events.drop(columns="_merge")


def build_assessment_features(
    snapshot,
    cutoff_week,
    assessments,
    assessment_events
):
    """특정 주차 종료 시점까지 확인 가능한 평가 Feature를 생성한다."""

    cutoff_day = cutoff_week * 7 - 1

    # 예측 대상 학생-강좌 명단
    student_courses = (
        snapshot[KEY_COLS]
        .drop_duplicates()
        .copy()
    )

    # 해당 시점까지 마감된 평가
    due_assessments = assessments.loc[
        assessments["date"].notna()
        & (assessments["date"] <= cutoff_day)
    ].copy()

    # 학생-강좌와 마감 평가 연결
    schedule = student_courses.merge(
        due_assessments,
        on=["code_module", "code_presentation"],
        how="left"
    )

    schedule["id_assessment"] = (
        schedule["id_assessment"].astype("Int64")
    )

    # 해당 시점까지 확인 가능한 제출만 사용
    known_submissions = assessment_events.loc[
        assessment_events["date_submitted"] <= cutoff_day,
        [
            "id_assessment",
            "id_student",
            "date_submitted",
            "is_banked",
            "score"
        ]
    ].copy()

    known_submissions["id_assessment"] = (
        known_submissions["id_assessment"].astype("Int64")
    )

    schedule = schedule.merge(
        known_submissions,
        on=["id_assessment", "id_student"],
        how="left",
        validate="many_to_one"
    )

    # 평가 제출 여부
    schedule["is_due"] = (
        schedule["id_assessment"].notna().astype(int)
    )

    schedule["submitted_due"] = (
        schedule["is_due"].eq(1)
        & schedule["date_submitted"].notna()
    ).astype(int)

    schedule["scored_due"] = (
        schedule["submitted_due"].eq(1)
        & schedule["score"].notna()
    ).astype(int)

    schedule["score_missing_due"] = (
        schedule["submitted_due"].eq(1)
        & schedule["score"].isna()
    ).astype(int)

    # 이월 평가와 지각 제출
    schedule["banked_due"] = (
        schedule["submitted_due"].eq(1)
        & schedule["is_banked"].eq(1)
    ).astype(int)

    schedule["late_due"] = (
        schedule["submitted_due"].eq(1)
        & schedule["date"].notna()
        & schedule["date_submitted"].gt(schedule["date"])
        & schedule["is_banked"].eq(0)
    ).astype(int)

    schedule["nonbanked_submitted"] = (
        schedule["submitted_due"].eq(1)
        & schedule["is_banked"].eq(0)
    ).astype(int)

    # 점수 계산용 컬럼
    schedule["score_for_stats"] = schedule["score"].where(
        schedule["scored_due"].eq(1)
    )

    schedule["weighted_score_part"] = (
        schedule["score"] * schedule["weight"]
    ).where(schedule["scored_due"].eq(1))

    schedule["scored_weight"] = schedule["weight"].where(
        schedule["scored_due"].eq(1)
    )

    schedule["submission_gap"] = (
        schedule["date_submitted"] - schedule["date"]
    ).where(schedule["nonbanked_submitted"].eq(1))

    # 평가 유형별 개수
    for assessment_type in ["TMA", "CMA", "Exam"]:
        name = assessment_type.lower()

        schedule[f"due_{name}"] = (
            schedule["is_due"].eq(1)
            & schedule["assessment_type"].eq(assessment_type)
        ).astype(int)

        schedule[f"submitted_{name}"] = (
            schedule["submitted_due"].eq(1)
            & schedule["assessment_type"].eq(assessment_type)
        ).astype(int)

    # 학생-강좌별 집계
    assessment_features = (
        schedule
        .groupby(KEY_COLS, as_index=False)
        .agg(
            assessment_due_count=("is_due", "sum"),
            assessment_due_weight=("weight", "sum"),
            assessment_submitted_due_count=(
                "submitted_due",
                "sum"
            ),
            assessment_scored_due_count=(
                "scored_due",
                "sum"
            ),
            assessment_missing_score_count=(
                "score_missing_due",
                "sum"
            ),
            assessment_banked_due_count=(
                "banked_due",
                "sum"
            ),
            assessment_late_count=(
                "late_due",
                "sum"
            ),
            assessment_nonbanked_submitted_count=(
                "nonbanked_submitted",
                "sum"
            ),
            assessment_mean_score=(
                "score_for_stats",
                "mean"
            ),
            assessment_median_score=(
                "score_for_stats",
                "median"
            ),
            assessment_min_score=(
                "score_for_stats",
                "min"
            ),
            assessment_max_score=(
                "score_for_stats",
                "max"
            ),
            weighted_score_sum=(
                "weighted_score_part",
                "sum"
            ),
            scored_weight_sum=(
                "scored_weight",
                "sum"
            ),
            assessment_mean_submission_gap=(
                "submission_gap",
                "mean"
            ),
            assessment_median_submission_gap=(
                "submission_gap",
                "median"
            ),
            assessment_due_tma_count=(
                "due_tma",
                "sum"
            ),
            assessment_submitted_tma_count=(
                "submitted_tma",
                "sum"
            ),
            assessment_due_cma_count=(
                "due_cma",
                "sum"
            ),
            assessment_submitted_cma_count=(
                "submitted_cma",
                "sum"
            ),
            assessment_due_exam_count=(
                "due_exam",
                "sum"
            ),
            assessment_submitted_exam_count=(
                "submitted_exam",
                "sum"
            )
        )
    )

    # 미제출 평가 개수
    assessment_features["assessment_missing_due_count"] = (
        assessment_features["assessment_due_count"]
        - assessment_features[
            "assessment_submitted_due_count"
        ]
    )

    # 평가 제출률
    assessment_features["assessment_submission_rate"] = (
        assessment_features[
            "assessment_submitted_due_count"
        ]
        / assessment_features[
            "assessment_due_count"
        ].replace(0, np.nan)
    ).fillna(0)

    # 평가 미제출률
    assessment_features["assessment_missing_due_rate"] = (
        assessment_features[
            "assessment_missing_due_count"
        ]
        / assessment_features[
            "assessment_due_count"
        ].replace(0, np.nan)
    ).fillna(0)

    # 지각 제출률
    assessment_features["assessment_late_rate"] = (
        assessment_features["assessment_late_count"]
        / assessment_features[
            "assessment_nonbanked_submitted_count"
        ].replace(0, np.nan)
    ).fillna(0)

    # 제출된 평가의 가중 평균 점수
    assessment_features[
        "assessment_weighted_mean_score"
    ] = (
        assessment_features["weighted_score_sum"]
        / assessment_features[
            "scored_weight_sum"
        ].replace(0, np.nan)
    ).fillna(0)

    # 아직 마감되지 않았더라도 이미 제출한 평가
    all_known = assessment_events.loc[
        assessment_events["date_submitted"] <= cutoff_day
    ].copy()

    all_known["known_score_missing"] = (
        all_known["score"].isna().astype(int)
    )

    all_known["known_banked"] = (
        all_known["is_banked"].eq(1).astype(int)
    )

    all_known_features = (
        all_known
        .groupby(KEY_COLS, as_index=False)
        .agg(
            any_known_submission_count=(
                "id_assessment",
                "size"
            ),
            any_known_scored_count=(
                "score",
                "count"
            ),
            any_known_score_missing_count=(
                "known_score_missing",
                "sum"
            ),
            any_known_banked_count=(
                "known_banked",
                "sum"
            ),
            any_known_mean_score=(
                "score",
                "mean"
            ),
            any_known_median_score=(
                "score",
                "median"
            )
        )
    )

    assessment_features = assessment_features.merge(
        all_known_features,
        on=KEY_COLS,
        how="left",
        validate="one_to_one"
    )

    assessment_feature_columns = [
        column
        for column in assessment_features.columns
        if column not in KEY_COLS
    ]

    assessment_features[assessment_feature_columns] = (
        assessment_features[
            assessment_feature_columns
        ].fillna(0)
    )

    # 기존 VLE Snapshot과 평가 Feature 연결
    final_snapshot = snapshot.merge(
        assessment_features,
        on=KEY_COLS,
        how="left",
        validate="one_to_one"
    )

    final_snapshot[assessment_feature_columns] = (
        final_snapshot[
            assessment_feature_columns
        ].fillna(0)
    )

    # 최종 검증
    if len(final_snapshot) != len(snapshot):
        raise ValueError(
            "평가 Feature 병합 후 Snapshot 행 수가 변경됐습니다."
        )

    if final_snapshot.duplicated(KEY_COLS).any():
        raise ValueError(
            "평가 Feature 병합 후 학생-강좌 키가 중복됐습니다."
        )

    return final_snapshot, assessment_features