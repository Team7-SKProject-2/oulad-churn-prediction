"""OULAD 정제 데이터 로딩. 컬럼 정의는 uploads/oulad_data_spec.md 참고.

data/
  interim/
    vle_metadata_clean.csv
    vle_weekly_features.csv
    vle_pre_course_features.csv
    student_info_processed.csv
    student_registration_processed.csv
    assessments_processed.csv
    courses_processed.csv
    student_assessment_processed.csv
    model_snapshot_week_{n}.csv  / cutoff_week별 모델 피처 스냅샷 (n=1,2,4,...)
"""
import re
from pathlib import Path
import pandas as pd
import streamlit as st

def _find_project_root(start: Path) -> Path:
    """lib/data.py 위치부터 상위로 올라가며 data/interim이 있는 폴더를 찾는다.
    (app.py가 저장소 루트에 있든, app/ 하위에 있든 동작하도록)"""
    for p in [start] + list(start.parents):
        if (p / "data" / "interim").exists():
            return p
    return start.parent  # 못 찾으면 lib의 상위 폴더로 fallback


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
INTERIM = PROJECT_ROOT / "data" / "interim"

KEY = ["code_module", "code_presentation", "id_student"]
COURSE_KEY = ["code_module", "code_presentation"]

REQUIRED_FILES = [
    "student_info_processed.csv",
    "student_registration_processed.csv",
    "vle_weekly_features.csv",
    "vle_pre_course_features.csv",
    "courses_processed.csv",
    "assessments_processed.csv",
    "student_assessment_processed.csv",
]

MODEL_SNAPSHOT_GLOB = "vle_snapshot_week_*.csv"
MODEL_SNAPSHOT_RE = re.compile(r"vle_snapshot_week_(\d+)\.csv$")


def data_available() -> bool:
    return all((INTERIM / f).exists() for f in REQUIRED_FILES)


@st.cache_data(show_spinner="원본 데이터 로딩 중…")
def load_raw():
    student_info = pd.read_csv(INTERIM / "student_info_processed.csv")
    registration = pd.read_csv(INTERIM / "student_registration_processed.csv")
    weekly = pd.read_csv(INTERIM / "vle_weekly_features.csv")
    pre_course = pd.read_csv(INTERIM / "vle_pre_course_features.csv")
    courses = pd.read_csv(INTERIM / "courses_processed.csv")

    assessments = pd.read_csv(INTERIM / "assessments_processed.csv")
    assessments = assessments[assessments["date"] != "?"].copy()
    assessments["date"] = assessments["date"].astype(float)
    assessments["due_week"] = (assessments["date"] // 7 + 1).astype(int)

    student_assessment = pd.read_csv(INTERIM / "student_assessment_processed.csv")

    return {
        "student_info": student_info,
        "registration": registration,
        "weekly": weekly,
        "pre_course": pre_course,
        "courses": courses,
        "assessments": assessments,
        "student_assessment": student_assessment,
    }


@st.cache_data(show_spinner=False)
def module_list(courses: pd.DataFrame) -> list[str]:
    return sorted(courses["code_module"].unique().tolist())


@st.cache_data(show_spinner=False)
def presentation_list(courses: pd.DataFrame, code_module: str) -> list[str]:
    return sorted(courses.loc[courses["code_module"] == code_module, "code_presentation"].unique().tolist())


@st.cache_data(show_spinner=False)
def week_click_ratio(weekly: pd.DataFrame) -> pd.DataFrame:
    """직전까지의 평균 클릭 수 대비 이번 주 클릭 비율(참여도 급감 탐지용)."""
    w = weekly.sort_values(KEY + ["week_index"]).copy()
    prior_avg = w.groupby(KEY)["total_clicks"].apply(lambda s: s.shift(1).expanding().mean())
    w["prior_avg_clicks"] = prior_avg.reset_index(level=list(range(len(KEY))), drop=True).values
    w["click_ratio"] = w["total_clicks"] / w["prior_avg_clicks"]
    return w


@st.cache_data(show_spinner="미제출 과제 집계 중…")
def missed_submissions_by_week(assessments: pd.DataFrame, student_assessment: pd.DataFrame) -> pd.DataFrame:
    """(code_module, code_presentation, id_student, week_index)별 '해당 주차까지 누적 미제출' 건수.

    due_cum(주차까지 마감된 과제 수) - sub_cum(주차까지 제출한 과제 수) 로 계산한다.
    학생별로 전 주차 그리드를 만들지 않고, 그룹별 pivot + ffill로 벡터화했다.
    """
    sa = student_assessment.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "due_week"]],
        on="id_assessment", how="inner",
    )

    due_counts = (
        assessments.groupby(["code_module", "code_presentation", "due_week"]).size()
        .rename("n_due").reset_index()
        .sort_values(["code_module", "code_presentation", "due_week"])
    )
    due_counts["due_cum"] = due_counts.groupby(["code_module", "code_presentation"])["n_due"].cumsum()

    sub_counts = (
        sa.groupby(["code_module", "code_presentation", "id_student", "due_week"]).size()
        .rename("n_sub").reset_index()
        .sort_values(["code_module", "code_presentation", "id_student", "due_week"])
    )
    sub_counts["sub_cum"] = sub_counts.groupby(["code_module", "code_presentation", "id_student"])["n_sub"].cumsum()

    out_frames = []
    for (module, pres), due_grp in due_counts.groupby(["code_module", "code_presentation"]):
        weeks = due_grp["due_week"].to_numpy()
        due_cum = due_grp.set_index("due_week")["due_cum"]

        sub_grp = sub_counts[(sub_counts["code_module"] == module) & (sub_counts["code_presentation"] == pres)]
        if sub_grp.empty:
            continue
        pivot = sub_grp.pivot(index="id_student", columns="due_week", values="sub_cum")
        pivot = pivot.reindex(columns=weeks).ffill(axis=1).fillna(0)

        missed = due_cum.to_numpy()[None, :] - pivot.to_numpy()
        missed = missed.clip(min=0)

        frame = pd.DataFrame(missed, index=pivot.index, columns=weeks).reset_index()
        frame = frame.melt(id_vars="id_student", var_name="week_index", value_name="missed_this_week")
        frame["code_module"] = module
        frame["code_presentation"] = pres
        out_frames.append(frame)

    if not out_frames:
        return pd.DataFrame(columns=["code_module", "code_presentation", "id_student", "week_index", "missed_this_week"])
    return pd.concat(out_frames, ignore_index=True)[
        ["code_module", "code_presentation", "id_student", "week_index", "missed_this_week"]
    ]


@st.cache_data(show_spinner="주차별 위험 스냅샷 계산 중…")
def build_master_table(_data: dict) -> pd.DataFrame:
    """(code_module, code_presentation, id_student, week_index)별 위험 스냅샷.

    대시보드 KPI가 사용하는 규칙 기반 소스. (모델 기반 페이지는 build_model_master_table 사용)
    인자명이 `_data`인 이유: 앞에 언더스코어가 붙은 인자는 st.cache_data가 해시하지 않는다
    (dict는 해시 불가) — 대신 내부에서 사용하는 원본 CSV들이 캐시되어 있으므로 안전하다.
    """
    from .risk import compute_risk_score
    from .theme import risk_grade

    weekly = week_click_ratio(_data["weekly"])
    missed = missed_submissions_by_week(_data["assessments"], _data["student_assessment"])
    reg = _data["registration"][KEY + ["date_registration"]]
    info = _data["student_info"][KEY + ["num_of_prev_attempts", "final_result", "gender", "region", "age_band_cd", "disability"]]

    m = weekly.merge(missed, on=KEY + ["week_index"], how="left")
    m["missed_this_week"] = m["missed_this_week"].fillna(0)
    m = m.merge(reg, on=KEY, how="left")
    m = m.merge(info, on=KEY, how="left")

    m["engagement_drop"] = (m["click_ratio"] < 0.5) & m["prior_avg_clicks"].notna()
    m["late_registration"] = m["date_registration"] > 0

    m["risk_score"] = compute_risk_score(
        m["engagement_drop"], m["missed_this_week"], m["num_of_prev_attempts"], m["late_registration"]
    )
    m["risk_grade"] = m["risk_score"].apply(risk_grade)
    return m


@st.cache_data(show_spinner=False)
def latest_snapshot_per_enrollment(master: pd.DataFrame) -> pd.DataFrame:
    """각 (module, presentation, student)의 가장 최근 주차 스냅샷만 남긴다."""
    idx = master.groupby(KEY)["week_index"].idxmax()
    return master.loc[idx].reset_index(drop=True)


# ── 모델 스냅샷 ──────────────────────────────────────────────
# model_snapshot_week_{n}.csv 는 학생×과목×cutoff_week 단위 피처 테이블이다.
# 기존 주차별 vle_weekly_features 기반 파이프라인과는 별개의 소스이며,
# 2/3/4번 페이지(모델 기반)는 이쪽을 사용한다.
#
# ⚠️ 중요: 어떤 학생이 특정 cutoff_week 이후 스냅샷에서 사라진다면, 그건
# 결측이 아니라 그 이전 cutoff_week에서 이미 이탈이 확정(target=1)되어
# 더 이상 관측되지 않는다는 뜻이다. 화면에서는 "데이터 없음"이 아니라
# "이탈 확정으로 관측 종료"로 구분해서 보여줘야 한다.

def model_snapshots_available() -> bool:
    return len(available_snapshot_weeks()) > 0


@st.cache_data(show_spinner=False)
def available_snapshot_weeks() -> list[int]:
    """data/interim/vle_snapshot_week_*.csv 로 실제 존재하는 cutoff_week 목록.
    3주차처럼 파일이 없는 주차는 자동으로 선택지에서 빠진다."""
    weeks = []
    if not INTERIM.exists():
        return weeks
    for p in INTERIM.glob(MODEL_SNAPSHOT_GLOB):
        m = MODEL_SNAPSHOT_RE.search(p.name)
        if m:
            weeks.append(int(m.group(1)))
    return sorted(weeks)


@st.cache_data(show_spinner="모델 스냅샷 로딩 중…")
def load_model_snapshots() -> pd.DataFrame:
    """존재하는 모든 cutoff_week 스냅샷을 하나로 합친다."""
    frames = [pd.read_csv(INTERIM / f"vle_snapshot_week_{w}.csv") for w in available_snapshot_weeks()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def student_last_seen(snapshots: pd.DataFrame) -> pd.DataFrame:
    """(module, presentation, student)별 마지막으로 관측된 cutoff_week 행.
    이후 주차에 행이 없으면 이 행이 '이탈 확정 시점의 최종 상태'가 된다."""
    if snapshots.empty:
        return snapshots
    idx = snapshots.groupby(KEY)["cutoff_week"].idxmax()
    return snapshots.loc[idx].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cohort_template(snapshots: pd.DataFrame, code_module: str, code_presentation: str, cutoff_week: int) -> pd.Series:
    """3번(이탈예측) 페이지용: 특정 과목·운영회차·cutoff_week 코호트의
    중앙값(수치형)/최빈값(범주형)으로 만든 '평균적인 학생' 템플릿 행.
    사용자가 입력하지 않은 나머지 피처를 이걸로 채워서 모델에 넣는다."""
    cohort = snapshots[
        (snapshots["code_module"] == code_module)
        & (snapshots["code_presentation"] == code_presentation)
        & (snapshots["cutoff_week"] == cutoff_week)
    ]
    if cohort.empty:
        cohort = snapshots[snapshots["cutoff_week"] == cutoff_week]
    numeric_cols = cohort.select_dtypes(include="number").columns
    template = cohort.iloc[0].copy()
    template[numeric_cols] = cohort[numeric_cols].median(numeric_only=True)
    for c in cohort.columns.difference(numeric_cols):
        mode = cohort[c].mode()
        if not mode.empty:
            template[c] = mode.iloc[0]
    return template
