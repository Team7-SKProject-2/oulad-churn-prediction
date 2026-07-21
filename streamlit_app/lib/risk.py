"""위험도 산정 로직 & 행동추천 카탈로그.

TODO(모델링): 기존 4개 페이지(대시보드/과목별 규칙 기반)는 지금 그대로 둔다.
2/3/4번(모델 기반) 페이지는 lib/model.py의 predict_risk()로 점수를 받고,
아래 factors_for_snapshot()으로 해석 가능한 위험요인 태그만 뽑아 쓴다.
점수(모델)와 요인 태깅(규칙)은 서로 독립적인 레이어라, 모델이 나중에
교체되어도 요인/행동추천 카탈로그는 그대로 재사용 가능하다.
TODO(모델링): 지금은 사용 가능한 피처(참여도 급감, 미제출, 이전 시도 횟수, 늦은 등록)로 만든 규칙 기반 점수임. -> 모델링 기반으로 예측한 확룰 또는 o/x로 변경
features가 확정되면 이 파일의 score_row()만 학습된 모델 추론 호출로 교체하면 나머지 페이지 코드는 그대로 재사용할 수 있도록 분리해두었다.
"""
import numpy as np
from .theme import risk_grade
import pandas as pd

RISK_FACTOR_CATALOG = {
    "engagement_drop": {"label": "참여도 급감", "detail": "최근 주차 클릭 수가 이전 평균 대비 크게 감소"},
    "no_submission": {"label": "과제 미제출", "detail": "마감 기한이 지난 과제가 1건 이상 미제출 상태"},
    "prev_withdrawal": {"label": "이전 시도 이력", "detail": "과거 동일 과목 재수강 이력 있음"},
    "late_registration": {"label": "늦은 수강신청", "detail": "개강 이후 등록으로 초기 적응 지연 우려"},
}

ACTION_CATALOG = {
    "engagement_drop": {"title": "학습 리마인더 발송", "desc": "최근 미접속 학생에게 자동 알림 메시지와 이번 주 학습 목표를 안내한다."},
    "no_submission": {"title": "과제 마감 리마인더", "desc": "미제출 과제 목록과 마감일을 개별 안내하고, 필요 시 연장을 검토한다."},
    "prev_withdrawal": {"title": "개인 학습 코칭 배정", "desc": "재수강생 대상으로 맞춤 학습 코치를 배정한다."},
    "late_registration": {"title": "온보딩 자료 재안내", "desc": "강좌 오리엔테이션과 기초 자료를 다시 안내한다."},
}


def compute_risk_score(engagement_drop, missed, prev_attempts, late_registration):
    """0~100 위험 점수. 스칼라/넘파이 배열/판다스 Series 모두 입력 가능(벡터화).
    (대시보드/기존 규칙 기반 페이지 전용 — 모델 기반 페이지는 lib/model.py 참고)
    """
    engagement_drop = np.asarray(engagement_drop, dtype=float)
    missed = np.minimum(np.asarray(missed, dtype=float), 3)
    prev_attempts = np.minimum(np.asarray(prev_attempts, dtype=float), 3)
    late_registration = np.asarray(late_registration, dtype=float)
    score = 20 + engagement_drop * 25 + missed * 15 + prev_attempts * 6 + late_registration * 10
    return np.clip(score, 0, 100)


def score_row(engagement_drop: bool, missed: int, prev_attempts: int, late_registration: bool) -> float:
    """단일 학생 입력용 헬퍼(규칙 기반, 레거시)."""
    return float(compute_risk_score(engagement_drop, missed, prev_attempts, late_registration))


def factors_for(engagement_drop: bool, missed: int, prev_attempts: int, late_registration: bool) -> list[str]:
    f = []
    if engagement_drop:
        f.append("engagement_drop")
    if missed >= 1:
        f.append("no_submission")
    if prev_attempts >= 1:
        f.append("prev_withdrawal")
    if late_registration:
        f.append("late_registration")
    return f

def factors_for(engagement_drop: bool, missed: int, prev_attempts: int, late_registration: bool) -> list[str]:
    f = []
    if engagement_drop:
        f.append("engagement_drop")
    if missed >= 1:
        f.append("no_submission")
    if prev_attempts >= 1:
        f.append("prev_withdrawal")
    if late_registration:
        f.append("late_registration")
    return f

def factors_for_snapshot(row) -> list[str]:
    """vle_snapshot_week_*.csv의 한 행(Series 또는 dict형 접근 가능한 row)에서
    위험요인 태그를 뽑는다. 모델 점수와 별개로 동작하는 해석용 레이어.

    row는 pandas Series (df.itertuples / df.apply(axis=1) 등에서 얻은 행)를 가정한다.
    """
    def g(key, default=0):
        v = row[key] if key in row.index else default
        return default if pd.isna(v) else v

    import pandas as pd  # 로컬 임포트: 모듈 최상단에 pandas 의존을 늘리지 않기 위함

    f = []
    if g("current_no_activity") == 1 or g("click_change_rate") <= -0.5:
        f.append("engagement_drop")
    if g("assessment_missing_due_count") >= 1:
        f.append("no_submission")
    if g("num_of_prev_attempts") >= 1:
        f.append("prev_withdrawal")
    if g("registered_after_start") == 1:
        f.append("late_registration")
    return f


def score_snapshot_row(row) -> float:
    """스냅샷 피처 기반 규칙(도메인 지식) 점수(0~100). ML 모델을 쓰지 않는 화면(2번 페이지)에서
    사용한다. factors_for_snapshot()과 완전히 같은 피처·같은 판정 기준을 보고 점수를 매겨서,
    화면에 뜨는 '위험 점수'와 '위험 요인 설명'이 항상 같은 근거에서 나오도록 맞췄다.
    """

    def g(key, default=0):
        v = row[key] if key in row.index else default
        return default if pd.isna(v) else v

    engagement_drop = 1.0 if (g("current_no_activity") == 1 or g("click_change_rate") <= -0.5) else 0.0
    missed = min(g("assessment_missing_due_count"), 3)
    prev_attempts = min(g("num_of_prev_attempts"), 3)
    late_registration = 1.0 if g("registered_after_start") == 1 else 0.0

    score = 20 + engagement_drop * 25 + missed * 15 + prev_attempts * 6 + late_registration * 10
    return float(min(max(score, 0), 100))


def actions_for(factor_keys: list[str]) -> list[dict]:
    return [ACTION_CATALOG[k] for k in factor_keys if k in ACTION_CATALOG]


__all__ = [
    "RISK_FACTOR_CATALOG", "ACTION_CATALOG", "compute_risk_score", "score_row",
    "factors_for", "factors_for_snapshot", "actions_for", "risk_grade",
]
