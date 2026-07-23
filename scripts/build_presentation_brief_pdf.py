from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "oulad_presentation_brief.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6FA3")
MID_BLUE = colors.HexColor("#D9EAF4")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALEST_BLUE = colors.HexColor("#F7FAFC")
PALE_GREEN = colors.HexColor("#EAF5F1")
PALE_YELLOW = colors.HexColor("#FFF7E3")
PALE_RED = colors.HexColor("#FCEEEF")
TEXT = colors.HexColor("#243746")
MUTED = colors.HexColor("#607789")
LINE = colors.HexColor("#C8D9E4")
WHITE = colors.white


def register_fonts():
    pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", r"C:\Windows\Fonts\malgunbd.ttf"))
    pdfmetrics.registerFontFamily(
        "Malgun", normal="Malgun", bold="Malgun-Bold", italic="Malgun", boldItalic="Malgun-Bold"
    )


register_fonts()
styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "TitleKo", parent=styles["Title"], fontName="Malgun-Bold", fontSize=26,
    leading=35, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8,
)
SUBTITLE = ParagraphStyle(
    "SubtitleKo", parent=styles["Normal"], fontName="Malgun", fontSize=13,
    leading=20, textColor=MUTED, spaceAfter=14,
)
SECTION = ParagraphStyle(
    "SectionKo", parent=styles["Heading1"], fontName="Malgun-Bold", fontSize=18,
    leading=25, textColor=NAVY, spaceBefore=2, spaceAfter=10,
)
H2 = ParagraphStyle(
    "H2Ko", parent=styles["Heading2"], fontName="Malgun-Bold", fontSize=12.5,
    leading=18, textColor=BLUE, spaceBefore=7, spaceAfter=5,
)
BODY = ParagraphStyle(
    "BodyKo", parent=styles["BodyText"], fontName="Malgun", fontSize=9.6,
    leading=15.2, textColor=TEXT, spaceAfter=5,
)
BODY_BOLD = ParagraphStyle(
    "BodyBoldKo", parent=BODY, fontName="Malgun-Bold",
)
SMALL = ParagraphStyle(
    "SmallKo", parent=BODY, fontSize=8.1, leading=12.2, textColor=MUTED,
)
TABLE_BODY = ParagraphStyle(
    "TableBodyKo", parent=BODY, fontSize=8.3, leading=12.1, spaceAfter=0,
)
TABLE_HEAD = ParagraphStyle(
    "TableHeadKo", parent=TABLE_BODY, fontName="Malgun-Bold", textColor=NAVY,
    alignment=TA_CENTER,
)
QA_Q = ParagraphStyle(
    "QAQ", parent=BODY, fontName="Malgun-Bold", fontSize=10.2, leading=15,
    textColor=NAVY, spaceAfter=2,
)
QA_A = ParagraphStyle(
    "QAA", parent=BODY, fontSize=9.1, leading=14.3, spaceAfter=0,
)
COVER_SMALL = ParagraphStyle(
    "CoverSmall", parent=SMALL, alignment=TA_CENTER, fontSize=8.8, leading=13,
)


def p(text, style=BODY, raw=False):
    value = text if raw else escape(str(text)).replace("\n", "<br/>")
    return Paragraph(value, style)


def cell(text, head=False, raw=False):
    return p(text, TABLE_HEAD if head else TABLE_BODY, raw=raw)


def section_title(number, title):
    return KeepTogether([
        Table(
            [[p(f"{number:02d}", TABLE_HEAD), p(title, SECTION)]],
            colWidths=[18 * mm, 160 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), MID_BLUE),
                ("BOX", (0, 0), (0, 0), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 5),
                ("RIGHTPADDING", (0, 0), (0, 0), 5),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
        Spacer(1, 4 * mm),
    ])


def soft_table(rows, widths, header=True, font_size=8.3, paddings=5):
    data = []
    for ridx, row in enumerate(rows):
        data.append([cell(v, head=header and ridx == 0, raw=True) for v in row])
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), paddings),
        ("RIGHTPADDING", (0, 0), (-1, -1), paddings),
        ("TOPPADDING", (0, 0), (-1, -1), paddings),
        ("BOTTOMPADDING", (0, 0), (-1, -1), paddings),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), MID_BLUE),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLUE),
        ])
        start = 1
    else:
        start = 0
    for ridx in range(start, len(rows)):
        commands.append(("BACKGROUND", (0, ridx), (-1, ridx), PALEST_BLUE if ridx % 2 else PALE_BLUE))
    return Table(data, colWidths=widths, repeatRows=1 if header else 0, style=TableStyle(commands))


def numbered_cards(items, accent=PALE_BLUE):
    rows = []
    for i, (title, body) in enumerate(items, 1):
        rows.append([
            p(str(i), TABLE_HEAD),
            p(f"<b>{escape(title)}</b><br/>{escape(body)}", TABLE_BODY, raw=True),
        ])
    table = Table(rows, colWidths=[12 * mm, 166 * mm], hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(len(rows)):
        commands.append(("BACKGROUND", (0, i), (0, i), MID_BLUE))
        commands.append(("BACKGROUND", (1, i), (1, i), accent if i % 2 == 0 else PALEST_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def metric_cards():
    data = [
        [p("271,663", TABLE_HEAD), p("1.2206%", TABLE_HEAD), p("0.110030", TABLE_HEAD), p("124", TABLE_HEAD)],
        [p("Early 평가 행", COVER_SMALL), p("다음 주 이탈률", COVER_SMALL), p("운영 임계값", COVER_SMALL), p("최종 Feature", COVER_SMALL)],
    ]
    return Table(data, colWidths=[44.5 * mm] * 4, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
        ("BACKGROUND", (0, 1), (-1, 1), PALEST_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))


def qa_block(number, question, answer):
    return KeepTogether([
        Table(
            [[p(f"Q{number}", TABLE_HEAD), p(question, QA_Q)]],
            colWidths=[15 * mm, 163 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), MID_BLUE),
                ("BACKGROUND", (1, 0), (1, 0), PALEST_BLUE),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        Table(
            [[p("A", TABLE_HEAD), p(answer, QA_A)]],
            colWidths=[15 * mm, 163 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), PALE_GREEN),
                ("BACKGROUND", (1, 0), (1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
        Spacer(1, 3.2 * mm),
    ])


class BriefDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        if doc.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(doc.leftMargin, height - 15 * mm, width - doc.rightMargin, height - 15 * mm)
            canvas.setFont("Malgun", 7.6)
            canvas.setFillColor(MUTED)
            canvas.drawString(doc.leftMargin, height - 11.5 * mm, "OULAD 다음 주 이탈 조기경보 - 발표 대비 자료")
            canvas.drawRightString(width - doc.rightMargin, 9 * mm, f"{doc.page}")
        canvas.restoreState()


def build_story():
    story = []

    # Cover
    story.extend([
        Spacer(1, 8 * mm),
        p("OULAD 학생 이탈 예측 프로젝트", TITLE),
        p("전공 3-4학년 수준 발표 대비 자료", SUBTITLE),
        Spacer(1, 3 * mm),
        Table([[p("골든타임", ParagraphStyle("CoverHero", parent=TITLE, fontSize=38, leading=48, textColor=WHITE, alignment=TA_CENTER))]],
              colWidths=[178 * mm], rowHeights=[32 * mm], style=TableStyle([
                  ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
              ])),
        Spacer(1, 4 * mm),
        p("현재까지의 학습 행동으로 다음 주 이탈을 예측하고, 제한된 상담 자원을 조기에 배분하는 CatBoost 기반 조기경보 시스템", ParagraphStyle("Lead", parent=BODY, fontName="Malgun-Bold", fontSize=13, leading=21, alignment=TA_CENTER, textColor=NAVY)),
        Spacer(1, 5 * mm),
        metric_cards(),
        Spacer(1, 6 * mm),
        p("30초 요약", H2),
        Table([[p("OULAD 7개 테이블을 학생-과목-운영 회차-주차 단위로 결합하여 다음 주 중도이탈을 예측했습니다. 동일 학생이 학습과 검증에 동시에 들어가지 않도록 학생 단위 3-Fold OOF를 사용했습니다. 1-10주차 Early 구간에서 CatBoost가 Precision 27.86%, F1 23.38%, PR-AUC 0.158890으로 가장 균형이 좋아 최종 모델로 선정되었습니다. 서비스는 확률 0.110030 이상을 위험군으로 분류하고 행동 신호에 맞는 개입안을 제안합니다.", TABLE_BODY)]],
              colWidths=[178 * mm], style=TableStyle([
                  ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
                  ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                  ("LEFTPADDING", (0, 0), (-1, -1), 9),
                  ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                  ("TOPPADDING", (0, 0), (-1, -1), 8),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
              ])),
        Spacer(1, 7 * mm),
        p("검토 기준: 원격 GitHub main 및 Google Slides 교차 확인", COVER_SMALL),
        p("작성일: 2026-07-22", COVER_SMALL),
        PageBreak(),
    ])

    # 1
    story.append(section_title(1, "비교 관점과 전체 구조"))
    story.append(p("모델 비교의 공정성을 위해 아래 조건을 먼저 고정하고, 입력 구조가 다른 실험은 별도로 해석합니다."))
    story.append(soft_table([
        ["비교 관점", "고정 조건 및 해석"],
        ["예측 질문", "현재 수강 중인 학생이 <b>다음 주에 이탈하는가</b>"],
        ["분석 단위", "학생 × 과목 × 운영 회차 × 예측 주차"],
        ["검증 방식", "동일 학생이 Fold를 넘지 않는 학생 단위 3-Fold OOF"],
        ["Early 평가", "1-10주차, 271,663행, 다음 주 이탈 3,316건(1.2206%)"],
        ["ML 비교", "CatBoost, XGBoost, Random Forest, ElasticNet"],
        ["DL 비교", "최근 4주 × 행동 11개 Feature를 이용한 GRU와 TCN"],
        ["운영 기준", "CatBoost 확률 0.110030 이상을 위험군으로 판정"],
        ["서비스 출력", "이탈확률, 위험 여부, 위험 행동, 유지 활동 제안"],
    ], [38 * mm, 140 * mm]))
    story.extend([
        Spacer(1, 6 * mm),
        p("전체 발표 구조", H2),
        soft_table([
            ["순서", "발표 흐름"],
            ["1", "문제 정의와 조기 개입의 필요성"],
            ["2", "OULAD 7개 테이블과 데이터 품질"],
            ["3", "주간 패널, 다음 주 Target, 누수 방지"],
            ["4", "ML과 DL 비교 및 CatBoost 선정"],
            ["5", "Early 임계값과 Streamlit 개입 서비스"],
            ["6", "한계, 외부 검증, 개선 방향"],
        ], [20 * mm, 158 * mm]),
        Spacer(1, 5 * mm),
        p("핵심 해석", H2),
        p("CatBoost는 124개 정형 및 누적 Feature를, GRU와 TCN은 최근 4주의 행동 11개 Feature를 사용했습니다. 따라서 딥러닝과의 비교는 알고리즘만의 순수 비교가 아니라 현재 입력 설계를 포함한 서비스 적합성 비교입니다."),
        PageBreak(),
    ])

    # 2
    story.append(section_title(2, "내용 요약본 5가지"))
    story.append(numbered_cards([
        ("문제 정의", "원격교육에서는 교직원이 이탈 징후를 직접 관찰하기 어렵기 때문에 학생이 실제로 이탈하기 전에 다음 주 위험을 예측하는 조기경보가 필요합니다."),
        ("데이터 설계", "OULAD 7개 테이블을 결합해 학생-과목-운영 회차-주차 패널을 만들었습니다. 최종 데이터는 895,005행, 126열이며 모델 입력은 124개입니다."),
        ("검증 설계", "같은 학생의 여러 과목과 주차 행이 학습과 검증에 동시에 들어가는 누수를 막기 위해 id_student 기준 3-Fold OOF를 사용했습니다."),
        ("최종 모델", "Early 구간 CatBoost는 Precision 27.86%, Recall 20.14%, F1 23.38%, PR-AUC 0.158890, ROC-AUC 0.843639를 기록했습니다."),
        ("서비스 의미", "모델의 목적은 학생을 자동 탈락자로 판정하는 것이 아니라 상담이 필요한 학생의 우선순위를 제시하고 위험 행동에 맞는 개입안을 추천하는 것입니다."),
    ]))
    story.extend([
        Spacer(1, 7 * mm),
        p("발표용 한 문장", H2),
        Table([[p("'예측 정확도를 자랑하는 프로젝트가 아니라, 매주 개입 가능한 시점과 대상을 찾아주는 운영형 조기경보 프로젝트입니다.'", ParagraphStyle("Quote", parent=BODY, fontName="Malgun-Bold", fontSize=12, leading=19, textColor=NAVY, alignment=TA_CENTER))]], colWidths=[178 * mm], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
            ("BOX", (0, 0), (-1, -1), 0.7, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ])),
        PageBreak(),
    ])

    # 3
    story.append(section_title(3, "발표 핵심점 5가지"))
    story.append(numbered_cards([
        ("최종 결과가 아니라 다음 주 예측", "매주 새로운 위험도를 계산하므로 운영자가 실제로 개입할 수 있습니다."),
        ("Accuracy보다 희소 양성 탐지가 중요", "Early 양성률이 1.22%에 불과해 Accuracy는 거의 의미가 없고 PR-AUC, Precision, Recall, F1을 함께 봐야 합니다."),
        ("CatBoost가 모든 지표에서 최고는 아님", "Recall은 ElasticNet 28.44%가 더 높지만 Precision 7.20%로 오경보 부담이 큽니다. CatBoost가 종합 균형에서 우수했습니다."),
        ("딥러닝이 항상 우수하지 않음", "희소하고 불균형한 정형 데이터에서는 범주형, 결측, 비선형 관계를 다루는 CatBoost가 더 적합했습니다."),
        ("예측과 개입 효과는 다른 문제", "모델은 위험자를 찾지만 추천 행동이 실제 이탈을 줄이는지는 A/B 테스트나 인과 검증이 추가로 필요합니다."),
    ], accent=PALE_GREEN))
    story.extend([
        Spacer(1, 7 * mm),
        p("모델 비교 핵심 수치", H2),
        soft_table([
            ["모델", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"],
            ["<b>CatBoost</b>", "<b>27.86%</b>", "20.14%", "<b>23.38%</b>", "<b>0.158890</b>", "<b>0.843639</b>"],
            ["XGBoost weighted", "20.56%", "20.75%", "20.65%", "0.118739", "0.837438"],
            ["Random Forest", "27.61%", "15.41%", "19.78%", "0.141936", "0.828475"],
            ["ElasticNet", "7.20%", "<b>28.44%</b>", "11.49%", "0.050780", "0.804845"],
        ], [43 * mm, 27 * mm, 27 * mm, 24 * mm, 28 * mm, 29 * mm], paddings=4),
        Spacer(1, 3 * mm),
        p("모든 수치는 1-10주차 학생 단위 OOF 부분집합 기준입니다.", SMALL),
        PageBreak(),
    ])

    # 4
    story.append(section_title(4, "제한값과 임계값 설정 이유 5가지"))
    story.append(soft_table([
        ["설정값", "설정 이유", "발표 시 주의점"],
        ["<b>7일 단위 주차</b>", "LMS 활동과 평가를 같은 시간축으로 정렬하고 매주 다음 주를 예측하기 위한 단위", "1주차 0-6일, 2주차 7-13일"],
        ["<b>서비스 1-10주차</b>", "이탈 위험이 초기에 높고 2주차 부근에서 가장 크게 나타나 조기 개입 가치가 높음", "학습은 전체 주차, 운영과 Early 평가는 1-10주차"],
        ["<b>학생 단위 3-Fold</b>", "동일 학생의 반복 행이 학습과 검증에 나뉘는 누수를 방지하면서 계산량과 검증 안정성을 확보", "독립 외부 Test를 의미하지 않음"],
        ["<b>임계값 0.110030</b>", "1-10주차 OOF 예측에서 F1이 최대가 되는 지점", "양성률 1.22%의 불균형 문제라 0.5가 최적값이 아님"],
        ["<b>상위 20% 지표</b>", "상담 자원이 제한된 상황에서 위험도 상위 일부가 실제 이탈자를 얼마나 포착하는지 비교", "서비스 판정 기준이 아니라 순위 비교용 지표"],
    ], [36 * mm, 85 * mm, 57 * mm]))
    story.extend([
        Spacer(1, 7 * mm),
        p("꼭 구분할 두 종류의 기준", H2),
        soft_table([
            ["기준", "역할"],
            ["CatBoost 확률 0.110030", "최종 모델의 위험군 분류 기준"],
            ["클릭 50% 이상 감소, 미제출 1건 이상", "위험 이유와 행동 추천을 만들기 위한 규칙 기반 해석 기준"],
        ], [65 * mm, 113 * mm]),
        Spacer(1, 4 * mm),
        p("두 값을 동일한 모델 임계값처럼 설명하면 안 됩니다. 모델이 위험도를 산출하고, 규칙은 사람이 이해할 수 있는 위험 요인과 개입안을 제공합니다.", SMALL),
        PageBreak(),
    ])

    # 5
    story.append(section_title(5, "데이터 설계 핵심 5가지"))
    story.append(numbered_cards([
        ("복합키 기반 주간 패널", "id_student + code_module + code_presentation + prediction_week가 한 행을 식별합니다. 복합키 중복은 0건입니다."),
        ("다음 주 Target", "withdraw_week가 prediction_week + 1과 같을 때만 1입니다. 나중에 이탈하더라도 바로 다음 주가 아니면 현재 행에서는 0입니다."),
        ("관측 가능 정보만 사용", "예측 시점 이후의 클릭, 제출, 점수와 final_result, date_unregistration, withdraw_week는 Feature에서 제외했습니다."),
        ("0과 NaN을 구분", "활동 로그 없음은 클릭 0이지만, 아직 평가가 없거나 점수가 확인되지 않은 경우는 NaN으로 유지하여 구조적 미관측을 보존합니다."),
        ("124개 Feature의 입력 계약", "학생 배경, 등록, 현재 및 누적 VLE 행동, 행동 변화량, 평가 제출과 점수를 포함하며 artifact에 이름, 순서, 범주형 목록을 저장합니다."),
    ]))
    story.extend([
        Spacer(1, 6 * mm),
        p("데이터 규모", H2),
        soft_table([
            ["구분", "값", "설명"],
            ["원천 테이블", "7개", "학생, 등록, 강좌, VLE, 콘텐츠, 평가, 제출"],
            ["최종 주간 데이터", "895,005행 × 126열", "식별자와 Target을 포함한 전체 테이블"],
            ["모델 Feature", "124개", "id_student와 Target 제외"],
            ["Early 운영 구간", "271,663행", "1-10주차"],
            ["Early 양성", "3,316건", "양성률 1.2206%"],
        ], [42 * mm, 49 * mm, 87 * mm]),
        PageBreak(),
    ])

    # 6
    story.append(section_title(6, "한계점 5가지"))
    story.append(numbered_cards([
        ("외부 일반화", "단일 영국 원격교육기관의 2013-2014년 데이터이므로 다른 대학, 국가, LMS에서도 같은 성능이 나온다고 단정할 수 없습니다."),
        ("독립 외부 Test 부재", "현재 결과와 임계값은 학생 단위 OOF에 기반하므로 완전히 독립된 최종 성능보다 다소 낙관적일 수 있습니다."),
        ("낮은 절대 포착률", "Precision 27.86%는 경보 약 3.6건 중 1건이 실제 이탈이라는 뜻이며 Recall 20.14%는 실제 이탈자의 약 80%를 놓친다는 뜻입니다."),
        ("행동 로그의 해석 한계", "클릭은 학습 시간, 이해도, 동기를 직접 측정하지 않습니다. 행동과 이탈의 연관성은 보여도 원인이라고 증명하지 못합니다."),
        ("서비스 점수 일관성", "모델 확률, 백분위 위험점수, 규칙 기반 점수가 공존하므로 발표와 화면에서 최종 모델 판정과 설명용 규칙을 명확히 구분해야 합니다."),
    ], accent=PALE_RED))
    story.extend([
        Spacer(1, 7 * mm),
        p("권장 개선 방향", H2),
        soft_table([
            ["현재 한계", "다음 단계"],
            ["단일 기관", "다기관 또는 다른 운영 회차의 외부 검증"],
            ["OOF 기반 임계값", "별도 validation에서 선택하고 독립 test에서 고정 평가"],
            ["낮은 Recall", "상담 비용별 threshold와 경보 예산 시뮬레이션"],
            ["개입 효과 미검증", "위험군 대상 A/B 테스트와 개입 효과 측정"],
            ["설명 레이어 혼재", "확률, 위험 판정, 설명 규칙의 명칭과 UI 분리"],
        ], [56 * mm, 122 * mm]),
        PageBreak(),
    ])

    # 7 Q&A pages
    story.append(section_title(7, "예상 질문과 답변 10가지"))
    qa = [
        ("왜 최종 이탈 여부가 아니라 다음 주 이탈을 예측했나요?", "최종 이탈 예측은 위험 기간이 너무 넓어 구체적인 개입 시점을 정하기 어렵습니다. 다음 주 예측은 매주 위험을 갱신하고 상담, 알림, 과제 지원을 즉시 연결할 수 있습니다."),
        ("왜 CatBoost를 최종 모델로 선택했나요?", "Early 구간에서 Precision 27.86%, F1 23.38%, PR-AUC 0.158890으로 가장 우수했고 ROC-AUC도 0.843639였습니다. Recall 하나가 아니라 오경보와 미탐의 균형을 고려했습니다."),
        ("ElasticNet의 Recall이 더 높은데 왜 사용하지 않았나요?", "ElasticNet Recall은 28.44%지만 Precision은 7.20%입니다. 실제 이탈자 한 명을 찾기 위해 너무 많은 오경보가 발생하므로 제한된 상담 자원에는 부담이 큽니다."),
        ("왜 Accuracy를 주요 지표로 사용하지 않았나요?", "Early 양성률이 1.2206%이므로 모두 비이탈로 예측해도 약 98.8% Accuracy가 나옵니다. 희소한 양성 탐지에는 PR-AUC, Precision, Recall, F1이 더 적절합니다."),
        ("왜 임계값이 0.5가 아니라 0.110030인가요?", "0.5는 관습적 기본값일 뿐 최적값이 아닙니다. 클래스 불균형이 심하고 별도 확률 보정을 하지 않았기 때문에 Early OOF에서 F1이 최대가 되는 값을 선택했습니다."),
        ("임계값을 같은 OOF에서 선택하면 과적합 아닌가요?", "가능성이 있습니다. 현재 임계값은 운영안 수립을 위한 OOF 기준입니다. 다음 단계에서는 별도 운영 회차나 연도 데이터를 validation과 test로 분리해 재검증해야 합니다."),
        ("왜 전체 주차로 학습하면서 서비스는 1-10주차만 사용하나요?", "전체 주차를 사용하면 더 많은 사례와 다양한 행동 패턴을 학습할 수 있습니다. 하지만 서비스 목적은 조기 개입이므로 실제 경보와 Early 평가는 1-10주차로 제한했습니다."),
        ("124개 Feature는 너무 많아 과적합되지 않나요?", "학생 단위 OOF로 일반화 성능을 확인했고 108개 축소 모델의 전체 주차 PR-AUC 0.093502보다 124개 모델이 0.094775로 조금 높았습니다. 차이가 작아 추가 축소 실험은 필요합니다."),
        ("딥러닝 성능이 낮았던 이유는 무엇인가요?", "GRU와 TCN은 최근 4주 행동 11개만 사용한 반면 CatBoost는 배경, 등록, 누적 행동, 평가를 포함한 124개를 사용했습니다. 딥러닝 자체보다 현재 데이터와 입력 설계에서 CatBoost가 더 적합했다고 해석합니다."),
        ("Precision 27.86%, Recall 20.14%면 실제로 사용할 수 있나요?", "완전 자동화에는 부족하지만 상담 우선순위를 정하는 보조 시스템으로는 활용할 수 있습니다. 개입 비용이 낮으면 Recall 중심으로 임계값을 낮추고, 인력이 부족하면 Precision 중심으로 높일 수 있습니다."),
    ]
    for i, (q, a) in enumerate(qa[:5], 1):
        story.append(qa_block(i, q, a))
    story.append(PageBreak())
    story.append(p("예상 질문과 답변 - 계속", SECTION))
    for i, (q, a) in enumerate(qa[5:], 6):
        story.append(qa_block(i, q, a))
    story.append(PageBreak())

    # Final source page
    story.extend([
        p("발표 직전 체크리스트", SECTION),
        numbered_cards([
            ("평가 범위 말하기", "전체 주차 지표와 Early 1-10주차 지표를 섞지 않습니다."),
            ("최고 지표 과장하지 않기", "CatBoost가 모든 지표에서 최고라고 말하지 않습니다."),
            ("0.110030의 의미", "보정된 절대 이탈확률 기준이 아니라 Early OOF F1 최적 운영 기준이라고 말합니다."),
            ("상위 20%와 임계값 분리", "Top 20%는 비교용 순위 지표, 0.110030은 서비스 분류 기준입니다."),
            ("인과 주장 피하기", "위험 신호는 연관성이며 추천 행동의 효과는 아직 검증되지 않았다고 답합니다."),
        ], accent=PALE_GREEN),
        Spacer(1, 8 * mm),
        p("근거 자료", H2),
        p('<link href="https://github.com/Team7-SKProject-2/oulad-churn-prediction/blob/main/README.md" color="#2F6FA3">GitHub main - README</link>', BODY, raw=True),
        p('<link href="https://github.com/Team7-SKProject-2/oulad-churn-prediction/blob/main/reports/final_model_comparison_report.md" color="#2F6FA3">최종 모델 비교 보고서</link>', BODY, raw=True),
        p('<link href="https://github.com/Team7-SKProject-2/oulad-churn-prediction/blob/main/reports/preprocessing_report.md" color="#2F6FA3">전처리 보고서</link>', BODY, raw=True),
        p('<link href="https://github.com/Team7-SKProject-2/oulad-churn-prediction/blob/main/docs/validation_plan.md" color="#2F6FA3">검증 계획</link>', BODY, raw=True),
        p('<link href="https://docs.google.com/presentation/d/1dVOLr6diyjyuGLbb_ELpjMsVgUz0l5HXQ6xhsnlZqNs/edit" color="#2F6FA3">Google Slides - 학생 이탈 예측 프로젝트</link>', BODY, raw=True),
        Spacer(1, 16 * mm),
        Table([[p("발표의 결론", TABLE_HEAD), p("다음 주 이탈 위험을 조기에 선별해 상담 자원을 우선 배분하는 의사결정 지원 시스템", BODY_BOLD)]], colWidths=[38 * mm, 140 * mm], style=TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), MID_BLUE),
            ("BACKGROUND", (1, 0), (1, 0), PALEST_BLUE),
            ("GRID", (0, 0), (-1, -1), 0.6, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])),
    ])
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BriefDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=20 * mm, bottomMargin=15 * mm,
        title="OULAD 학생 이탈 예측 프로젝트 발표 대비 자료",
        author="Codex",
        subject="원격 main 및 Google Slides 교차 검토",
    )
    doc.build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
