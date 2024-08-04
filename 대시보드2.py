import streamlit as st
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plotly.graph_objects as go

# JSONL 파일에서 데이터 불러오기
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# 데이터 불러오기
certificates_data = load_data("dashboard.jsonl")

# 데이터프레임 생성 및 전처리
df = pd.DataFrame(certificates_data)
mlb = MultiLabelBinarizer()

# 모든 관련 필드를 함께 인코딩
all_fields = df['related_fields'] + df['related_departments'] + df['related_majors']
encoded_fields = pd.DataFrame(mlb.fit_transform(all_fields),
                              columns=mlb.classes_,
                              index=df.index)

features = pd.concat([encoded_fields,
                      df['difficulty'], 
                      pd.DataFrame(df['career_level'].tolist()).max(axis=1),
                      df['popularity']], 
                      axis=1)

# 학부, 전공, 희망분야 관계 정의
departments = {
    "기계공학부": ["기계공학"],
    "메카트로닉스공학부": ["생산시스템전공", "제어시스템전공", "디지털시스템전공"],
    "전기전자통신공학부": ["전기공학전공", "전자공학전공", "정보통신공학전공"],
    "컴퓨터공학부": ["컴퓨터공학"],
    "디자인공학전공": ["디자인공학"],
    "건축공학전공": ["건축공학"],
    "에너지신소재공학전공": ["에너지신소재공학"],
    "화학생명공학전공": ["화학생명공학"],
    "산업경영학부": ["융합경영전공", "데이터경영전공"],
    "고용서비스정책학과": ["고용서비스정책"]
}

majors_fields = {
    "기계공학": ["기계설계", "자동차", "로봇", "항공우주"],
    "생산시스템전공": ["생산관리", "품질관리", "물류관리", "공정설계"],
    "제어시스템전공": ["자동제어", "로봇제어", "센서 및 계측", "스마트팩토리"],
    "디지털시스템전공": ["임베디드시스템", "사물인터넷(IoT)", "디지털신호처리", "컴퓨터비전"],
    "전기공학전공": ["전력시스템", "전기기기", "신재생에너지", "전력전자"],
    "전자공학전공": ["반도체", "디스플레이", "통신시스템", "전자회로설계"],
    "정보통신공학전공": ["네트워크", "무선통신", "정보보안", "신호처리"],
    "컴퓨터공학": ["소프트웨어개발", "인공지능", "빅데이터", "클라우드컴퓨팅"],
    "디자인공학": ["제품디자인", "UX/UI디자인", "그래픽디자인", "산업디자인"],
    "건축공학": ["건축설계", "건설관리", "건축환경", "구조공학"],
    "에너지신소재공학": ["신재생에너지", "나노소재", "전자재료", "에너지저장"],
    "화학생명공학": ["화학공정", "생물공학", "의약품개발", "환경공학"],
    "융합경영전공": ["마케팅", "재무관리", "인사관리", "전략경영"],
    "데이터경영전공": ["데이터분석", "비즈니스인텔리전스", "디지털마케팅", "핀테크"],
    "고용서비스정책": ["고용정책", "직업상담", "인적자원개발", "노동법"]
}

def recommend_certificates(user_grade, user_department, user_major, user_field, acquired_certs, top_n=5):
    # 사용자 프로필 생성
    user_profile = [1 if f in [user_field, user_department, user_major] else 0 for f in mlb.classes_]
    user_profile.extend([min(user_grade, 5) / 5,  # 난이도를 1-5 스케일로 정규화
                         user_grade,
                         5])  # 인기도는 5로 설정 (모든 자격증을 고려)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity([user_profile], features)[0]
    
    # 전공 및 희망분야에 맞는 자격증 필터링
    mask = (df['related_fields'].apply(lambda fields: user_field in fields) |
            df['related_majors'].apply(lambda majors: user_major in majors) |
            df['related_departments'].apply(lambda depts: user_department in depts))
    
    # 취득한 자격증 제외
    mask = mask & (~df['name'].isin(acquired_certs))
    
    # 학년에 따른 자격증 필터링
    mask = mask & (
        ((df['name'].str.contains('기사') & (user_grade == 4)) |
         (df['name'].str.contains('산업기사') & (user_grade >= 2)) |
         (~df['name'].str.contains('기사') & ~df['name'].str.contains('산업기사')))
    )
    
    filtered_indices = similarities[mask].argsort()[-top_n:][::-1]
    recommendations = df[mask].iloc[filtered_indices]
    
    return recommendations[['name', 'type', 'related_fields', 'difficulty', 'popularity', 'schedule', 'fee', 'description', 'graduation_requirement']]

# 새로운 함수: 동일 전공 재학생/졸업생의 자격증 통계
def get_alumni_certificates(department, major):
    # 실제 구현에서는 데이터베이스에서 이 정보를 가져와야 합니다.
    # 여기서는 예시 데이터를 사용합니다.
    example_data = [
        {"name": "정보처리기사", "count": 150},
        {"name": "리눅스마스터", "count": 100},
        {"name": "네트워크관리사", "count": 80},
        {"name": "CCNA", "count": 70},
        {"name": "데이터분석준전문가", "count": 60}
    ]
    return pd.DataFrame(example_data)

# 세션 상태 초기화
if 'acquired_certificates' not in st.session_state:
    st.session_state.acquired_certificates = []
    
# IPP 인턴십 데이터 로드 함수
def load_ipp_data():
    # 실제 구현에서는 데이터베이스나 API에서 데이터를 가져와야 합니다.
    # 여기서는 예시 데이터를 사용합니다.
    return pd.DataFrame({
        "기업명": ["테크놀로지 주식회사", "글로벌 시스템즈", "스마트 솔루션스", "이노베이션 랩스", "퓨처 테크", "메가 코퍼레이션"],
        "분야": ["소프트웨어 개발", "네트워크 엔지니어링", "데이터 분석", "인공지능", "클라우드 컴퓨팅", "로보틱스"],
        "기간": ["6개월", "4개월", "3개월", "12개월", "2개월", "9개월"],
        "지원자격": ["3학년 이상", "2학년 이상", "3학년 이상", "4학년", "2학년 이상", "3학년 이상"],
        "마감일": ["2024-09-30", "2024-08-15", "2024-10-31", "2024-09-15", "2024-08-31", "2024-11-30"],
        "관련학과": [
            ["컴퓨터공학부", "전기전자통신공학부"],
            ["전기전자통신공학부", "컴퓨터공학부"],
            ["컴퓨터공학부", "산업경영학부"],
            ["컴퓨터공학부", "전기전자통신공학부"],
            ["컴퓨터공학부", "메카트로닉스공학부"],
            ["메카트로닉스공학부", "컴퓨터공학부"]
        ],
        "우대조건": [
            ["학점 3.5 이상", "토익 700점 이상"],
            ["운전면허 보유", "한국사 자격증"],
            ["데이터 분석 관련 자격증"],
            ["인공지능 관련 프로젝트 경험", "학점 3.8 이상"],
            ["클라우드 자격증", "토익 800점 이상"],
            ["로봇 관련 경진대회 수상 경력"]
        ]
    })

# 기간을 정수로 변환하는 함수
def parse_duration(duration):
    try:
        return int(duration.split()[0])
    except (ValueError, AttributeError, IndexError):
        return 0  # 변환할 수 없는 경우 0을 반환

# 인턴십 기간 분류 함수
def classify_duration(months):
    if 1 <= months <= 4:
        return "단기 (1~4개월)"
    elif 6 <= months <= 12:
        return "장기 (6개월~1년)"
    else:
        return "기타"

    
# Streamlit 앱 설정
st.set_page_config(layout="wide", page_title="학생 종합 역량 관리 시스템")
st.title("🎓 학생 종합 역량 관리 시스템")

# 사이드바 구성
with st.sidebar:
    st.header("사용자 정보 입력")
    grade = st.selectbox("학년", [1, 2, 3, 4])
    department = st.selectbox("학부", list(departments.keys()))
    majors = departments[department]
    major = st.selectbox("전공", majors)
    fields = majors_fields[major]
    field = st.selectbox("희망분야", fields)

    # 현재 선택된 탭에 따라 다른 사이드바 내용 표시
    if st.session_state.get('current_tab') in [None, '추천 자격증', '우리 학교 재학생/졸업생이 취득한 자격증']:
        st.subheader("취득한 자격증 선택")
        all_certificates = sorted(df['name'].tolist())
        selected_cert = st.selectbox("자격증 선택", [""] + all_certificates)
        if selected_cert and st.button("추가"):
            if selected_cert not in st.session_state.acquired_certificates:
                st.session_state.acquired_certificates.append(selected_cert)
                st.success(f"'{selected_cert}'가 취득한 자격증 목록에 추가되었습니다.")
                st.rerun()
        
        st.subheader("취득한 자격증")
        for i, cert in enumerate(st.session_state.acquired_certificates):
            col1, col2 = st.columns([0.9, 0.1])
            col1.write(cert)
            if col2.button("x", key=f"remove_{i}", help="제거"):
                removed_cert = st.session_state.acquired_certificates.pop(i)
                st.success(f"'{removed_cert}'가 취득한 자격증 목록에서 제거되었습니다.")
                st.rerun()
    
    elif st.session_state.get('current_tab') == 'IPP 인턴십 공고':
        st.subheader("인턴십 검색 옵션")
        duration_options = ["단기 (1~4개월)", "장기 (6개월~1년)"]
        selected_duration = st.multiselect("인턴십 기간", options=duration_options, default=duration_options)
        
        ipp_data = load_ipp_data()
        field_options = ipp_data['분야'].unique().tolist()
        selected_fields = st.multiselect("분야 선택", options=field_options)
        
        min_gpa = st.slider("최소 학점", 0.0, 4.5, 0.0, 0.1)
        
        if st.button("인턴십 검색"):
            st.session_state.search_internships = True
        else:
            st.session_state.search_internships = False

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📊 추천 자격증", "👨‍🎓 우리 학교 재학생/졸업생이 취득한 자격증", "🏢 IPP 인턴십 공고"])


with tab1:
    if st.sidebar.button("자격증 추천 받기"):
        recommendations = recommend_certificates(grade, department, major, field, st.session_state.acquired_certificates)
        
        if recommendations.empty:
            st.warning("선택한 조건에 맞는 추천 자격증이 없습니다.")
        else:
            st.header(f"📋 {grade}학년 {department} {major} {field} 분야 추천 자격증")
            
            for _, cert in recommendations.iterrows():
                with st.expander(f"{cert['name']} - {cert['type']} | 난이도: {'🌟' * int(cert['difficulty'])} | 인기도: {'🔥' * int(cert['popularity'])} | 졸업요건: {cert['graduation_requirement']}"):
                    st.write(f"**관련 분야:** {', '.join(cert['related_fields'])}")
                    st.write(f"**시험 일정:** {cert['schedule']}")
                    st.write(f"**응시료:** {cert['fee']}")
                    st.write(f"**설명:** {cert['description']}")
                    
                    # 코멘트 섹션
                    st.subheader("💬 코멘트")
                    if 'comments' not in st.session_state:
                        st.session_state.comments = {c['name']: [] for c in certificates_data}
                    for comment in st.session_state.comments[cert['name']]:
                        st.text(comment)
                    
                    # 새 코멘트 입력
                    new_comment = st.text_input(f"'{cert['name']}'에 대한 코멘트를 남겨주세요:", key=f"comment_{cert['name']}")
                    if st.button("코멘트 추가", key=f"add_{cert['name']}"):
                        st.session_state.comments[cert['name']].append(new_comment)
                        st.success("코멘트가 추가되었습니다.")
                        st.rerun()

            # 비교 테이블
            st.subheader("자격증 간단 비교")
            comparison_table = recommendations[['name', 'type', 'difficulty', 'popularity', 'graduation_requirement']].copy()
            comparison_table['difficulty'] = comparison_table['difficulty'].apply(lambda x: '🌟' * int(x))
            comparison_table['popularity'] = comparison_table['popularity'].apply(lambda x: '🔥' * int(x))
            st.table(comparison_table.set_index('name'))

with tab2:
    st.header(f"👨‍🎓 {department} {major} 재학생/졸업생 취득 자격증")
    alumni_certs = get_alumni_certificates(department, major)
    
    # Plotly를 사용한 가로 막대 차트
    fig = go.Figure(go.Bar(
        x=alumni_certs['count'],
        y=alumni_certs['name'],
        orientation='h',
        marker_color='skyblue',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.6
    ))
    fig.update_layout(
        title='재학생/졸업생 자격증 취득 현황',
        xaxis_title='취득 인원',
        yaxis_title='자격증명',
        height=400,
        width=700
    )
    st.plotly_chart(fig)
    
    # 테이블로 상세 정보 표시
    st.table(alumni_certs)
    
    st.info("""
    - 이 데이터는 최근 5년간의 취득 현황을 바탕으로 합니다.
    - 실제 취득 현황은 변동될 수 있으며, 개인의 관심사와 진로 계획에 따라 선택하는 자격증이 다를 수 있습니다.
    - 자세한 정보는 학과 사무실이나 취업지원센터에 문의해주세요.
    """)
    
with tab3:
    st.session_state.current_tab = 'IPP 인턴십 공고'
    st.header("🏢 IPP 인턴십 공고")
    
    if st.session_state.get('search_internships', False):
        ipp_data = load_ipp_data()
        
        # 학과 필터링
        ipp_data = ipp_data[ipp_data['관련학과'].apply(lambda x: department in x)]
        
        if ipp_data.empty:
            st.warning(f"{department} 관련 IPP 인턴십 공고가 현재 없습니다.")
        else:
            # 기간 분류
            ipp_data['기간_정수'] = ipp_data['기간'].apply(parse_duration)
            ipp_data['기간_분류'] = ipp_data['기간_정수'].apply(classify_duration)
            
            # 데이터 필터링
            filtered_data = ipp_data[ipp_data['기간_분류'].isin(selected_duration)]
            if selected_fields:
                filtered_data = filtered_data[filtered_data['분야'].isin(selected_fields)]
            
            # GPA 필터링 (예시 - 실제 데이터에 맞게 수정 필요)
            filtered_data = filtered_data[filtered_data['우대조건'].apply(lambda x: any(f"학점 {min_gpa} 이상" in cond for cond in x))]
            
            # 인턴십 공고 표시 함수
            def display_internships(data, duration_type):
                st.subheader(f"📅 {duration_type} 인턴십")
                if not data.empty:
                    for _, ipp in data.iterrows():
                        with st.expander(f"{ipp['기업명']} - {ipp['분야']} ({ipp['기간']})"):
                            st.write(f"**지원자격:** {ipp['지원자격']}")
                            st.write(f"**마감일:** {ipp['마감일']}")
                            st.write(f"**관련학과:** {', '.join(ipp['관련학과'])}")
                            st.write("**우대조건:**")
                            for condition in ipp['우대조건']:
                                st.write(f"- {condition}")
                            if st.button("지원하기", key=f"apply_{duration_type}_{ipp['기업명']}"):
                                st.success(f"{ipp['기업명']}에 지원서가 제출되었습니다!")
                else:
                    st.info(f"현재 조건에 맞는 {duration_type} 인턴십 공고가 없습니다.")
            
            # 단기 인턴십 표시
            if "단기 (1~4개월)" in selected_duration:
                display_internships(filtered_data[filtered_data['기간_분류'] == "단기 (1~4개월)"], "단기")
            
            # 장기 인턴십 표시
            if "장기 (6개월~1년)" in selected_duration:
                display_internships(filtered_data[filtered_data['기간_분류'] == "장기 (6개월~1년)"], "장기")
    else:
        st.info("좌측 사이드바에서 검색 옵션을 선택하고 '인턴십 검색' 버튼을 클릭하세요.")

    st.info("""
    - IPP 인턴십은 학교와 기업이 공동으로 운영하는 장기현장실습 프로그램입니다.
    - 실제 근무 경험을 통해 실무 능력을 향상시킬 수 있는 좋은 기회입니다.
    - 지원 전 반드시 학과사무실이나 취업지원센터에 문의하여 학점 인정 등의 세부사항을 확인하세요.
    - 기업별로 세부 요구사항이 다를 수 있으니, 지원 전 꼼꼼히 확인하시기 바랍니다.
    - 우대조건을 충족하면 지원 시 가산점을 받을 수 있습니다. 하지만 필수 조건은 아니니 자신감을 가지고 도전해보세요!
    """)
    
# 추가 정보 섹션
st.header("추가 정보")
st.info("""
- 각 자격증의 이름을 클릭하면 상세 정보를 볼 수 있습니다.
- 자격증 종류는 국가기술자격, 국가전문자격, 국가공인민간자격, 민간자격으로 구분됩니다.
- 난이도는 1-5 스케일(🌟)로 표시됩니다.
- 인기도는 1-5 스케일(🔥)로 표시됩니다.
- 졸업요건은 O/X로 표시됩니다.
- 시험 일정과 응시료는 변경될 수 있으니 반드시 공식 웹사이트에서 최신 정보를 확인하세요.
- 취득한 자격증은 추천 목록에서 제외됩니다.
- 각 자격증에 대한 코멘트를 남겨 다른 사용자들과 정보를 공유할 수 있습니다.
""")

# 사용자 피드백
st.header("💬 전체 피드백")
feedback = st.text_area("추천 시스템에 대한 의견을 남겨주세요:")
if st.button("제출"):
    st.success("피드백이 제출되었습니다. 감사합니다!")

# 푸터
st.markdown("---")
st.markdown("© 2024 자격증 추천 대시보드 | 개발: 자격증뭐따조")

# 개발자 노트
st.sidebar.markdown("---")
st.sidebar.subheader("개발자 노트")
st.sidebar.info("""
이 앱은 데모 버전입니다. 실제 구현 시 고려해야 할 사항:
1. 실제 자격증 데이터베이스 연동 및 정기적인 업데이트
2. 사용자 피드백 및 코멘트 저장을 위한 데이터베이스 구축
3. 추천 알고리즘 지속적 개선
4. 보안 및 개인정보 보호 강화
5. 자격증 정보의 실시간 업데이트 시스템 구축
6. 사용자 인증 시스템 구현
""")
