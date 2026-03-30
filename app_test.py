import streamlit as st
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
import pytz
import altair as alt

# =========================
# 1. 기본 설정
# =========================
st.set_page_config(page_title="사고 위험도 분석", layout="wide")

model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("✈️ AI 기반 공항 지상조업 사고 위험도 사전 예측 시스템")

st.warning("""
⚠️ 현재 작업 상태 또는 작업 계획을 입력하세요.
""")

# =========================
# 2. 카테고리 정의
# =========================

equip_list = [
    '터그','토잉','로더','스텝','제방빙','푸쉬백','리프트','고소','급유','지게차',
    '버스','차량','승용','승합','트럭','탑차','달리','사다리','작업대','스탠드',
    'PDU','장비','항공기','기타'
]

task_list = ["주행", "접근", "작업중", "후진", "주차", "보행", "기타"]

location_list = [
    "계류장", "여객터미널", "화물터미널", "도로", "정치장",
    "주차장", "정비고", "탑승동", "탑승교", "주기장",
    "활주로", "동력동", "기타 시설"
]

weather_list = ["없음", "눈", "비", "강풍", "안개"]

# =========================
# 3. 시간 자동 설정
# =========================
kst = pytz.timezone("Asia/Seoul")
hour = datetime.now(kst).hour
time_value = "주간" if 6 <= hour < 18 else "야간"

# =========================
# 4. 입력 UI
# =========================
st.markdown("### 🔧 작업 상황 입력")

col1, col2, col3 = st.columns(3)

with col1:
    equip = st.selectbox("장비 (주체)", equip_list)

with col2:
    task = st.selectbox("작업 상태", task_list)

with col3:
    location = st.selectbox("위치", location_list)

col4, col5 = st.columns(2)

with col4:
    st.text_input("시간", value=time_value, disabled=True)

with col5:
    weather = st.selectbox("위험 기상", weather_list)

run = st.button("🚀 분석 실행")

# =========================
# 5. 함수
# =========================

def safe_transform(col, value):
    le = encoders[col]
    return le.transform([value])[0] if value in le.classes_ else 0

def get_slim_category(name):
    if not name or name == '없음':
        return '기타/미분류'

    if '항공기' in name:
        return '항공기'

    elif any(kw in name for kw in ['터그', '토잉', '로더', '스텝', '제방빙', '푸쉬백', '리프트', '고소', '급유', '지게차']):
        return '조업특수장비'

    elif any(kw in name for kw in ['버스', '차량', '승용', '승합', '트럭', '탑차', '스타렉스', '모닝', '마티즈', '점검']):
        return '운송수송차량'

    elif any(kw in name for kw in ['달리', '사다리', '작업대', '스탠드', 'PDU', '장비']):
        return '조업보조도구'

    elif any(kw in name for kw in ['시설', '벽면', '천정', '울타리', '문', '탑승교', '경계석', '배관']):
        return '시설물'

    elif any(kw in name for kw in ['작업자', '보행자', '승객']):
        return '인적요소'

    else:
        return '기타/미분류'

# =========================
# 6. 실행
# =========================

if run:

    # -------------------------
    # (1) Feature 구성
    # -------------------------
    features = {
        "equip": equip,
        "equip_cat": get_slim_category(equip),
        "task": task,
        "location": location,
        "time": time_value,
        "weather": weather if weather != "없음" else "기타"
    }

    # -------------------------
    # (2) 입력 확인 UI
    # -------------------------
    st.subheader("📋 입력된 작업 상황")

    feature_df = pd.DataFrame({
        "항목": ["장비", "장비 카테고리", "작업 상태", "위치", "시간", "날씨"],
        "값": list(features.values())
    })

    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    # -------------------------
    # (3) 모델 입력
    # -------------------------
    X = np.array([[
        safe_transform("equip", features["equip"]),
        safe_transform("equip_cat", features["equip_cat"]),
        safe_transform("task", features["task"]),
        safe_transform("location", features["location"]),
        safe_transform("time", features["time"]),
        safe_transform("weather", features["weather"]),
    ]])

    # -------------------------
    # (4) 예측
    # -------------------------
    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))

    label_map = {
        0: "🟢 낮은 위험",
        1: "🟠 중간 위험",
        2: "🔴 높은 위험"
    }

    # -------------------------
    # (5) 결과 탭 구성
    # -------------------------
    tab1, tab2 = st.tabs(["📊 사고 위험도 예측 결과", "🔎 유사 사고 사례 보기"])
    
    # -------------------------
    # TAB 1: 위험도 결과
    # -------------------------
    with tab1:
    
        st.subheader("📊 사고 위험도 예측 결과")
    
        col1, col2 = st.columns([1, 2])
    
        # 왼쪽: 최종 위험도
        with col1:
            st.markdown("### 🎯 최종 위험도")
    
            if pred_class == 0:
                st.success(label_map[pred_class])
            elif pred_class == 1:
                st.warning(label_map[pred_class])
            else:
                st.error(label_map[pred_class])
    
            st.markdown(f"""
            <div style="font-size:14px; color:gray;">Confidence</div>
            <div style="font-size:20px; font-weight:bold;">
                {np.max(proba):.2%}
            </div>
            """, unsafe_allow_html=True)
    
        # 오른쪽: 확률 분포
        with col2:
    
            chart_df = pd.DataFrame({
                "위험도": ["낮은위험", "중간위험", "높은위험"],
                "확률": proba
            })
    
            chart = (
                alt.Chart(chart_df)
                .mark_bar(size=25)
                .encode(
                    y=alt.Y("위험도:N", sort=["낮은위험", "중간위험", "높은위험"]),
                    x=alt.X("확률:Q", axis=alt.Axis(format="%")),
                    color=alt.Color("위험도:N", legend=None)
                )
            )
    
            st.altair_chart(chart, use_container_width=True)
    
        # 설명
        st.info(""" 
    ℹ️ 위험도 기준
    
    - 🟢 낮은위험: 경미한 상황  
    - 🟠 중간위험: 장비 손상 가능  
    - 🔴 높은위험: 인명/항공기 위험  
    """)
    
    # -------------------------
    # TAB 2: 유사 사고 사례
    # -------------------------
    
    def mock_similar_cases(): 
        return pd.DataFrame([
            {
                "Final_Score": 0.8009,
                "Cos_Sim_Score": 0.7009,
                "Equip_Match": "YES",
                "Previous_Accident": "램프버스가 후진 중 운전부주의로 차량 대기장소에서 급유를 위해 정차중이던 급유차량 추돌",
                "Equip_Cats": "운송수송차량 / 조업특수장비",
                "Severity": 3
            },
            {
                "Final_Score": 0.7697,
                "Cos_Sim_Score": 0.6697,
                "Equip_Match": "YES",
                "Previous_Accident": "승객을 수송중이던 램프버스가 교차로 운행 중 반대편 차량에 정차중인 차량 추돌",
                "Equip_Cats": "운송수송차량 / 운송수송차량",
                "Severity": 3
            },
            {
                "Final_Score": 0.7657,
                "Cos_Sim_Score": 0.6657,
                "Equip_Match": "YES",
                "Previous_Accident": "주행 중이던 램프버스와 화물 적재 후 조업도로에 진입 중이던 터그가 간 충돌",
                "Equip_Cats": "운송수송차량 / 조업특수장비",
                "Severity": 3
            }
        ])
    
    with tab2:
    
        st.subheader("🔎 유사 사고 사례 보기")
    
        # (1) 데이터
        similar_df = mock_similar_cases()
    
        # (2) 정렬
        similar_df = similar_df.sort_values(by="Final_Score", ascending=False).reset_index(drop=True)
    
        # (3) 순위
        similar_df["유사도 순위"] = similar_df.index + 1
    
        # (4) 표시용
        display_df = similar_df[["유사도 순위", "Previous_Accident", "Equip_Cats"]].rename(columns={
            "Previous_Accident": "사고 내용",
            "Equip_Cats": "관련 장비 카테고리"
        })
    
        # (5) 출력
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
