import streamlit as st
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
import pytz
import altair as alt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. 기본 설정
# =========================
st.set_page_config(page_title="사고 위험도 분석", layout="wide")

xgb_model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

# 🔥 SBERT 로드
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = np.load("new_embeddings.npy")
data = pd.read_csv("final_data_for_embedding.csv").fillna("없음")

st.title("✈️ AI 기반 공항 지상조업 사고 리스크 분석 시스템")

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

# 🔥 유사도 함수 (TOP3)
def find_similar_cases(equip, task, location, weather):

    query = f"[장비: {equip}] [작업: {task}] [장소: {location}] [날씨: {weather}]"

    query_vec = sbert_model.encode([query])
    sims = cosine_similarity(query_vec, sentence_embeddings)[0]

    top_idx = np.argsort(sims)[::-1][:3]

    result = data.iloc[top_idx].copy()
    result["Final_Score"] = sims[top_idx]

    return result

# =========================
# 6. 실행
# =========================

if run:

    features = {
        "equip": equip,
        "equip_cat": get_slim_category(equip),
        "task": task,
        "location": location,
        "time": time_value,
        "weather": weather if weather != "없음" else "기타"
    }

    st.subheader("📋 입력된 작업 상황")

    st.dataframe(pd.DataFrame({
        "항목": ["장비", "장비 카테고리", "작업 상태", "위치", "시간", "날씨"],
        "값": list(features.values())
    }), use_container_width=True, hide_index=True)

    X = np.array([[
        safe_transform("equip", features["equip"]),
        safe_transform("equip_cat", features["equip_cat"]),
        safe_transform("task", features["task"]),
        safe_transform("location", features["location"]),
        safe_transform("time", features["time"]),
        safe_transform("weather", features["weather"]),
    ]])

    proba = xgb_model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))

    label_map = {
        0: "🟢 낮은 위험",
        1: "🟠 중간 위험",
        2: "🔴 높은 위험"
    }

    tab1, tab2 = st.tabs(["📊 사고 위험도 예측 결과", "🔎 유사 사고 사례 보기"])

    # -------------------------
    # TAB 1
    # -------------------------
    with tab1:

        col1, col2 = st.columns([1, 2])

        with col1:
            if pred_class == 0:
                st.success(label_map[pred_class])
            elif pred_class == 1:
                st.warning(label_map[pred_class])
            else:
                st.error(label_map[pred_class])

            st.markdown(f"**Confidence:** {np.max(proba):.2%}")

        with col2:
            chart_df = pd.DataFrame({
                "위험도": ["낮은위험", "중간위험", "높은위험"],
                "확률": proba
            })

            color_scale = alt.Scale(
                domain=["낮은위험", "중간위험", "높은위험"],
                range=["#4CAF50", "#FF9800", "#F44336"]
            )

            bars = alt.Chart(chart_df).mark_bar(size=20).encode(
                y="위험도:N",
                x=alt.X("확률:Q", axis=alt.Axis(format="%")),
                color=alt.Color("위험도:N", scale=color_scale, legend=None)
            ).properties(height=220)

            st.altair_chart(bars, use_container_width=True)

    # -------------------------
    # TAB 2 (🔥 실제 유사도)
    # -------------------------
    with tab2:

        st.subheader("🔎 유사 사고 사례 (TOP 3)")

        sim_df = find_similar_cases(
            features["equip"],
            features["task"],
            features["location"],
            features["weather"]
        )

        sim_df = sim_df.sort_values(by="Final_Score", ascending=False).reset_index(drop=True)
        sim_df["유사도 순위"] = sim_df.index + 1

        display_df = sim_df[[
            "유사도 순위",
            "corrected_text",
            "equip",
            "task",
            "risk"
        ]].rename(columns={
            "corrected_text": "사고 내용",
            "equip": "장비",
            "task": "작업",
            "risk": "위험도"
        })

        st.dataframe(display_df, use_container_width=True, hide_index=True)
