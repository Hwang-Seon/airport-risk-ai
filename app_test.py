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
# 모드 설정
# =========================

st.set_page_config(page_title="ICN 산업재해 관리 시스템 Ⓒ 2026 순두부", layout="wide")

st.caption("Developed by Team 순두부 | Incheon Airport AI Project")
st.markdown("---")

if "mode" not in st.session_state:
    st.session_state.mode = None

def go_home(): # 뒤로가기함수
    if st.button("⬅️ 처음으로 돌아가기"):
        st.session_state.mode = None
        st.rerun()

if st.session_state.mode is None:

    st.markdown("# ✈️ 공항 지상조업 AI 시스템")
    st.markdown("### 원하는 기능을 선택하세요\n")

    col1, col2 = st.columns(2)

    # -------------------------
    # 카드 1: 위험도 분석
    # -------------------------
    with col1:
        st.markdown("""
        ### 📊 작업 위험도 분석하기
        - 작업 상황 위험도 예측  
        - 유사 사고 사례 보기  
        """)

        if st.button("👉 분석 시작", use_container_width=True):
            st.session_state.mode = "analysis"
            st.rerun()

    # -------------------------
    # 카드 2: DB 추가
    # -------------------------
    with col2:
        st.markdown("""
        ### 📝 사고 발생 DB 추가하기
        - 사고 상황 분류  
        - DB 데이터 자동 추가  
        """)

        if st.button("👉 등록 시작", use_container_width=True):
            st.session_state.mode = "db"
            st.rerun()

    st.stop()

# =========================
# A. 작업 위험도 분석 모드
# =========================

if st.session_state.mode == "analysis":
    go_home()
    st.markdown("## 📊 작업 위험도 분석")

    # =========================
    # 1. 기본 설정
    # =========================
    
    # 유사도 임베딩 데이터/모델
    sim_data = pd.read_csv("final_data_for_embedding.csv").fillna("없음")
    sim_embeddings = np.load("new_embeddings.npy")
    
    @st.cache_resource
    def load_sbert():
        return SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    sbert_model = load_sbert()
    
    st.warning("""
    ⚠️ 현재 작업 상태 또는 작업 계획을 입력하세요.
    """)
    
    # ML 모델/엔코더
    xgb_model = joblib.load("xgb_model.pkl")
    encoders = joblib.load("encoders.pkl")
    
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
        equip = st.selectbox("작업 장비 (주체)", equip_list)
    
    with col2:
        task = st.selectbox("작업 상태", task_list)
    
    with col3:
        location = st.selectbox("작업 위치", location_list)
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.text_input("시간 (주/야간 자동설정 됩니다.)", value=time_value, disabled=True)
    
    with col5:
        weather = st.selectbox("위험 기상", weather_list)
    
    run = st.button("🚀 분석 실행")
    
    # =========================
    # 5. 함수
    # =========================
    
    # 엔코더 함수
    def safe_transform(col, value):
        le = encoders[col]
        return le.transform([value])[0] if value in le.classes_ else 0
    
    # 장비 분류 함수
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
    
    # 유사 사례 탐색 함수
    def find_similar_cases(equip, task, location, weather):
    
        query = f"[장비: {equip}] [작업: {task}] [장소: {location}] [날씨: {weather}]"
        query_vec = sbert_model.encode([query])
    
        if len(query_vec.shape) == 1:
            query_vec = query_vec.reshape(1, -1)
    
        sims = cosine_similarity(query_vec, sim_embeddings)[0]
    
        top_idx = np.argsort(sims)[::-1][:3]
    
        result = sim_data.iloc[top_idx].copy()
        result["Final_Score"] = sims[top_idx]
    
        return result
    
    
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
        proba = xgb_model.predict_proba(X)[0]
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
            
                # 🔥 색상 매핑
                color_scale = alt.Scale(
                    domain=["낮은위험", "중간위험", "높은위험"],
                    range=["#4CAF50", "#FF9800", "#F44336"]  # 초록 / 주황 / 빨강
                )
            
                # 막대
                bars = (
                    alt.Chart(chart_df)
                    .mark_bar(size=20, cornerRadius=6)
                    .encode(
                        y=alt.Y(
                            "위험도:N",
                            sort=["낮은위험", "중간위험", "높은위험"],
                            title=""
                        ),
                        x=alt.X(
                            "확률:Q",
                            title="확률",
                            axis=alt.Axis(format="%")
                        ),
                        color=alt.Color("위험도:N", scale=color_scale, legend=None)
                    )
                    .properties(height=220)
                )
            
                # 텍스트 (확률 표시)
                text = (
                    alt.Chart(chart_df)
                    .mark_text(
                        align="left",
                        baseline="middle",
                        dx=5,
                        fontSize=13,
                        fontWeight="bold"
                    )
                    .encode(
                        y=alt.Y("위험도:N", sort=["낮은위험", "중간위험", "높은위험"]),
                        x="확률:Q",
                        text=alt.Text("확률:Q", format=".1%")
                    )
                )
            
                st.altair_chart(bars + text, use_container_width=True)
            
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
        
            display_df["유사도 점수"] = sim_df["Final_Score"].round(3)
            display_df["위험도"] = display_df["위험도"].map({0: "🟢 낮음", 1: "🟠 중간", 2: "🔴 높음"})
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# =========================
# A. 작업 위험도 분석 모드
# =========================

elif st.session_state.mode == "db":
    go_home()
    st.markdown("## 📝 사고 데이터 등록")

    st.error("""
    ⚠️ 데모 버전 안내 ⚠️
    현재 시스템은 실제 LLM이 아닌 예시 데이터로 동작합니다.  
    👉 입력창에 제공된 예시 문장으로만 정상 동작합니다.
    👉 다른 문장을 입력할 경우 정확한 분석이 수행되지 않을 수 있습니다.
    """)

    accident_text = st.text_area(
        "사고 상황 입력",
        value="폭설로 미끄러운 GES도로에서 램프버스가 후진 중 트럭과 충돌"
    )

    run_db = st.button("📥 DB 저장 실행")

    def mock_llm(text):
        return {
            "equip": "버스",
            "task": "후진",
            "location": "도로",
            "weather": "눈",
            "risk": 2
        }

    if run_db:

        llm_result = mock_llm(accident_text)

        kst = pytz.timezone("Asia/Seoul")
        hour = datetime.now(kst).hour
        time_value = "주간" if 6 <= hour < 18 else "야간"

        st.subheader("💡 LLM 기반 사고 구조 분류 결과")

        llm_df = pd.DataFrame({
            "항목": ["장비", "작업", "위치", "시간", "날씨", "위험도"],
            "값": [
                llm_result["equip"],
                llm_result["task"],
                llm_result["location"],
                time_value,
                llm_result["weather"],
                llm_result["risk"]
            ]
        })

        st.dataframe(llm_df, use_container_width=True, hide_index=True)

        st.success("✅ 사고 데이터가 DB에 저장되었습니다.")
