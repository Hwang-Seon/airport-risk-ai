import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =========================
# 1. 기본 설정
# =========================
st.set_page_config(page_title="사고 위험도 분석", layout="wide")

model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

# =========================
# 2. Sidebar (입력 영역)
# =========================
st.title("✈️ AI 기반 공항 지상조업 사고 리스크 분석 시스템")

text = st.text_area(
    "사고 설명 입력 (※ 데모 버전으로, 예시 창에 입력된 값으로 실행하여 주십시오.)",
    value="폭설로 미끄러워진 GSE도로에서 램프버스가 정지하지 못하고 앞서가던 트럭을 추돌"
)

run = st.button("🚀 분석 실행")

# =========================
# 3. 함수
# =========================
def mock_llm(text):
    return {
        "category": "차량-차량",
        "equip_1": "버스",
        "equip_2": "트럭",
        "status": "주행",
        "cause": "기상악화"
    }

def safe_transform(col, value):
    le = encoders[col]
    return le.transform([value])[0] if value in le.classes_ else 0


# =========================
# 4. 실행
# =========================

if run:

    # -------------------------
    # (1) LLM 결과
    # -------------------------
    llm_result = mock_llm(text)

    st.subheader("💡 LLM 기반 사고 구조 분석 결과")

    llm_df = pd.DataFrame({
    "항목": ["사고 유형", "장비1", "장비2", "상태", "원인"],
    "값": [
        llm_result["category"],
        llm_result["equip_1"],
        llm_result["equip_2"],
        llm_result["status"],
        llm_result["cause"]
    ]
    })
    
    st.dataframe(llm_df, use_container_width=True, hide_index=True)

    # -------------------------
    # (2) Feature Encoding
    # -------------------------
    features = {
        "category": llm_result["category"],
        "equip_1": llm_result["equip_1"],
        "equip_2": llm_result["equip_2"],
        "equip_1_cat": llm_result["equip_1"],
        "equip_2_cat": llm_result["equip_2"],
        "status": llm_result["status"],
        "cause": llm_result["cause"]
    }

    X = np.array([[
        safe_transform("category", features["category"]),
        safe_transform("equip_1", features["equip_1"]),
        safe_transform("equip_2", features["equip_2"]),
        safe_transform("equip_1_cat", features["equip_1_cat"]),
        safe_transform("equip_2_cat", features["equip_2_cat"]),
        safe_transform("status", features["status"]),
        safe_transform("cause", features["cause"]),
    ]])

    # -------------------------
    # (3) 예측
    # -------------------------
    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))

    label_map = {
        0: "🟢 낮은 위험",
        1: "🟠 중간 위험",
        2: "🔴 높은 위험"
    }

    # -------------------------
    # (4) 결과 탭 구성
    # -------------------------
    
    tab1, tab2 = st.tabs(["📊 사고 위험도 예측 결과", " 🔎 유사 사고 사례 보기"])

    # -------------------------
    # TAB 1: 위험도
    # -------------------------
    with tab1:
    
        st.subheader("📊 사고 위험도 예측 결과")
    
        col1, col2 = st.columns([1, 2])
    
        # -------------------------
        # (1) 왼쪽: 최종 위험도
        # -------------------------
        with col1:
            st.markdown("### 🎯 최종 위험도")
    
            if pred_class == 0:
                st.success(label_map[pred_class])
            elif pred_class == 1:
                st.warning(label_map[pred_class])
            else:
                st.error(label_map[pred_class])
    
            st.markdown(f"""
            <div style="text-align:left;">
                <div style="font-size:12px; color:gray;">Confidence</div>
                <div style="font-size:16px; font-weight:bold;">
                    {np.max(proba):.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
        # -------------------------
        # (2) 오른쪽: 확률 분포
        # -------------------------
        with col2:
            st.markdown("### 📈 위험도 클래스 확률 분포")
        
            import altair as alt
        
            chart_df = pd.DataFrame({
                "위험도": ["낮은위험", "중간위험", "높은위험"],
                "확률": proba
            })
        
            # 색상
            color_scale = alt.Scale(
                domain=["낮은위험", "중간위험", "높은위험"],
                range=["#4CAF50", "#FF9800", "#F44336"]
            )
        
            # 막대
            bars = (
                alt.Chart(chart_df)
                .mark_bar(size=25, cornerRadius=6)  # 👈 두께 줄여서 겹침 방지
                .encode(
                    y=alt.Y(
                        "위험도:N",
                        sort=["낮은위험", "중간위험", "높은위험"],
                        title="",
                        axis=alt.Axis(labelFontSize=14, labelFontWeight="bold")  # 👈 범주 강조
                    ),
                    x=alt.X(
                        "확률:Q",
                        title="확률",
                        axis=alt.Axis(format="%")
                    ),
                    color=alt.Color("위험도:N", scale=color_scale, legend=None)
                )
                .properties(height=200)  # 👈 전체 높이 늘려서 간격 확보
            )
        
            # 텍스트 (막대 안쪽)
            text = (
                alt.Chart(chart_df)
                .mark_text(
                    align="right",
                    baseline="middle",
                    dx=-5,
                    fontSize=14,
                    fontWeight="bold",
                    color="white"  # 👈 막대 안에서 잘 보이게
                )
                .encode(
                    y=alt.Y("위험도:N", sort=["낮은위험", "중간위험", "높은위험"]),
                    x="확률:Q",
                    text=alt.Text("확률:Q", format=".1%")
                )
            )
        
            st.altair_chart(bars + text, use_container_width=True)
    
    
        # -------------------------
        # (3) 설명
        # -------------------------
        with st.expander("ℹ️ 위험도 기준"):
            st.write("""
            - 🟢 낮은위험: 단순 접촉, 경미 사고  
            - 🟠 중간위험: 장비/시설 손상  
            - 🔴 높은위험: 인명 피해 또는 항공기 손상  
            """)

    # -------------------------
    # TAB 2: 유사 사고 사례 출력
    # -------------------------

    # 아래 데이터는 입력 예시 데이터를 실제 유사도 분류 프로그램에 적용시킨 결과를 가져온 것입니다. 
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
    
        # -------------------------
        # (1) 데이터 불러오기
        # -------------------------
        similar_df = mock_similar_cases()
    
        # -------------------------
        # (2) 유사도 기준 정렬
        # -------------------------
        similar_df = similar_df.sort_values(by="Final_Score", ascending=False).reset_index(drop=True)
    
        # -------------------------
        # (3) 순위 컬럼 생성
        # -------------------------
        similar_df["유사도 순위"] = similar_df.index + 1
    
        # -------------------------
        # (4) 필요한 정보 추출
        # -------------------------
        display_df = similar_df[["유사도 순위", "Previous_Accident", "Equip_Cats"]].rename(columns={
            "Previous_Accident": "사고 내용",
            "Equip_Cats": "관련 장비 카테고리"
        })
    
        # -------------------------
        # (5) 출력
        # -------------------------
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
