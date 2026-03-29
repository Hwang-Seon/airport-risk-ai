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
st.title("✈️ 사고 입력")

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
st.title("✈️ AI 기반 공항 지상조업 사고 리스크 분석 시스템")

if run:

    # -------------------------
    # (1) LLM 결과
    # -------------------------
    llm_result = mock_llm(text)

    st.subheader("🔎 사고 구조 분석")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("사고 유형", llm_result["category"])
    col2.metric("장비1", llm_result["equip_1"])
    col3.metric("장비2", llm_result["equip_2"])
    col4.metric("상태", llm_result["status"])
    col5.metric("원인", llm_result["cause"])

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
    tab1, tab2, tab3 = st.tabs(["📊 위험도", "📈 확률 분포", "📚 유사 사례"])

    # -------------------------
    # TAB 1: 위험도
    # -------------------------
    with tab1:
        st.subheader("📊 최종 위험도")

        st.markdown(f"# {label_map[pred_class]}")
        st.metric("Confidence", f"{np.max(proba):.2%}")

    # -------------------------
    # TAB 2: 확률 시각화
    # -------------------------
    with tab2:
        st.subheader("📈 클래스별 확률")

        chart_df = pd.DataFrame({
            "위험도": ["낮은", "중간", "높은"],
            "확률": proba
        })

        st.bar_chart(chart_df.set_index("위험도"))

    # -------------------------
    # TAB 3: 유사 사례
    # -------------------------
    with tab3:
        st.subheader("📚 유사 사고 사례")

        df = pd.DataFrame([
            {"사고": "램프버스 후진 중 급유차 추돌", "위험도": "높음"},
            {"사고": "교차로 정차 차량 추돌", "위험도": "높음"},
            {"사고": "버스-터그 충돌", "위험도": "높음"}
        ])

        st.dataframe(df, use_container_width=True)

    # -------------------------
    # 설명
    # -------------------------
    with st.expander("ℹ️ 위험도 기준 설명"):
        st.write("""
        - 🟢 낮은위험: 단순 접촉, 경미 사고  
        - 🟠 중간위험: 장비/시설 손상  
        - 🔴 높은위험: 인명 피해 또는 항공기 손상  
        """)
