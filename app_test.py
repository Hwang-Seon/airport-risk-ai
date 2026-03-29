import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# 1. 모델 & encoder 로드
# =========================
model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="항공 사고 분석 시스템", layout="centered")

st.title("✈️ 항공 사고 위험도 분석 시스템")

# =========================
# 2. 입력 (자연어 사고)
# =========================
st.subheader("사고 입력")

text = st.text_area(
    "사고 내용을 입력하세요",
    value="폭설로 미끄러워진 GSE도로에서 램프버스가 정지하지 못하고 앞서가던 트럭을 추돌함"
)

# =========================
# 3. LLM 결과 (여기서는 MOCK)
#    → 실제로는 GPT API 자리
# =========================
def mock_llm(text):

    return {
        "category": "차량-차량",
        "equip_1": "버스",
        "equip_2": "트럭",
        "status": "주행",
        "cause": "기상악화",
        "S": 0,
        "H": 0,
        "E": 1,
        "L": 1,
        "severity_score": 3
    }

# =========================
# 4. 위험도 라벨
# =========================
def get_risk_label(pred_class):
    if pred_class == 0:
        return "🟢 낮은 위험"
    elif pred_class == 1:
        return "🟠 중간 위험"
    else:
        return "🔴 높은 위험"

# =========================
# 5. 실행
# =========================
if st.button("분석 실행"):

    # -------------------------
    # (1) LLM 결과 생성
    # -------------------------
    llm_result = mock_llm(text)

    st.subheader("🧠 LLM 사고 구조화 결과")

    df_llm = pd.DataFrame([llm_result])
    st.dataframe(df_llm)

    # -------------------------
    # (2) ML 입력 변환
    # -------------------------
    X = np.array([[
        encoders["category"].transform([llm_result["category"]])[0],
        encoders["equip_1"].transform([llm_result["equip_1"]])[0],
        encoders["equip_2"].transform([llm_result["equip_2"]])[0],
        encoders["equip_1_cat"].transform([llm_result["equip_1"]])[0],  # 필요시 수정
        encoders["equip_2_cat"].transform([llm_result["equip_2"]])[0],  # 필요시 수정
        encoders["status"].transform([llm_result["status"]])[0],
        encoders["cause"].transform([llm_result["cause"]])[0],
    ]])

    # -------------------------
    # (3) 모델 예측
    # -------------------------
    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(np.max(proba))
    risk_label = get_risk_label(pred_class)

    # -------------------------
    # (4) 결과 출력
    # -------------------------
    st.subheader("📊 위험도 분석 결과")

    st.markdown(f"## {risk_label}")

    st.metric("신뢰도", f"{confidence:.2%}")

    st.write("### 클래스별 확률 (Low / Medium / High)")
    st.bar_chart({
        "Low": [proba[0]],
        "Medium": [proba[1]],
        "High": [proba[2]]
    })

    st.write(f"예측 클래스: {pred_class}")

    # -------------------------
    # (5) 설명
    # -------------------------
    st.info(
        "LLM이 자연어 사고를 구조화한 결과를 기반으로 "
        "XGBoost 모델이 위험도를 3단계로 분류합니다."
    )
