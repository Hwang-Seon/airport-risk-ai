import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# 1. 모델 & encoder 로드
# =========================
model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="항공 사고 위험도 분석", layout="centered")

st.title("✈️ 항공 사고 위험도 분석 시스템")

# =========================
# 2. 허용 vocabulary (핵심 안전장치)
# =========================
VOCAB = {
    col: list(encoders[col].classes_)
    for col in encoders.keys()
}

# =========================
# 3. LLM (Mock or API 대체)
#    → 반드시 "허용된 값만 출력"
# =========================
def mock_llm(text):

    # 실제 GPT로 바꿔도 반드시 이 구조 유지
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
# 4. 안전 변환 함수 (핵심)
# =========================
def safe_map(value, allowed_list):
    if value in allowed_list:
        return value
    else:
        return allowed_list[0]  # fallback (안전값)

# =========================
# 5. 위험도 정의
# =========================
def get_risk_label(pred_class):
    if pred_class == 0:
        return "🟢 낮은 위험"
    elif pred_class == 1:
        return "🟠 중간 위험"
    else:
        return "🔴 높은 위험"

# =========================
# 6. UI 입력
# =========================
text = st.text_area(
    "사고 내용을 입력하세요",
    value="폭설로 미끄러워진 GSE도로에서 램프버스가 정지하지 못하고 트럭을 추돌함"
)

# =========================
# 7. 실행
# =========================
if st.button("분석 실행"):

    # -------------------------
    # (1) LLM 사고 구조화
    # -------------------------
    llm_result = mock_llm(text)

    # 🔥 핵심: 안전한 label로 강제 정제
    llm_result["category"] = safe_map(llm_result["category"], VOCAB["category"])
    llm_result["equip_1"] = safe_map(llm_result["equip_1"], VOCAB["equip_1"])
    llm_result["equip_2"] = safe_map(llm_result["equip_2"], VOCAB["equip_2"])
    llm_result["status"] = safe_map(llm_result["status"], VOCAB["status"])
    llm_result["cause"] = safe_map(llm_result["cause"], VOCAB["cause"])

    # -------------------------
    # (2) LLM 결과 UI 출력
    # -------------------------
    st.subheader("🧠 LLM 사고 분류 결과 (Structured Output)")

    st.dataframe(pd.DataFrame([llm_result]))

    # -------------------------
    # (3) ML 입력 벡터 생성
    # -------------------------
    X = np.array([[
        encoders["category"].transform([llm_result["category"]])[0],
        encoders["equip_1"].transform([llm_result["equip_1"]])[0],
        encoders["equip_2"].transform([llm_result["equip_2"]])[0],
        encoders["equip_1_cat"].transform([llm_result["equip_1"]])[0],
        encoders["equip_2_cat"].transform([llm_result["equip_2"]])[0],
        encoders["status"].transform([llm_result["status"]])[0],
        encoders["cause"].transform([llm_result["cause"]])[0],
    ]])

    # -------------------------
    # (4) 모델 예측
    # -------------------------
    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(np.max(proba))

    risk_label = get_risk_label(pred_class)

    # -------------------------
    # (5) 결과 출력
    # -------------------------
    st.subheader("📊 위험도 분석 결과")

    st.markdown(f"## {risk_label}")

    st.metric("Confidence", f"{confidence:.2%}")

    st.write("### 클래스별 확률 (Low / Medium / High)")

    st.bar_chart({
        "Low": [proba[0]],
        "Medium": [proba[1]],
        "High": [proba[2]]
    })

    st.write(f"예측 클래스: {pred_class}")

    # -------------------------
    # (6) 설명 (심사용)
    # -------------------------
    st.info(
        "LLM은 사고를 구조화하는 역할만 수행하며, "
        "모든 출력은 학습된 taxonomy로 제한됩니다. "
        "이후 XGBoost 모델이 위험도를 3단계로 분류합니다."
    )
