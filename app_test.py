import streamlit as st
import joblib
import numpy as np

# =========================
# 1. 모델 & encoder 로드
# =========================
model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="항공 사고 위험도 분석", layout="centered")

st.title("✈️ 항공 사고 위험도 분석 시스템 (XGBoost)")

# =========================
# 2. 입력 UI (학습 데이터 기반)
# =========================
st.subheader("사고 정보 입력")

category = st.selectbox("category", encoders["category"].classes_)
equip_1 = st.selectbox("equip_1", encoders["equip_1"].classes_)
equip_2 = st.selectbox("equip_2", encoders["equip_2"].classes_)
equip_1_cat = st.selectbox("equip_1_cat", encoders["equip_1_cat"].classes_)
equip_2_cat = st.selectbox("equip_2_cat", encoders["equip_2_cat"].classes_)
status = st.selectbox("status", encoders["status"].classes_)
cause = st.selectbox("cause", encoders["cause"].classes_)

# =========================
# 3. 위험도 정의 함수
# =========================
def get_risk_label(pred_class):
    if pred_class == 0:
        return "🟢 낮은 위험"
    elif pred_class == 1:
        return "🟠 중간 위험"
    else:
        return "🔴 높은 위험"

# =========================
# 4. 예측 실행
# =========================
if st.button("위험도 분석 실행"):

    # ---- encoding ----
    X = np.array([[
        encoders["category"].transform([category])[0],
        encoders["equip_1"].transform([equip_1])[0],
        encoders["equip_2"].transform([equip_2])[0],
        encoders["equip_1_cat"].transform([equip_1_cat])[0],
        encoders["equip_2_cat"].transform([equip_2_cat])[0],
        encoders["status"].transform([status])[0],
        encoders["cause"].transform([cause])[0],
    ]])

    # ---- prediction ----
    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(np.max(proba))
    risk_label = get_risk_label(pred_class)

    # =========================
    # 5. 결과 출력
    # =========================
    st.subheader("📊 분석 결과")

    st.markdown(f"## {risk_label}")

    st.metric("신뢰도 (Confidence)", f"{confidence:.2%}")

    st.write("### 클래스별 확률 (Low / Medium / High)")

    st.bar_chart({
        "Low": [proba[0]],
        "Medium": [proba[1]],
        "High": [proba[2]]
    })

    st.write("---")
    st.write(f"예측 클래스: {pred_class}")

    # =========================
    # 6. 설명 (심사용 포인트)
    # =========================
    st.info(
        "본 모델은 XGBoost 기반 다중 분류 모델로, "
        "사고 위험도를 Low / Medium / High 3단계로 분류합니다. "
        "출력 확률을 기반으로 위험도를 해석합니다."
    )
