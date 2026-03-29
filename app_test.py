import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =========================
# 1. 모델 / 인코더 로드
# =========================
model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="사고 위험도 분석", layout="centered")
st.title("✈️ AI 기반 공항 지상조업 사고 리스크 분석 시스템")

# =========================
# 2. LLM (현재는 mock)
# =========================
def mock_llm(text):
    return {
        "category": "차량-차량",
        "equip_1": "버스",
        "equip_2": "트럭",
        "status": "주행",
        "cause": "기상악화"
    }

# =========================
# 3. feature engineering (학습과 동일해야 함)
# =========================
'''
def get_vehicle_cat(equip):
    mapping = {
        "버스": "운송수송차량",
        "트럭": "운송수송차량",
        "승용차": "운송수송차량",
        "택시": "운송수송차량",
        "지게차": "산업장비",
        "기타": "기타"
    }
    return mapping.get(equip, "기타")
'''
# =========================
# 4. 안전 인코딩
# =========================
def safe_transform(col, value):
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return 0  # unseen fallback

# =========================
# 5. 입력 UI
# =========================
text = st.text_area(
    "사고 설명 입력",
    value="폭설로 미끄러워진 GSE도로에서 램프버스가 정지하지 못하고 앞서가던 트럭을 추돌"
)

# =========================
# 6. 실행 버튼
# =========================
if st.button("분석 실행"):

    # -------------------------
    # (1) LLM 구조화
    # -------------------------
    llm_result = mock_llm(text)

    st.subheader("🔎 LLM 기반 사고 유형 및 원인 분류 결과")

    st.json(llm_result)

    # -------------------------
    # (2) feature engineering (핵심)
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

    # -------------------------
    # (3) encoding (학습과 동일 순서 중요)
    # -------------------------
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
    # (4) prediction
    # -------------------------
    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))

    label_map = {
        0: "🟢 낮은 위험",
        1: "🟠 중간 위험",
        2: "🔴 높은 위험"
    }

    # -------------------------
    # (5) 결과 출력
    # -------------------------
    st.subheader("📊 위험도 예측 결과")

    st.markdown(f"## {label_map[pred_class]}")

    st.metric("Confidence", f"{np.max(proba):.2%}")

    st.write("### 위험도 클래스별 확률")

    st.bar_chart({
        "낮은위험": [proba[0]],
        "중간위험": [proba[1]],
        "높은위험": [proba[2]]
    })

    st.info(
        "낮은위험 : 단순 절차 위반, 단순 접촉, 장비 오작동 등 "
        "중간위험 : 장비 파손, 시설 파손, 경미한 인명 부상, 항공기 근접 사고 등 "
        "높은위험 : 인명 피해, 항공기 직접 피해 등")

    # -------------------------
    # (6) 디버깅 정보 (중요)
    # -------------------------
    st.write("---")

    st.info(
        "LLM(GPT5.4)이 사고 정보를 구조화하고, "
        "XGBoost 기반 학습 모델로 위험도를 예측합니다."
    )
