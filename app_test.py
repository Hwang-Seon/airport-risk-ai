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
# 2. LLM 분류 과정 (데모 버전으로, mock 입력)
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
# 3. 인코딩 함수 정의
# =========================
def safe_transform(col, value):
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return 0  # unseen fallback

# =========================
# 4. 입력
# =========================
text = st.text_area(
    "사고 설명 입력 (※ 데모 버전으로, 예시 창에 입력된 값으로 실행하여 주십시오.)",
    value="폭설로 미끄러워진 GSE도로에서 램프버스가 정지하지 못하고 앞서가던 트럭을 추돌"
)

# =========================
# 6. 실행
# =========================
if st.button("분석 실행"):

    # -------------------------
    # (1) LLM 구조화
    # -------------------------
    llm_result = mock_llm(text)

    st.subheader("🔎 LLM 기반 사고 유형 및 원인 분류 결과")

    #st.json(llm_result)
    llm_df = pd.DataFrame(list(llm_result.items()), columns=["항목", "값"])
    st.table(llm_df)

    # -------------------------
    # (2) feature engineering
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
    # (3) encoding 
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
    # (5) 위험도 분석 결과 출력
    # -------------------------

    
    st.subheader("📊 위험도 예측 결과")

    st.markdown(f"## {label_map[pred_class]}")
    
    st.metric("Confidence", f"{np.max(proba):.2%}")
    
    st.write("### 위험도 클래스별 확률")
    
    df = pd.DataFrame({
        "위험도": ["낮은 위험", "중간 위험", "높은 위험"],
        "확률": [f"{proba[0]:.2%}", f"{proba[1]:.2%}", f"{proba[2]:.2%}"]
    })
    
    st.table(df)

    st.markdown("""
    ℹ️ **위험도 정의**
    
    - 🟢 **낮은위험** : 단순 절차 위반, 단순 접촉, 장비 오작동 등  
    - 🟠 **중간위험** : 장비 파손, 시설 파손, 경미한 인명 부상, 항공기 근접 사고 등  
    - 🔴 **높은위험** : 인명 피해, 항공기 직접 피해 등  
    """)

    # -------------------------
    # (6) 유사 사례 추출
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

    
    # -------------------------
    # (7) 추가 정보
    # -------------------------
    st.write("---")

    st.info(
        "LLM(GPT5.4)이 사고 정보를 구조화하고, "
        "XGBoost 기반 학습 모델로 위험도를 예측합니다."
    )
