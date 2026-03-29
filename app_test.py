import streamlit as st

st.title("공항 사고 분석 TEST")

text = st.text_area("사고 입력")

if st.button("분석"):
    st.write(text)
