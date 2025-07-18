import streamlit as st
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_admission(gre, toefl, university_rating, sop, lor, cgpa, research):
    # Manually set weights (from training phase)
    weights = np.array([0.0035, 0.01, 0.02, 0.12, 0.1, 0.6, 0.3])
    bias = -3.5

    x = np.array([gre, toefl, university_rating, sop, lor, cgpa, research])
    z = np.dot(weights, x) + bias
    prob = sigmoid(z)

    return 1 if prob >= 0.5 else 0, prob

st.set_page_config(page_title="Admission Predictor", page_icon="🎓")
st.title("🎓 Student Admission Predictor")
st.write("This simple app predicts whether a student will be admitted based on academic features.")

st.header("📊 Enter Student Data")

gre = st.slider("GRE Score", 260, 340, 300)
toefl = st.slider("TOEFL Score", 80, 120, 100)
university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.slider("SOP Strength (1.0 - 5.0)", 1.0, 5.0, 3.0, step=0.5)
lor = st.slider("LOR Strength (1.0 - 5.0)", 1.0, 5.0, 3.0, step=0.5)
cgpa = st.slider("CGPA (out of 10)", 6.0, 10.0, 8.0, step=0.1)
research = st.radio("Research Experience", ["No", "Yes"])
research_bin = 1 if research == "Yes" else 0

if st.button("Predict Admission"):
    result, prob = predict_admission(gre, toefl, university_rating, sop, lor, cgpa, research_bin)
    st.subheader("🎯 Result")
    if result == 1:
        st.success(f"✅ The student is likely to be ADMITTED!\nProbability: {prob:.2f}")
    else:
        st.error(f"❌ The student is likely to be REJECTED.\nProbability: {prob:.2f}")
