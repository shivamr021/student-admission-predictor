import streamlit as st
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Student Admission Predictor", layout="centered")
st.title("🎓 Student Admission Predictor")
st.markdown("""
Enter your academic details below to predict whether you'll be admitted.
""")

# --- Input form ---
gre = st.number_input("GRE Score (out of 340)", min_value=0, max_value=340, value=300)
toefl = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=100)
university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
sop = st.slider("Statement of Purpose (SOP) Strength", 1.0, 5.0, 3.0, step=0.5)
lor = st.slider("Letter of Recommendation (LOR) Strength", 1.0, 5.0, 3.0, step=0.5)
cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.0, step=0.01)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Convert research to binary
research = 1 if research == "Yes" else 0

# --- Prediction Logic ---
# Normalization constants (from original dataset)
mean_vals = np.array([316.47, 107.13, 3.09, 3.37, 3.49, 8.60, 0.56])
std_vals = np.array([11.48, 6.05, 1.14, 1.01, 0.90, 0.60, 0.50])

# Sample trained weights (from logistic regression model)
theta = np.array([-0.35, 0.47, 0.54, 0.14, 0.20, 0.58, 0.35, 0.91])  # last is bias term

# Prepare feature vector
x = np.array([gre, toefl, university_rating, sop, lor, cgpa, research])
x_norm = (x - mean_vals) / std_vals
x_input = np.append(x_norm, 1)  # Add bias term

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Make prediction
prob = sigmoid(np.dot(theta, x_input))
pred = 1 if prob >= 0.5 else 0

# --- Show result ---
if st.button("Predict Admission"):
    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of Admission:** {prob:.2f}")
    if pred:
        st.success("🎉 You are likely to be admitted!")
    else:
        st.error("😞 You are less likely to be admitted.")

    st.caption("Note: This prediction is based on a logistic regression model trained on a small dataset. Results may not be highly accurate.")
