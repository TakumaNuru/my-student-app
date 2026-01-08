import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Load bundled models
# -------------------------------
bundle = joblib.load("output/random_forest_bundle.pkl")
rf_reg = bundle["regression_model"]
rf_cls = bundle["classification_model"]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Student Performance Prediction App")

# Threshold slider
PASS_THRESHOLD = st.slider("Set Pass Threshold", min_value=50, max_value=500, value=200)

st.write("Enter student scores below to predict Total Score (Regression) and Pass/Fail (Classification).")

# Input fields
math = st.number_input("Math Score", min_value=0, max_value=100, value=50)
english = st.number_input("English Score", min_value=0, max_value=100, value=50)
science = st.number_input("Science Score", min_value=0, max_value=100, value=50)
history = st.number_input("History Score", min_value=0, max_value=100, value=50)
civic = st.number_input("Civic Score", min_value=0, max_value=100, value=50)

# -------------------------------
# Create input DataFrame
# -------------------------------
X_new = pd.DataFrame(
    [[math, english, science, history, civic]],
    columns=["Math", "English", "Science", "History", "Civic"]
)

# -------------------------------
# Predictions
# -------------------------------
# Regression prediction
total_pred = rf_reg.predict(X_new)[0]

# Classification prediction (model-based)
passfail_model_pred = rf_cls.predict(X_new)[0]

# Classification prediction (threshold-based)
passfail_threshold_pred = "Pass" if total_pred >= PASS_THRESHOLD else "Fail"

# -------------------------------
# Display results
# -------------------------------
st.subheader("📊 Predictions")
st.write(f"Predicted Total Score: **{total_pred:.2f}**")
st.write(f"Predicted Outcome (Model): **{passfail_model_pred}**")
st.write(f"Predicted Outcome (Threshold={PASS_THRESHOLD}): **{passfail_threshold_pred}**")