import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Dynamic Student Performance App with ML Predictions")

# -------------------------------
# 1. Load trained models
# -------------------------------
bundle = joblib.load("output/random_forest_bundle.pkl")
rf_reg = bundle["regression_model"]
rf_cls = bundle["classification_model"]

# -------------------------------
# 2. Upload CSV or enter manually
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:")
    st.dataframe(df.head())
    subject_list = [col for col in df.columns if col not in ["Total","Average","Grade","PassFail"]]
    student_list = df.index.tolist()
else:
    # Manual entry
    subjects = st.text_input("Enter subjects (comma-separated)", "Math,English,Science")
    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]

    students = st.text_input("Enter student names (comma-separated)", "Alice,Bob,Charlie")
    student_list = [s.strip() for s in students.split(",") if s.strip()]

    data = {}
    for subj in subject_list:
        scores = []
        for student in student_list:
            score = st.number_input(f"{student}'s {subj} score", min_value=0, max_value=100, value=50)
            scores.append(score)
        data[subj] = scores

    df = pd.DataFrame(data, index=student_list)

# -------------------------------
# 3. Threshold slider
# -------------------------------
PASS_THRESHOLD = st.slider("Set Pass Threshold", min_value=0, max_value=500, value=200)

# -------------------------------
# 4. Compute rule-based results
# -------------------------------
df["Total"] = df[subject_list].sum(axis=1)
df["Average"] = df[subject_list].mean(axis=1)

def assign_grade(avg: float) -> str:
    if avg >= 70: return "A"
    elif avg >= 60: return "B"
    elif avg >= 50: return "C"
    elif avg >= 40: return "D"
    else: return "F"

df["Grade"] = df["Average"].apply(assign_grade)
df["PassFail"] = df["Total"].apply(lambda x: "Pass" if x >= PASS_THRESHOLD else "Fail")

# -------------------------------
# 5. ML Predictions
# -------------------------------
X_new = df[subject_list]

df["Predicted_Total"] = rf_reg.predict(X_new).round(1)
df["Predicted_PassFail"] = rf_cls.predict(X_new)

# -------------------------------
# 6. Display results
# -------------------------------
st.subheader("📊 Student Results with Predictions")
st.dataframe(df)

# -------------------------------
# 7. Download results
# -------------------------------
csv = df.to_csv().encode("utf-8")
st.download_button(
    label="📥 Download Results as CSV",
    data=csv,
    file_name="student_results.csv",
    mime="text/csv",
)