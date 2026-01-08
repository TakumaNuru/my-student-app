import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

st.title("🎓 Dynamic Student Performance App (Retrains Models)")

# -------------------------------
# 1. Upload CSV or enter manually
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:")
    st.dataframe(df.head())
    subject_list = [col for col in df.columns if col not in ["Total","Average","Grade","PassFail"]]
    student_list = df.index.tolist()
else:
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
# 2. Threshold slider
# -------------------------------
PASS_THRESHOLD = st.slider("Set Pass Threshold", min_value=0, max_value=500, value=200)

# -------------------------------
# 3. Compute rule-based results
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
# 4. Retrain ML models dynamically
# -------------------------------
X = df[subject_list]
y_reg = df["Total"]
y_cls = df["PassFail"]

# Train/test split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)
_, _, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.3, random_state=42)

# Train regression model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train_reg)

# Train classification model
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_train, y_train_cls)

# -------------------------------
# 5. Predictions
# -------------------------------
df["Predicted_Total"] = rf_reg.predict(X).round(1)
df["Predicted_PassFail"] = rf_cls.predict(X)

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

# -------------------------------
# 8. Show model performance
# -------------------------------
st.subheader("📈 Model Performance")
st.write(f"Regression R²: {r2_score(y_test_reg, rf_reg.predict(X_test)):.3f}")
st.write(f"Classification Accuracy: {accuracy_score(y_test_cls, rf_cls.predict(X_test)):.3f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test_cls, rf_cls.predict(X_test), labels=["Pass","Fail"]))