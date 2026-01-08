import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.title("🎓 Dynamic Student Performance App (Sample Data + Reset + Caching)")

# -------------------------------
# 1. Cached CSV loader
# -------------------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# -------------------------------
# 2. Reset button
# -------------------------------
if st.button("🔄 Reset Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared! Upload a new file or re-enter data.")

# -------------------------------
# 3. Sample Data button
# -------------------------------
if st.button("📊 Load Sample Data"):
    df = pd.DataFrame({
        "Math": [80, 65, 90],
        "English": [70, 60, 85],
        "Science": [75, 55, 95],
        "History": [60, 50, 88],
        "Civic": [72, 58, 92]
    }, index=["Alice", "Bob", "Charlie"])
    st.success("Sample data loaded!")
    subject_list = ["Math","English","Science","History","Civic"]
    student_list = df.index.tolist()
else:
    # -------------------------------
    # 4. Upload CSV or enter manually
    # -------------------------------
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = load_csv(uploaded_file)
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
# 5. Threshold slider
# -------------------------------
PASS_THRESHOLD = st.slider("Set Pass Threshold", min_value=0, max_value=500, value=200)

# -------------------------------
# 6. Compute rule-based results
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
# 7. Cached model training
# -------------------------------
@st.cache_resource
def train_models(X, y_reg, y_cls):
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X, y_reg)

    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cls.fit(X, y_cls)

    return rf_reg, rf_cls

X = df[subject_list]
y_reg = df["Total"]
y_cls = df["PassFail"]

rf_reg, rf_cls = train_models(X, y_reg, y_cls)

# -------------------------------
# 8. Predictions
# -------------------------------
df["Predicted_Total"] = rf_reg.predict(X).round(1)
df["Predicted_PassFail"] = rf_cls.predict(X)

# -------------------------------
# 9. Display results
# -------------------------------
st.subheader("📊 Student Results with Predictions")
st.dataframe(df)

# -------------------------------
# 10. Download results
# -------------------------------
csv = df.to_csv().encode("utf-8")
st.download_button(
    label="📥 Download Results as CSV",
    data=csv,
    file_name="student_results.csv",
    mime="text/csv",
)