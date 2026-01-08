import streamlit as st
import pandas as pd

# Example data
students = ["Alice", "Bob", "Charlie", "David", "Emma"]
actual = ["Pass", "Fail", "Pass", "Fail", "Pass"]
predicted = ["Pass", "Pass", "Fail", "Fail", "Pass"]

# Convert Pass/Fail to numeric (Pass=1, Fail=0)
df = pd.DataFrame({
    "Student": students,
    "Actual": [1 if x=="Pass" else 0 for x in actual],
    "Predicted": [1 if x=="Pass" else 0 for x in predicted]
})

st.bar_chart(df.set_index("Student"))