import streamlit as st
#import matplotlib.pyplot as plt
import pandas as pd



# Example data
students = ["Alice", "Bob", "Charlie", "David", "Emma"]
actual = ["Pass", "Fail", "Pass", "Fail", "Pass"]
predicted = ["Pass", "Pass", "Fail", "Fail", "Pass"]

# Convert Pass/Fail to numeric (Pass=1, Fail=0)
actual_numeric = [1 if x=="Pass" else 0 for x in actual]
predicted_numeric = [1 if x=="Pass" else 0 for x in predicted]

# Plot side-by-side bars
fig, ax = plt.subplots()
x = range(len(students))
ax.bar([i-0.2 for i in x], actual_numeric, width=0.4, label="Actual", color="blue")
ax.bar([i+0.2 for i in x], predicted_numeric, width=0.4, label="Predicted", color="orange")

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(students)
ax.set_ylabel("Pass=1, Fail=0")
ax.set_title("Actual vs Predicted Pass/Fail Outcomes")
ax.legend()

# Show in Streamlit
st.pyplot(fig)