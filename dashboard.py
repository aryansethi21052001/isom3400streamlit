import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Title and Description
# -------------------------------
st.title("Business Sales Dashboard")
st.write("Analyse monthly sales data interactively!")

# -------------------------------
# Sample Data
# -------------------------------
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
sales = np.random.randint(5000, 20000, size = 12)
expenses = np.random.randint(3000, 15000, size = 12)

data = pd.DataFrame({
    "Month": months,
    "Sales": sales,
    "Expenses": expenses
})

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")
selected_months = st.sidebar.multiselect("Select Months", months, default = months)
show_expenses = st.sidebar.checkbox("Show Expenses", value = True)

# Filter data
filtered_data = data[data["Month"].isin(selected_months)]

# -------------------------------
# Display Data Table
# -------------------------------
st.subheader("Filtered Data")
st.dataframe(filtered_data)

# -------------------------------
# Interactive Chart
# -------------------------------
st.subheader("Sales Chart")
fig, ax = plt.subplots()
ax.plot(filtered_data["Month"], filtered_data["Sales"], marker = 'o', label = "Sales")
if show_expenses:
    ax.plot(filtered_data["Month"], filtered_data["Expenses"], marker = 'o', label = "Expenses")
ax.set_title("Monthly Performance")
ax.set_xlabel("Month")
ax.set_ylabel("Amount ($)")
ax.legend()
st.pyplot(fig)

# -------------------------------
# KPI Metrics
# -------------------------------
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${filtered_data['Sales'].sum():,.0f}")
col2.metric("Total Expenses", f"${filtered_data['Expenses'].sum():,.0f}")
col3.metric("Profit", f"${(filtered_data['Sales'].sum() - filtered_data['Expenses'].sum()):,.0f}")
