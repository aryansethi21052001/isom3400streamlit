import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Personal Expense Tracker")

if 'expenses' not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=['Date','Category','Amount','Description'])

with st.form("expense_form"):
    st.subheader("Add New Expense")
    date = st.date_input("Date")
    category = st.selectbox("Category",["Food","Transport","Entertainment","Bills","Other"])
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    description = st.text_input("Description")

    add_expense_button = st.form_submit_button("Add Expense")

    if add_expense_button:
        new_expense = pd.DataFrame({
            'Date': [date],
            'Category': [category],
            'Amount': [amount],
            'Description': [description]
        })

        st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
        st.success("Expense added!")
    
if not st.session_state.expenses.empty:
    st.subheader("Your Expenses")
    st.dataframe(st.session_state.expenses)

    st.subheader("Summary")
    total_spent = st.session_state.expenses['Amount'].sum()
    st.write(f"Total Spent: ${total_spent:.2f}")

    category_totals = st.session_state.expenses.groupby('Category')['Amount'].sum()

    fig, ax = plt.subplots(figsize=(10,6))
    ax.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%')
    ax.set_title("Expenses by Category")
    st.pyplot(fig)
else:
    st.info("No expenses recorded yet. Add some expenses above.")
