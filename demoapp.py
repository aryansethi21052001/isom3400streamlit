import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Show basic info
    st.subheader("Data Overview")
    st.write(f"Shape: {df.shape}")
    st.write("First 5 rows:")
    st.dataframe(df.head())
    
    # Data info
    st.write("Data Types:")
    st.write(df.dtypes)
    
    # Column selection for analysis
    st.subheader("Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-axis", df.columns)
    with col2:
        y_axis = st.selectbox("Y-axis", df.columns)
    
    # Plot type selection
    plot_type = st.radio(
        "Select plot type:",
        ["Scatter Plot", "Line Plot", "Histogram", "Bar Chart"]
    )
    
    # Generate plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == "Scatter Plot":
        ax.scatter(df[x_axis], df[y_axis], alpha=0.7, color='blue')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{x_axis} vs {y_axis}")
        ax.grid(True, alpha=0.3)
        
    elif plot_type == "Line Plot":
        ax.plot(df[x_axis], df[y_axis], color='green', linewidth=2)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{x_axis} vs {y_axis}")
        ax.grid(True, alpha=0.3)
        
    elif plot_type == "Histogram":
        ax.hist(df[x_axis], bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel(x_axis)
        ax.set_ylabel('Frequency')
        ax.set_title(f"Distribution of {x_axis}")
        ax.grid(True, alpha=0.3)
        
    elif plot_type == "Bar Chart":
        value_counts = df[x_axis].value_counts().head(10)
        ax.bar(value_counts.index, value_counts.values, color='purple', alpha=0.7)
        ax.set_xlabel(x_axis)
        ax.set_ylabel('Count')
        ax.set_title(f"Top 10 {x_axis} Values")
        plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Basic statistics
    st.subheader("Basic Statistics")
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        st.write("Numerical Columns Statistics:")
        st.write(df.describe())
    
else:
    st.info("Please upload a CSV file to get started")
    
    # Sample data for demonstration
    st.subheader("Try with Sample Data")
    if st.button("Generate Sample Data"):
        sample_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Sales': [120, 150, 180, 90, 200, 170],
            'Customers': [50, 65, 80, 45, 95, 75],
            'Revenue': [1200, 1500, 1800, 900, 2000, 1700]
        })
        st.dataframe(sample_data)
        
        # Show sample plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sample_data['Month'], sample_data['Sales'], color='skyblue')
        ax.set_xlabel('Month')
        ax.set_ylabel('Sales')
        ax.set_title('Sample Sales Data')
        st.pyplot(fig)

# Caching for performance
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# Session state for persistence
if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("Increment Counter"):
    st.session_state.counter += 1

st.write(f"Counter: {st.session_state.counter}")

# Expander for additional options
with st.expander("Advanced Options"):
    st.slider("Select range", 0, 100, (25, 75))
    st.color_picker("Choose a color")
