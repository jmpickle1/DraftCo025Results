import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px

# ---------------------------------------------------
# Page Config & App Title
# ---------------------------------------------------
st.set_page_config(
    page_title="2025 Pro Day Results",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Header with logo (uncomment if logo file available)
# col_logo, col_title = st.columns([1, 5])
# with col_logo:
#     st.image("football_logo.png", width=60)
# with col_title:
#     st.markdown("## 🏈 2025 Pro Day Results Dashboard")

st.markdown("## 🚀 2025 Pro Day Results Dashboard")
st.markdown("Welcome to the RPM Pro Day data explorer. Use the filters on the left to interact with the data.")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
data_path = r"/Users/jacobpickle/Documents/RPM/2025 Pro Day Data/2025 Individual Results.xlsx"
proday2025 = pd.read_excel(data_path)

# ---------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------
st.sidebar.header("Filter Results")

# Ensure Position and School exist and dropna for clean filters
positions = sorted(proday2025['Position'].dropna().unique())
schools = sorted(proday2025['School'].dropna().unique())

# Sidebar multiselect filters
selected_positions = st.sidebar.multiselect("Select Position(s)", positions, default=positions)
selected_schools = st.sidebar.multiselect("Select School(s)", schools, default=schools)

# Apply filters
filtered_data = proday2025[
    proday2025['Position'].isin(selected_positions) &
    proday2025['School'].isin(selected_schools)
]

# ---------------------------------------------------
# Sortable Data Table
# ---------------------------------------------------
st.markdown("### 📋 Filtered Pro Day Data")
st.dataframe(
    filtered_data,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------
# Optional Chart (Uncomment if numeric column exists)
# if 'FortyYardDash' in proday2025.columns:
#     st.markdown("### ⏱️ 40 Yard Dash Distribution")
#     chart = alt.Chart(filtered_data).mark_bar().encode(
#         x=alt.X('FortyYardDash:Q', bin=True, title="40-Yard Dash Time (s)"),
#         y=alt.Y('count()', title='Number of Players'),
#         tooltip=['FortyYardDash']
#     ).properties(height=300)
#     st.altair_chart(chart, use_container_width=True)

st.write("✅ App is running")

try:
    df = pd.read_excel("2025 Individual Results.xlsx")
    st.write("Loaded data:", df.shape)
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ Error loading file: {e}")
