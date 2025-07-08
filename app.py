import streamlit as st
import pandas as pd

# Load the Excel file
df = pd.read_excel(r"/Users/jacobpickle/Documents/RPM/2025 Pro Day Data/2025 Individual Results.xlsx")

# Title and description
st.title("2025 Pro Day Dashboard")
st.markdown("Explore individual results from the 2025 Pro Day")

# Show raw data toggle
if st.checkbox("Show Raw Data"):
    st.write(df)

# Column selector
columns = st.multiselect("Select columns to display:", options=df.columns, default=df.columns.tolist())
st.dataframe(df[columns])

# Search by player (assuming a column like 'Name' exists)
if "Name" in df.columns:
    search_name = st.text_input("Search for a player by name:")
    filtered_df = df[df["Name"].str.contains(search_name, case=False, na=False)]
    st.dataframe(filtered_df[columns])
