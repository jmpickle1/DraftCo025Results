import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="2025 Draft Class Athletic Testing Model Results",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Load Data (CSV)
# ---------------------------------------------------
csv_path = "Updated 2025 Contract Predictions.csv"


try:
    proday2025 = pd.read_csv(csv_path)
    st.success("✅ CSV loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    st.stop()

# Convert all numeric-looking columns to numeric types, except Name, School, Position
for col in proday2025.columns:
    if col not in ['Name', 'School', 'Position']:
        proday2025[col] = pd.to_numeric(proday2025[col], errors='coerce')

positions = sorted(proday2025['Position'].dropna().unique())
schools = sorted(proday2025['School'].dropna().unique())

# Prepare columns for custom chart dropdown
exclude_cols = ['Position', 'School', 'Name']
numeric_cols = [col for col in proday2025.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(proday2025[col])]

# ---------------------------------------------------
# Sidebar Filters with Select All dropdowns
# ---------------------------------------------------
st.sidebar.header("📊 Filter Results")

# Positions
select_all_positions = st.sidebar.checkbox("Select All Positions", value=True)
if select_all_positions:
    selected_positions = positions
else:
    selected_positions = st.sidebar.multiselect("Select Position(s)", positions, default=[])

# Schools
select_all_schools = st.sidebar.checkbox("Select All Schools", value=True)
if select_all_schools:
    selected_schools = schools
else:
    selected_schools = st.sidebar.multiselect("Select School(s)", schools, default=[])

# Player name search
search_name = st.sidebar.text_input("Search by Player Name")

# Filter dataframe based on selections
filtered_base = proday2025[
    proday2025['Position'].isin(selected_positions) &
    proday2025['School'].isin(selected_schools)
]

if search_name:
    filtered_base = filtered_base[filtered_base['Name'].str.contains(search_name, case=False, na=False)]

# Ensure numeric conversion on filtered data for relevant columns before styling
larger_better_cols = ['Height', 'Weight', 'Arm_Length', 'vert_leap', 'broad_jump', 'Predicted Career APY']
smaller_better_cols = ['forty_yard', 'Shuttle', 'three_cone']

for col in larger_better_cols + smaller_better_cols:
    if col in filtered_base.columns:
        filtered_base[col] = pd.to_numeric(filtered_base[col], errors='coerce')

# Keep only columns present in data and numeric
larger_better_cols = [col for col in larger_better_cols if col in filtered_base.columns and pd.api.types.is_numeric_dtype(filtered_base[col])]
smaller_better_cols = [col for col in smaller_better_cols if col in filtered_base.columns and pd.api.types.is_numeric_dtype(filtered_base[col])]

# ---------------------------------------------------
# Color scale function (green gradient, robust with inversion for speed)
# ---------------------------------------------------
def color_scale(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    cmap = sns.color_palette("Greens", as_cmap=True)

    # Larger is better: higher value = darker green
    for col in larger_better_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min, col_max = df[col].min(), df[col].max()
            range_val = col_max - col_min
            if range_val == 0 or pd.isna(range_val):
                norm = pd.Series(0.5, index=df.index)
            else:
                norm = (df[col] - col_min) / range_val
            colors = norm.apply(lambda x: cmap(x))
            styles[col] = ['background-color: rgba({},{},{},{})'.format(
                int(c[0]*255), int(c[1]*255), int(c[2]*255), 0.6) for c in colors]

    # Smaller is better: lower value = darker green (invert scale)
    for col in smaller_better_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min, col_max = df[col].min(), df[col].max()
            range_val = col_max - col_min
            if range_val == 0 or pd.isna(range_val):
                norm = pd.Series(0.5, index=df.index)
            else:
                norm = 1 - ((df[col] - col_min) / range_val)
            colors = norm.apply(lambda x: cmap(x))
            styles[col] = ['background-color: rgba({},{},{},{})'.format(
                int(c[0]*255), int(c[1]*255), int(c[2]*255), 0.6) for c in colors]

    return styles

# ---------------------------------------------------
# Tabs: Dashboard, Custom Charts, Player Report
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📊 Custom Charts", "📋 Player Report"])

with tab1:
    st.title("🏠 Dashboard")
    st.subheader("Filtered Pro Day Data")

    if filtered_base.empty:
        st.warning("⚠️ No data matches your filters.")
    else:
        df_to_style = filtered_base.copy()

        # Round all numeric columns to 2 decimals EXCEPT 'Name', 'School', 'Position'
        for col in df_to_style.columns:
            if col not in ['Name', 'School', 'Position'] and pd.api.types.is_numeric_dtype(df_to_style[col]):
                df_to_style[col] = df_to_style[col].round(2)

        # Format 'Predicted Career APY' as dollars in display copy only
        df_display = df_to_style.copy()
        if 'Predicted Career APY' in df_display.columns:
            df_display['Predicted Career APY'] = df_display['Predicted Career APY'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "")

        # Apply color scale (using unformatted df_to_style for numeric accuracy)
        styled_df = df_to_style.style.apply(color_scale, axis=None)

        st.dataframe(styled_df, use_container_width=True)

    # Optional: 40-Yard Dash Chart
    if 'FortyYardDash' in filtered_base.columns:
        st.subheader("⏱️ 40-Yard Dash Distribution")
        chart = alt.Chart(filtered_base.dropna(subset=['FortyYardDash'])).mark_bar().encode(
            x=alt.X('FortyYardDash:Q', bin=True, title='40-Yard Dash Time'),
            y=alt.Y('count()', title='Number of Players'),
            tooltip=['FortyYardDash']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

with tab2:
    st.title("Custom Leaderboards")

    pos_for_chart = st.selectbox("Select Position for Chart", positions)
    stat_for_chart = st.selectbox("Select Statistic", numeric_cols)

    chart_data = proday2025[
        (proday2025['Position'] == pos_for_chart) & 
        proday2025[stat_for_chart].notna()
    ]

    if chart_data.empty:
        st.warning("⚠️ No data available for this selection.")
    else:
        top10 = chart_data.nlargest(10, stat_for_chart)
        bar_chart = alt.Chart(top10).mark_bar().encode(
            x=alt.X(f"{stat_for_chart}:Q", title=stat_for_chart),
            y=alt.Y('Name:N', sort='-x', title='Player'),
            tooltip=['Name', stat_for_chart]
        ).properties(height=400)
        st.altair_chart(bar_chart, use_container_width=True)

with tab3:
    st.title("📋 Player Report Card")

    # Player selector (autocomplete)
    all_players = proday2025['Name'].dropna().unique()
    selected_player = st.selectbox("Select Player", sorted(all_players))

    if selected_player:
        player_data = proday2025[proday2025['Name'] == selected_player]

        if player_data.empty:
            st.warning("Player not found!")
        else:
            player_data = player_data.iloc[0]  # Series of player stats
            player_position = player_data['Position']

            st.markdown(f"### Report for: {selected_player}")
            st.markdown(f"**School:** {player_data['School']}")
            st.markdown(f"**Position:** {player_position}")

            # Get only players of the same position
            position_group = proday2025[proday2025['Position'] == player_position]

            # Define stats to evaluate (numeric columns only, exclude Name, School, Position)
            report_stats = [col for col in proday2025.columns if col not in ['Name', 'School', 'Position'] and pd.api.types.is_numeric_dtype(proday2025[col])]

            speed_stats = ['forty_yard', 'Shuttle', 'three_cone']

            # Calculate percentiles for each stat for this player within their position group
            percentiles = {}
            for stat in report_stats:
                stat_values = position_group[stat].dropna()
                player_stat_value = player_data[stat]

                if pd.isna(player_stat_value):
                    percentiles[stat] = None
                else:
                    raw_perc = stats.percentileofscore(stat_values, player_stat_value) / 100  # convert to 0-1 scale
                    # Flip percentile for speed stats (lower is better)
                    if stat in speed_stats:
                        perc = 1 - raw_perc
                    else:
                        perc = raw_perc
                    percentiles[stat] = round(perc * 100, 1)

            # Build a DataFrame for display
            report_df = pd.DataFrame({
                'Statistic': list(percentiles.keys()),
                'Percentile': [f"{v}%" if v is not None else "N/A" for v in percentiles.values()]
            })

            st.dataframe(report_df, use_container_width=False, width=400)
