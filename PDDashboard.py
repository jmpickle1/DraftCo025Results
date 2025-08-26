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

# Clean and convert numeric columns, handling special cases
def clean_numeric_column(series):
    """Clean a series by replacing common non-numeric values and converting to numeric"""
    # Replace common non-numeric placeholders
    cleaned = series.replace(['--', '-', 'N/A', 'NA', '', ' '], np.nan)
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(cleaned, errors='coerce')

# Convert all numeric-looking columns to numeric types, except Name, School, Position
for col in proday2025.columns:
    if col not in ['Name', 'School', 'Position']:
        proday2025[col] = clean_numeric_column(proday2025[col])

# Additional data validation
print(f"Data shape: {proday2025.shape}")
print(f"Null values per column:")
for col in proday2025.columns:
    null_count = proday2025[col].isnull().sum()
    if null_count > 0:
        print(f"  {col}: {null_count} nulls")
        
# Remove rows where critical columns are all null
critical_cols = ['Predicted Career APY', 'forty_yard', 'vert_leap', 'broad_jump']
proday2025 = proday2025.dropna(subset=['Predicted Career APY'])  # Must have contract prediction

positions = sorted(proday2025['Position'].dropna().unique())
schools = sorted(proday2025['School'].dropna().unique())

# Prepare columns for custom chart dropdown
exclude_cols = ['Position', 'School', 'Name']
numeric_cols = [col for col in proday2025.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(proday2025[col])]

# ---------------------------------------------------
# Sidebar with Key Stats and Filters
# ---------------------------------------------------
st.sidebar.header("📊 Dataset Overview")

# Key statistics
total_players = len(proday2025)
total_positions = len(positions)
total_schools = len(schools)
avg_contract = proday2025['Predicted Career APY'].mean()
top_contract = proday2025['Predicted Career APY'].max()

st.sidebar.metric("Total Players", f"{total_players:,}")
st.sidebar.metric("Positions", total_positions)
st.sidebar.metric("Schools", total_schools)
st.sidebar.metric("Avg Contract APY", f"${avg_contract:,.0f}")
st.sidebar.metric("Top Contract APY", f"${top_contract:,.0f}")

st.sidebar.header("🔍 Filter Results")

# Contract value filter
min_contract = st.sidebar.slider(
    "Minimum Contract APY ($)",
    min_value=int(proday2025['Predicted Career APY'].min()),
    max_value=int(proday2025['Predicted Career APY'].max()),
    value=int(proday2025['Predicted Career APY'].min()),
    step=100000,
    format="$%d"
)

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
    (proday2025['Position'].isin(selected_positions)) &
    (proday2025['School'].isin(selected_schools)) &
    (proday2025['Predicted Career APY'] >= min_contract)
]

if search_name:
    filtered_base = filtered_base[filtered_base['Name'].str.contains(search_name, case=False, na=False)]

# Show filtered count
st.sidebar.info(f"Showing {len(filtered_base)} of {total_players} players")

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
# Tabs: Dashboard, Analytics, Custom Charts, Player Report
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📈 Analytics", "📊 Custom Charts", "📋 Player Report"])

with tab1:
    st.title("🏠 Dashboard")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drafted_threshold = filtered_base['Predicted Career APY'].quantile(0.7)  # Top 30% as "draftable"
        likely_drafted = len(filtered_base[filtered_base['Predicted Career APY'] >= drafted_threshold])
        st.metric("Likely Drafted", likely_drafted, f"{likely_drafted/len(filtered_base)*100:.1f}%" if len(filtered_base) > 0 else "0%")
    
    with col2:
        if len(filtered_base) > 0:
            avg_forty = filtered_base['forty_yard'].mean()
            st.metric("Avg 40-Yard", f"{avg_forty:.2f}s")
        else:
            st.metric("Avg 40-Yard", "N/A")
    
    with col3:
        if len(filtered_base) > 0:
            avg_vert = filtered_base['vert_leap'].mean()
            st.metric("Avg Vertical", f"{avg_vert:.1f}\"")
        else:
            st.metric("Avg Vertical", "N/A")
    
    with col4:
        if len(filtered_base) > 0:
            top_position = filtered_base['Position'].mode().iloc[0]
            position_count = len(filtered_base[filtered_base['Position'] == top_position])
            st.metric("Top Position", f"{top_position} ({position_count})")
        else:
            st.metric("Top Position", "N/A")
    
    st.subheader("Filtered Pro Day Data")

    if filtered_base.empty:
        st.warning("⚠️ No data matches your filters.")
    else:
        df_to_style = filtered_base.copy()

        # Round all numeric columns to 2 decimals EXCEPT 'Name', 'School', 'Position'
        for col in df_to_style.columns:
            if col not in ['Name', 'School', 'Position'] and pd.api.types.is_numeric_dtype(df_to_style[col]):
                df_to_style[col] = df_to_style[col].round(2)

        # Format 'Predicted Career APY' as dollars for display
        df_display = df_to_style.copy()
        if 'Predicted Career APY' in df_display.columns:
            df_display['Predicted Career APY'] = df_display['Predicted Career APY'].apply(
                lambda x: f"${x:,.0f}" if pd.notnull(x) else "")

        # Sort by contract value descending
        df_display = df_display.sort_values('Predicted Career APY', ascending=False, key=lambda x: df_to_style['Predicted Career APY'])

        # Apply color scale and handle potential data type issues
        try:
            styled_df = df_to_style.sort_values('Predicted Career APY', ascending=False).style.apply(color_scale, axis=None)
            st.dataframe(styled_df, use_container_width=True, height=600)
        except Exception as e:
            st.warning(f"Styling failed, showing unstyled data: {e}")
            # Fallback to unstyled dataframe
            display_df = df_to_style.sort_values('Predicted Career APY', ascending=False)
            st.dataframe(display_df, use_container_width=True, height=600)

with tab2:
    st.title("📈 Analytics")
    
    if filtered_base.empty:
        st.warning("⚠️ No data matches your filters.")
    else:
        # Contract Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Contract Distribution")
            contract_hist = alt.Chart(filtered_base).mark_bar().encode(
                x=alt.X('Predicted Career APY:Q', bin=alt.Bin(maxbins=20), title='Contract APY ($)'),
                y=alt.Y('count()', title='Number of Players'),
                tooltip=['count()']
            ).properties(height=300)
            st.altair_chart(contract_hist, use_container_width=True)
        
        with col2:
            st.subheader("🏃‍♂️ 40-Yard Dash Distribution")
            if 'forty_yard' in filtered_base.columns:
                forty_hist = alt.Chart(filtered_base.dropna(subset=['forty_yard'])).mark_bar().encode(
                    x=alt.X('forty_yard:Q', bin=alt.Bin(maxbins=20), title='40-Yard Dash (s)'),
                    y=alt.Y('count()', title='Number of Players'),
                    tooltip=['count()']
                ).properties(height=300)
                st.altair_chart(forty_hist, use_container_width=True)
        
        # Contract Analysis with error handling
        try:
            position_stats = filtered_base.groupby('Position').agg({
                'Predicted Career APY': ['count', 'mean', 'max'],
                'forty_yard': 'mean',
                'vert_leap': 'mean'
            }).round(2)
            
            position_stats.columns = ['Count', 'Avg Contract', 'Max Contract', 'Avg 40-Yard', 'Avg Vertical']
            position_stats = position_stats.sort_values('Avg Contract', ascending=False)
            
            # Format contract columns as currency
            for col in ['Avg Contract', 'Max Contract']:
                position_stats[col] = position_stats[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(position_stats, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating position analysis: {e}")
        
        # School Analysis with error handling
        try:
            st.subheader("🏫 Top Schools by Average Contract (Min 3 Players)")
            school_stats = filtered_base.groupby('School').agg({
                'Predicted Career APY': ['count', 'mean', 'max']
            }).round(0)
            
            school_stats.columns = ['Player Count', 'Avg Contract', 'Max Contract']
            school_stats = school_stats[school_stats['Player Count'] >= 3].sort_values('Avg Contract', ascending=False)
            
            # Format contract columns
            for col in ['Avg Contract', 'Max Contract']:
                school_stats[col] = school_stats[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(school_stats.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error generating school analysis: {e}")

with tab3:
    st.title("📊 Custom Charts")

    pos_for_chart = st.selectbox("Select Position for Chart", positions)
    stat_for_chart = st.selectbox("Select Statistic", numeric_cols)

    chart_data = proday2025[
        (proday2025['Position'] == pos_for_chart) & 
        proday2025[stat_for_chart].notna()
    ]

    if chart_data.empty:
        st.warning("⚠️ No data available for this selection.")
    else:
        # Show both top and bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Top 10 {pos_for_chart} - {stat_for_chart}")
            top10 = chart_data.nlargest(10, stat_for_chart)
            bar_chart_top = alt.Chart(top10).mark_bar(color='green').encode(
                x=alt.X(f"{stat_for_chart}:Q", title=stat_for_chart),
                y=alt.Y('Name:N', sort='-x', title='Player'),
                tooltip=['Name', 'School', stat_for_chart]
            ).properties(height=400)
            st.altair_chart(bar_chart_top, use_container_width=True)
        
        with col2:
            if stat_for_chart in smaller_better_cols:  # For speed stats, show fastest (smallest)
                st.subheader(f"Fastest 10 {pos_for_chart} - {stat_for_chart}")
                bottom10 = chart_data.nsmallest(10, stat_for_chart)
            else:  # For other stats, show bottom 10
                st.subheader(f"Bottom 10 {pos_for_chart} - {stat_for_chart}")
                bottom10 = chart_data.nsmallest(10, stat_for_chart)
            
            bar_chart_bottom = alt.Chart(bottom10).mark_bar(color='red').encode(
                x=alt.X(f"{stat_for_chart}:Q", title=stat_for_chart),
                y=alt.Y('Name:N', sort='x', title='Player'),
                tooltip=['Name', 'School', stat_for_chart]
            ).properties(height=400)
            st.altair_chart(bar_chart_bottom, use_container_width=True)

with tab4:
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
            player_school = player_data['School']
            player_contract = player_data['Predicted Career APY']

            # Header with key info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### {selected_player}")
                st.markdown(f"**School:** {player_school}")
                st.markdown(f"**Position:** {player_position}")
            
            with col2:
                st.metric("Predicted Contract APY", f"${player_contract:,.0f}")
                
                # Contract rank among all players
                contract_rank = (proday2025['Predicted Career APY'] > player_contract).sum() + 1
                total_players = len(proday2025)
                st.metric("Overall Rank", f"{contract_rank} of {total_players}")
            
            with col3:
                # Position rank
                position_group = proday2025[proday2025['Position'] == player_position]
                position_rank = (position_group['Predicted Career APY'] > player_contract).sum() + 1
                position_total = len(position_group)
                st.metric("Position Rank", f"{position_rank} of {position_total}")

            # Get only players of the same position for percentile calculations
            position_group = proday2025[proday2025['Position'] == player_position]

            # Define stats to evaluate (numeric columns only, exclude Name, School, Position)
            report_stats = [col for col in proday2025.columns if col not in ['Name', 'School', 'Position'] and pd.api.types.is_numeric_dtype(proday2025[col])]
            speed_stats = ['forty_yard', 'Shuttle', 'three_cone']

            # Calculate percentiles for each stat for this player within their position group
            percentiles = {}
            values = {}
            
            for stat in report_stats:
                stat_values = position_group[stat].dropna()
                player_stat_value = player_data[stat]
                values[stat] = player_stat_value

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
                'Value': [f"{v:.2f}" if pd.notnull(v) else "N/A" for v in values.values()],
                'Percentile vs Position': [f"{v}%" if v is not None else "N/A" for v in percentiles.values()]
            })

            # Add performance indicators
            def get_performance_indicator(percentile_str):
                if percentile_str == "N/A":
                    return "⚪"
                perc = float(percentile_str.replace('%', ''))
                if perc >= 90:
                    return "🟢 Elite"
                elif perc >= 75:
                    return "🔵 Above Average"
                elif perc >= 50:
                    return "🟡 Average"
                elif perc >= 25:
                    return "🟠 Below Average"
                else:
                    return "🔴 Poor"

            report_df['Performance'] = report_df['Percentile vs Position'].apply(get_performance_indicator)

            st.subheader("Athletic Performance Breakdown")
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            # Radar chart would be nice here, but keeping it simple for now
            st.subheader("Quick Summary")
            
            # Calculate overall grade
            valid_percentiles = [p for p in percentiles.values() if p is not None]
            if valid_percentiles:
                overall_grade = sum(valid_percentiles) / len(valid_percentiles)
                
                if overall_grade >= 80:
                    grade_color = "🟢"
                    grade_text = "Elite Prospect"
                elif overall_grade >= 65:
                    grade_color = "🔵"
                    grade_text = "Strong Prospect"
                elif overall_grade >= 50:
                    grade_color = "🟡"
                    grade_text = "Average Prospect"
                elif overall_grade >= 35:
                    grade_color = "🟠"
                    grade_text = "Below Average"
                else:
                    grade_color = "🔴"
                    grade_text = "Poor Prospect"
                
                st.markdown(f"**Overall Grade:** {grade_color} {overall_grade:.1f}/100 - {grade_text}")
                
                # Strengths and weaknesses
                strengths = [stat for stat, perc in percentiles.items() if perc and perc >= 75]
                weaknesses = [stat for stat, perc in percentiles.items() if perc and perc <= 25]
                
                if strengths:
                    st.markdown(f"**Strengths:** {', '.join(strengths)}")
                if weaknesses:
                    st.markdown(f"**Areas for Improvement:** {', '.join(weaknesses)}")