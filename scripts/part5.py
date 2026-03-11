import streamlit as st
import pandas as pd
import sqlite3


st.set_page_config (
    page_title = "Fitbit Research Dashboard",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

def get_db_connection():
    return sqlite3.connect('fitbit_database.db')

# Data loading functions
@st.cache_data
def load_general_stats():
    conn = get_db_connection()
    df_activity = pd.read_sql("SELECT TotalSteps, Calories, ActivityDate FROM daily_activity", conn)
    conn.close()
    return df_activity

@st.cache_data
def get_user_list():
    conn = get_db_connection()
    ids = pd.read_sql("SELECT DISTINCT Id FROM daily_activity", conn)
    conn.close()
    return ids['Id'].tolist()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Individual Analytics", "Sleep Analysis"])

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "Project Overview":
    st.title("🏃 Fitbit Research General Statistics")
    df = load_general_stats()
    
    # Numerical Summaries (KPIs)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", len(df))
    col2.metric("Avg Daily Steps", int(df['TotalSteps'].mean()))
    col3.metric("Avg Calories Burned", int(df['Calories'].mean()))
    
    # Graphical Summary
    st.subheader("Activity Distribution Across All Participants")
    fig = px.histogram(df, x="TotalSteps", nbins=50, title="Distribution of Daily Steps", color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: INDIVIDUAL ANALYTICS ---
elif page == "Individual Analytics":
    st.title("👤 Individual Participant Report")
    
    user_ids = get_user_list()
    selected_id = st.sidebar.selectbox("Select Participant ID", user_ids)
    
    # Time Component: Date Filter
    date_range = st.sidebar.date_input("Select Date Range", [])
    
    conn = get_db_connection()
    query = f"SELECT * FROM daily_activity WHERE Id = {selected_id}"
    df_user = pd.read_sql(query, conn)
    df_user['ActivityDate'] = pd.to_datetime(df_user['ActivityDate'])
    
    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_user = df_user[(df_user['ActivityDate'] >= start) & (df_user['ActivityDate'] <= end)]

    st.write(f"Showing data for User: **{selected_id}**")
    
    # Multi-panel graphical summary
    tab1, tab2 = st.tabs(["Trends", "Activity Breakdown"])
    with tab1:
        fig_line = px.line(df_user, x='ActivityDate', y='TotalSteps', title="Step Trend Over Time")
        st.plotly_chart(fig_line, use_container_width=True)
    with tab2:
        # Numerical summary for the individual
        st.dataframe(df_user.describe().T)

# --- PAGE 3: SLEEP ANALYSIS ---
elif page == "Sleep Analysis":
    st.title("🌙 Sleep Duration Factors")
    st.write("Does activity levels affect how long participants sleep?")
    
    conn = get_db_connection()
    # Merging activity and sleep duration
    query = """
    SELECT a.Id, a.VeryActiveMinutes, a.SedentaryMinutes, s.duration 
    FROM daily_activity a
    JOIN (SELECT Id, COUNT(*) as duration FROM minute_sleep GROUP BY Id, logId) s 
    ON a.Id = s.Id
    """
    df_sleep = pd.read_sql(query, conn)
    conn.close()

    # Regression visualization
    var_to_check = st.selectbox("Select variable to compare with Sleep:", ["VeryActiveMinutes", "SedentaryMinutes"])
    
    fig_sleep = px.scatter(df_sleep, x=var_to_check, y="duration", trendline="ols", 
                           title=f"Correlation: {var_to_check} vs Sleep Duration",
                           labels={"duration": "Sleep Minutes", var_to_check: "Active Minutes"})
    st.plotly_chart(fig_sleep, use_container_width=True)