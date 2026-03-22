import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = str(Path(__file__).parent / "fitbit_database.db")


def get_connection():
    return sqlite3.connect(DB_PATH)

def query_to_df(conn, query, params=()):
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

def parse_dates(series):
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().sum() > len(parsed) * 0.5:
        parsed = pd.to_datetime(series, errors="coerce", format="%m/%d/%Y %I:%M:%S %p")
    return parsed.dt.date.astype(str)

def get_all_user_ids():
    conn = get_connection()
    df = query_to_df(conn, "SELECT DISTINCT Id FROM daily_activity ORDER BY Id")
    conn.close()
    return df["Id"].astype(int).tolist()

def get_user_classes():
    conn = get_connection()
    df = query_to_df(conn, """
        SELECT Id, COUNT(*) AS days_tracked,
               CASE WHEN COUNT(*) <= 10 THEN 'Light'
                    WHEN COUNT(*) <= 15 THEN 'Moderate'
                    ELSE 'Heavy' END AS Class
        FROM daily_activity GROUP BY Id ORDER BY Id
    """)
    conn.close()
    df["Id"] = df["Id"].astype(int)
    return df

def get_overview_stats():
    conn = get_connection()
    df = query_to_df(conn, """
        SELECT COUNT(DISTINCT Id) AS total_users,
               AVG(TotalSteps) AS avg_steps, AVG(Calories) AS avg_calories,
               AVG(TotalDistance) AS avg_distance,
               AVG(SedentaryMinutes) AS avg_sedentary,
               AVG(VeryActiveMinutes) AS avg_very_active
        FROM daily_activity
    """)
    conn.close()
    return df.iloc[0].to_dict()

def get_total_distance_per_user():
    conn = get_connection()
    df = query_to_df(conn, """
        SELECT Id, ROUND(SUM(TotalDistance), 2) AS TotalDistance
        FROM daily_activity GROUP BY Id ORDER BY TotalDistance DESC
    """)
    conn.close()
    df["Id"] = df["Id"].astype(int)
    return df

def get_workout_frequency_by_weekday():
    conn = get_connection()
    df = query_to_df(conn, "SELECT ActivityDate FROM daily_activity")
    conn.close()
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], infer_datetime_format=True)
    df["Weekday"] = df["ActivityDate"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = df["Weekday"].value_counts().reindex(order).reset_index()
    counts.columns = ["Weekday", "Count"]
    return counts

def get_avg_metrics_by_weekday():
    conn = get_connection()
    df = query_to_df(conn, "SELECT ActivityDate, TotalSteps, Calories FROM daily_activity")
    conn.close()
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], infer_datetime_format=True)
    df["Weekday"] = df["ActivityDate"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return df.groupby("Weekday")[["TotalSteps", "Calories"]].mean().reindex(order).reset_index()

def get_steps_and_active_minutes():
    conn = get_connection()
    df = query_to_df(conn, "SELECT ActivityDate, TotalSteps, VeryActiveMinutes FROM daily_activity")
    conn.close()
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], infer_datetime_format=True).dt.normalize()
    return df

def get_user_stats_merged():
    conn = get_connection()
    activity = query_to_df(conn, """
        SELECT Id, AVG(TotalSteps) AS AvgSteps, AVG(Calories) AS AvgCalories,
               AVG(SedentaryMinutes) AS AvgSedentary
        FROM daily_activity GROUP BY Id
    """)
    sleep = query_to_df(conn, """
        SELECT Id, CAST(COUNT(*) AS FLOAT) / COUNT(DISTINCT logId) AS AvgSleepMinutes
        FROM minute_sleep GROUP BY Id
    """)
    conn.close()
    return activity.merge(sleep, on="Id", how="inner")

def get_user_daily_activity(user_id, start_date=None, end_date=None):
    conn = get_connection()
    df = query_to_df(conn, "SELECT * FROM daily_activity WHERE Id = ?", (user_id,))
    conn.close()
    if df.empty:
        return df
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], infer_datetime_format=True)
    if start_date:
        df = df[df["ActivityDate"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["ActivityDate"] <= pd.to_datetime(end_date)]
    return df.sort_values("ActivityDate")

def get_user_summary(user_id):
    conn = get_connection()
    df = query_to_df(conn, """
        SELECT COUNT(*) AS days_tracked, ROUND(AVG(TotalSteps), 0) AS avg_steps,
               ROUND(AVG(Calories), 0) AS avg_calories,
               ROUND(SUM(TotalDistance), 1) AS total_distance,
               ROUND(AVG(VeryActiveMinutes), 1) AS avg_very_active_min
        FROM daily_activity WHERE Id = ?
    """, (user_id,))
    conn.close()
    return df.iloc[0].to_dict()

def get_user_weight(user_id):
    conn = get_connection()
    df = query_to_df(conn, "SELECT Date, WeightKg, BMI, Fat FROM weight_log WHERE Id = ?", (user_id,))
    conn.close()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        df = df.sort_values("Date")
        df["Fat"] = df["Fat"].fillna(df["Fat"].mean())
    return df

def get_steps_calories_data():
    conn = get_connection()
    df = query_to_df(conn, "SELECT Id, TotalSteps, Calories FROM daily_activity")
    conn.close()
    return df

def get_user_hourly_merged(user_id):
    conn = get_connection()
    hr    = query_to_df(conn, "SELECT Time, Value FROM heart_rate WHERE Id = ?", (user_id,))
    steps = query_to_df(conn, "SELECT ActivityHour, StepTotal FROM hourly_steps WHERE Id = ?", (user_id,))
    conn.close()
    if hr.empty or steps.empty:
        return pd.DataFrame()
    hr["Hour"] = pd.to_datetime(hr["Time"], infer_datetime_format=True).dt.floor("h")
    hr_agg = hr.groupby("Hour")["Value"].mean().reset_index(name="AvgHeartRate")
    steps["Hour"] = pd.to_datetime(steps["ActivityHour"], infer_datetime_format=True)
    return steps.merge(hr_agg, on="Hour", how="inner")

def get_daily_sleep_and_activity():
    conn = get_connection()
    sleep_raw = query_to_df(conn, "SELECT Id, date AS RawDate FROM minute_sleep")
    activity  = query_to_df(conn, """
        SELECT Id, ActivityDate,
               VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes AS TotalActiveMinutes,
               SedentaryMinutes, TotalSteps, Calories
        FROM daily_activity
    """)
    conn.close()
    if sleep_raw.empty:
        return pd.DataFrame()
    sleep_raw["Id"] = sleep_raw["Id"].astype(int)
    sleep_raw["ActivityDate"] = parse_dates(sleep_raw["RawDate"])
    sleep_agg = sleep_raw.groupby(["Id", "ActivityDate"]).size().reset_index(name="SleepMinutes")
    activity["Id"] = activity["Id"].astype(int)
    activity["ActivityDate"] = parse_dates(activity["ActivityDate"])
    merged = sleep_agg.merge(activity, on=["Id", "ActivityDate"], how="inner")
    merged["ActivityDate"] = pd.to_datetime(merged["ActivityDate"])
    merged["IsWeekend"] = merged["ActivityDate"].dt.dayofweek >= 5
    return merged

def get_user_sleep_timeline(user_id):
    conn = get_connection()
    df = query_to_df(conn, "SELECT date AS RawDate FROM minute_sleep WHERE Id = ?", (user_id,))
    conn.close()
    if df.empty:
        return df
    df["SleepDate"] = pd.to_datetime(df["RawDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["SleepDate"])
    result = df.groupby("SleepDate").size().reset_index(name="SleepMinutes")
    result["SleepDate"] = pd.to_datetime(result["SleepDate"])
    return result.sort_values("SleepDate")

def add_blocks(df, time_col):
    df["Hour"] = pd.to_datetime(df[time_col], infer_datetime_format=True).dt.hour
    df["Block"] = pd.cut(df["Hour"], bins=[0, 4, 8, 12, 16, 20, 24],
                         labels=["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"], right=False)
    return df

def get_hourly_steps_by_block():
    conn = get_connection()
    df = query_to_df(conn, "SELECT ActivityHour, StepTotal FROM hourly_steps")
    conn.close()
    return add_blocks(df, "ActivityHour").groupby("Block", observed=True)["StepTotal"].mean().reset_index()


def get_hourly_calories_by_block():
    conn = get_connection()
    df = query_to_df(conn, "SELECT ActivityHour, Calories FROM hourly_calories")
    conn.close()
    return add_blocks(df, "ActivityHour").groupby("Block", observed=True)["Calories"].mean().reset_index()


def get_sleep_minutes_by_block():
    conn = get_connection()
    df = query_to_df(conn, "SELECT date AS SleepTime FROM minute_sleep")
    conn.close()
    return add_blocks(df, "SleepTime").groupby("Block", observed=True).size().reset_index(name="SleepMinutes")


def get_hourly_steps_all():
    conn = get_connection()
    df = query_to_df(conn, "SELECT ActivityHour, StepTotal FROM hourly_steps")
    conn.close()
    df["HourOfDay"] = pd.to_datetime(df["ActivityHour"], infer_datetime_format=True).dt.hour
    return df

def get_user_heart_rate(user_id, start_date=None, end_date=None):
    conn = get_connection()
    df = query_to_df(conn, "SELECT Time, Value AS HeartRate FROM heart_rate WHERE Id = ? LIMIT 5000", (user_id,))
    conn.close()
    if df.empty:
        return df
    df["Time"] = pd.to_datetime(df["Time"], infer_datetime_format=True)
    if start_date:
        df = df[df["Time"].dt.date >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df["Time"].dt.date <= pd.to_datetime(end_date).date()]
    return df

def get_user_hourly_intensity(user_id, start_date=None, end_date=None):
    conn = get_connection()
    df = query_to_df(conn, "SELECT ActivityHour, TotalIntensity FROM hourly_intensity WHERE Id = ?", (user_id,))
    conn.close()
    if df.empty:
        return df
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], infer_datetime_format=True)
    if start_date:
        df = df[df["ActivityHour"].dt.date >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df["ActivityHour"].dt.date <= pd.to_datetime(end_date).date()]
    return df.sort_values("ActivityHour")