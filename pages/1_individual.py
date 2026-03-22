import os, sys, importlib.util
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data.py')
_spec = importlib.util.spec_from_file_location("data", _data_path)
data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import numpy as np

st.set_page_config(
    page_title="Individual Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="auto",
)
st.logo(image="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Fitbit_logo16.svg/3840px-Fitbit_logo16.svg.png", size="large")


user_ids = data.get_all_user_ids()
user_id = st.sidebar.selectbox("select user ID", user_ids)
st.sidebar.markdown("---")
st.sidebar.markdown("**Date range**")
start_date = st.sidebar.date_input("From", value=pd.to_datetime("2016-01-01"))
end_date = st.sidebar.date_input("To", value=pd.to_datetime("2016-12-31"))

start_str = str(start_date)
end_str = str(end_date)

df = data.get_user_daily_activity(user_id, start_str, end_str)
summary = data.get_user_summary(user_id)
classes = data.get_user_classes().set_index("Id")

st.title(f"👤 User {user_id}")
user_class = classes.loc[user_id, "Class"] if user_id in classes.index else "—"
st.markdown(f"**Classification:** `{user_class}` · {start_date} → {end_date}")

if df.empty:
    st.warning("No data found for this user in the selected date range.")
    st.stop()

# metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Days tracked",int(summary["days_tracked"]), border=True)
col2.metric("Average daily steps",f"{int(summary['avg_steps'])}", border=True)
col3.metric("Average daily calories",f"{int(summary['avg_calories'])}",  border=True)
col4.metric("Total distance",f"{summary['total_distance']} km", border=True)
col5.metric("Average very active",f"{summary['avg_very_active_min']} min", border=True)

# calories over time
st.subheader("Calories burned over time")
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df["ActivityDate"], df["Calories"], color="#52dee3", linewidth=2)
ax.fill_between(df["ActivityDate"], df["Calories"], alpha=0.15, color="#52dee3")
ax.axhline(df["Calories"].mean(), color="#e94560", linewidth=1, linestyle="--",label=f"Avg: {int(df['Calories'].mean())} kcal")
max_idx = df["Calories"].idxmax()
ax.scatter(df.loc[max_idx, "ActivityDate"], df.loc[max_idx, "Calories"],color="#e94560", s=60)
ax.annotate(f"{int(df.loc[max_idx, 'Calories']):,}",
            xy=(df.loc[max_idx, "ActivityDate"], df.loc[max_idx, "Calories"]),
            xytext=(8, 6), textcoords="offset points", fontsize=8, color="#e94560")
ax.set_ylabel("Calories")
ax.set_title(f"Daily calories — User {user_id}", fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_ylim(bottom=df["Calories"].min() * 0.85)
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=30)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# steps and activity
st.subheader("Steps & activity breakdown")
col1, col2 = st.columns(2)
colors = ["#e94560" if v == df["TotalSteps"].max() else "#52dee3" for v in df["TotalSteps"]]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(df["ActivityDate"], df["TotalSteps"], color=colors, width=0.8, alpha=0.8)
ax.axhline(df["TotalSteps"].mean(), color="red", linewidth=1, linestyle="--",label=f"Avg: {int(df['TotalSteps'].mean()):,} steps")
ax.set_ylabel("Steps")
ax.set_title("Daily steps", fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_ylim(bottom=0)
ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=7)
plt.xticks(rotation=30)
fig.tight_layout()
col1.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3))
ax.stackplot(df["ActivityDate"],
             df["VeryActiveMinutes"],df["FairlyActiveMinutes"],df["LightlyActiveMinutes"], df["SedentaryMinutes"],
             labels=["Very Active", "Fairly Active", "Lightly Active", "Sedentary"],
             colors=["#e94560", "#f4a261", "#52dee3", "#2a2a3e"], alpha=0.85)
ax.set_ylabel("Minutes")
ax.set_title("Activity breakdown", fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.legend(fontsize=6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.35))
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=7)
plt.xticks(rotation=30)
fig.tight_layout()
col2.pyplot(fig)
plt.close(fig)

# regression
st.subheader("Relationship between calories and steps taken")
all_df = data.get_steps_calories_data()

y = all_df["Calories"]
X = all_df[["TotalSteps"]].copy()
X = pd.concat([X, pd.get_dummies(all_df["Id"], prefix="Id", drop_first=True)], axis=1)
X = sm.add_constant(X).astype(float)
model = sm.OLS(y, X).fit()
user_df   = all_df[all_df["Id"] == user_id].copy()
slope     = model.params["TotalSteps"]
intercept = model.params["const"]
id_effect = model.params.get(f"Id_{user_id}", 0)
x_range   = np.linspace(user_df["TotalSteps"].min(), user_df["TotalSteps"].max(), 200)
y_pred    = intercept + id_effect + slope * x_range

col1, col2 = st.columns([2, 1])
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(user_df["TotalSteps"], user_df["Calories"],color="#1a1a2e", alpha=0.6, s=40, label="Observations")
ax.plot(x_range, y_pred, color="#e94560", linewidth=2, label="Regression line")
ax.set_xlabel("Total Steps")
ax.set_ylabel("Calories")
ax.set_title(f"Steps vs calories — User {user_id}", fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col1.pyplot(fig)
plt.close(fig)

col2.markdown("**Model summary**")
col2.metric("R²",f"{model.rsquared:.3f}")
col2.metric("Steps coefficient",f"{slope:.4f}")
col2.metric("ID effect",f"{id_effect:.1f}")
col2.caption("Model: Calories = β0 + β1·TotalSteps + β2·Id fitted on all users.")
with st.expander("Full model summary"):
    st.text(model.summary().as_text())

st.subheader("Last recorded day vs averages")
last_cal = user_df["Calories"].iloc[-1]
user_avg = user_df["Calories"].mean()
overall_avg = all_df["Calories"].mean()
col1, col2, col3 = st.columns(3)
col1.metric("Last day calories",f"{int(last_cal):,}")
col2.metric("Your average",f"{int(user_avg):,}",
            delta=f"{int(last_cal - user_avg):+,} vs your avg")
col3.metric("Everyone's average", f"{int(overall_avg):,}",
            delta=f"{int(last_cal - overall_avg):+,} vs all users")

# steps vs heart rate
st.subheader("Steps vs heart rate")
hr_df = data.get_user_hourly_merged(user_id)
if not hr_df.empty:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(hr_df["StepTotal"], hr_df["AvgHeartRate"], alpha=0.3, color="#52dee3", s=20)
    z = np.polyfit(hr_df["StepTotal"], hr_df["AvgHeartRate"], 1)
    x_line = np.linspace(hr_df["StepTotal"].min(), hr_df["StepTotal"].max(), 200)
    ax.plot(x_line, np.poly1d(z)(x_line), color="#e94560", linewidth=2)
    ax.set_xlabel("Steps per hour")
    ax.set_ylabel("Avg heart rate (bpm)")
    ax.set_title(f"Steps vs heart rate — User {user_id}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No heart rate data available for this user.")

# weight and bmi
weight = data.get_user_weight(user_id)
if not weight.empty:
    st.subheader("Weight & BMI")
    col1, col2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(weight["Date"], weight["WeightKg"], marker="o", color="#e94560", linewidth=2)
    ax.set_ylabel("Weight (kg)")
    ax.set_title("Weight over time", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30)
    fig.tight_layout()
    col1.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(weight["Date"], weight["BMI"], marker="o", color="#1a1a2e", linewidth=2)
    ax.axhline(25, color="#e94560", linestyle="--", linewidth=1, label="Overweight (25)")
    ax.set_ylabel("BMI")
    ax.set_title("BMI over time", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30)
    fig.tight_layout()
    col2.pyplot(fig)
    plt.close(fig)
else:
    st.info("Weight data not available for this user.")

# heart rate and intensity
st.subheader("Heart rate & exercise intensity")
hr = data.get_user_heart_rate(user_id, start_str, end_str)
intensity = data.get_user_hourly_intensity(user_id, start_str, end_str)

if not hr.empty and not intensity.empty:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    ax1.plot(hr["Time"], hr["HeartRate"], color="#e94560", linewidth=0.8, alpha=0.8)
    ax1.set_ylabel("Heart rate (bpm)")
    ax1.set_title("Heart rate", fontweight="bold")
    ax2.bar(intensity["ActivityHour"], intensity["TotalIntensity"], color="#1a1a2e", width=0.04)
    ax2.set_ylabel("Total intensity")
    ax2.set_title("Hourly exercise intensity", fontweight="bold")
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Heart rate or intensity data not available for this user / date range.")