import os, sys, importlib.util
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data.py')
_spec = importlib.util.spec_from_file_location("data", _data_path)
data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf

st.set_page_config(
    page_title="sleep Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="auto",
)
st.logo(image="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Fitbit_logo16.svg/3840px-Fitbit_logo16.svg.png", size="large")

st.sidebar.markdown("---")
user_ids = data.get_all_user_ids()
selected_user = st.sidebar.selectbox("Highlight User", ["All"] + [str(u) for u in user_ids])

st.title("😴 Sleep Analysis")
st.markdown("Does activity level, sedentary time, or the day of week affect how long users sleep?")

df = data.get_daily_sleep_and_activity()
if df.empty:
    st.warning("Sleep data could not be loaded.")
    st.stop()

# metrics
avg_sleep = df["SleepMinutes"].mean()
median_sleep = df["SleepMinutes"].median()
avg_weekday = df[~df["IsWeekend"]]["SleepMinutes"].mean()
avg_weekend = df[df["IsWeekend"]]["SleepMinutes"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Sleep",f"{avg_sleep/60:.1f} hrs")
col2.metric("Median Sleep", f"{median_sleep/60:.1f} hrs")
col3.metric("Avg Weekday Sleep",f"{avg_weekday/60:.1f} hrs")
col4.metric("Avg Weekend Sleep",f"{avg_weekend/60:.1f} hrs")

# distribution
st.subheader("Sleep distribution")
col1, col2 = st.columns(2)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(df["SleepMinutes"] / 60, bins=25, color="#52dee3", edgecolor="black", linewidth=0.5)
ax.axvline(avg_sleep / 60, color="red", linewidth=1, linestyle="--",
           label=f"Mean: {avg_sleep/60:.1f}h")
ax.set_xlabel("Sleep duration (hours)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of sleep duration", fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col1.pyplot(fig)
plt.close(fig)

weekday_sleep = (
    df.assign(Weekday=df["ActivityDate"].dt.day_name())
    .groupby("Weekday")["SleepMinutes"].mean()
    .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    / 60)

fig, ax = plt.subplots(figsize=(5, 3.5))
colors = ["#e94560" if d in ["Saturday", "Sunday"] else "#52dee3" for d in weekday_sleep.index]
ax.bar(weekday_sleep.index, weekday_sleep.values, color=colors)
ax.set_ylabel("Avg sleep (hours)")
ax.set_title("Average sleep by weekday", fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.axhline(7, color="#e94560", linewidth=1, linestyle=":", alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col2.pyplot(fig)
plt.close(fig)
diff = avg_weekend - avg_weekday

# sleep vs active minutes
st.subheader("Sleep vs active minutes")
model_active = smf.ols("SleepMinutes ~ TotalActiveMinutes", data=df).fit()
col1, col2   = st.columns([2, 1])
highlight = selected_user != "All"
uid = int(selected_user) if highlight else None
mask = df["Id"] == uid if highlight else pd.Series([False] * len(df))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df.loc[~mask, "TotalActiveMinutes"], df.loc[~mask, "SleepMinutes"] / 60,
           color="#aaaaaa", alpha=0.4, s=20, label="All users")
if highlight:
    ax.scatter(df.loc[mask, "TotalActiveMinutes"], df.loc[mask, "SleepMinutes"] / 60,
               color="#e94560", s=50, label=f"User {uid}")
x_range = np.linspace(df["TotalActiveMinutes"].min(), df["TotalActiveMinutes"].max(), 200)
y_pred  = model_active.params["Intercept"] + model_active.params["TotalActiveMinutes"] * x_range
ax.plot(x_range, y_pred / 60, color="#0f3460", linewidth=2, label="Regression")
ax.set_xlabel("Total active minutes")
ax.set_ylabel("Sleep duration (hours)")
ax.set_title("Active minutes vs sleep duration", fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col1.pyplot(fig)
plt.close(fig)

coef = model_active.params["TotalActiveMinutes"]
pval = model_active.pvalues["TotalActiveMinutes"]
col2.markdown("**OLS Results**")
col2.metric("R²",f"{model_active.rsquared:.3f}")
col2.metric("Active min coefficient",f"{coef:.3f}")
col2.metric("p-value",f"{pval:.4f}")
direction = "negatively" if coef < 0 else "positively"
sig = "significant" if pval < 0.05 else "not significant"

# sleep vs sedentary
st.subheader("Sleep vs sedentary minutes")
model_sed = smf.ols("SleepMinutes ~ SedentaryMinutes", data=df).fit()
residuals = model_sed.resid
col1, col2, col3 = st.columns(3)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.scatter(df["SedentaryMinutes"], df["SleepMinutes"] / 60,
           color="#1a1a2e", alpha=0.35, s=18)
x_r = np.linspace(df["SedentaryMinutes"].min(), df["SedentaryMinutes"].max(), 200)
y_r = (model_sed.params["Intercept"] + model_sed.params["SedentaryMinutes"] * x_r) / 60
ax.plot(x_r, y_r, color="red", linewidth=2)
ax.set_xlabel("Sedentary minutes")
ax.set_ylabel("Sleep (hours)")
ax.set_title("Sedentary vs sleep", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col1.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.scatter(model_sed.fittedvalues, residuals, color="black", alpha=0.4, s=18)
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs fitted", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col2.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
ax.scatter(osm, osr, color="#1a1a2e", alpha=0.5, s=18)
ax.plot(osm, slope * np.array(osm) + intercept, color="#e94560", linewidth=2)
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("Q-Q plot (normality check)", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col3.pyplot(fig)
plt.close(fig)

# individual sleep timeline
st.subheader("Individual sleep timeline")
if selected_user != "All":
    sleep_tl = data.get_user_sleep_timeline(int(selected_user))
    if not sleep_tl.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(sleep_tl["SleepDate"], sleep_tl["SleepMinutes"] / 60,
                color="#e94560", linewidth=2, marker="o", markersize=4)
        ax.axhline(7, color="#1a1a2e", linestyle="--", linewidth=1, label="7h target")
        ax.set_ylabel("Sleep (hours)")
        ax.set_title(f"Sleep timeline — User {selected_user}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No sleep data available for this user.")
else:
    st.info("Select a specific user in the sidebar to see their sleep timeline!!!")