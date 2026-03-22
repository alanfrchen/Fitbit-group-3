import os, sys, importlib.util
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data.py')
_spec = importlib.util.spec_from_file_location("data", _data_path)
data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Time of day Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="auto",
)
st.logo(image="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Fitbit_logo16.svg/3840px-Fitbit_logo16.svg.png", size="large")

st.sidebar.markdown("---")
st.sidebar.markdown("Day divided into 4 hour blocks.")
st.sidebar.markdown("Values are **averages across all participants**.")

st.title("🕐 Time of Day")
st.markdown("How does activity, calorie burn, and sleep vary throughout the day?")

blocks = ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]
colors    = ["#e94560", "#c73652", "#0f3460", "#16213e", "#1a1a2e", "#0d0d1a"]

steps    = data.get_hourly_steps_by_block()
calories = data.get_hourly_calories_by_block()
sleep    = data.get_sleep_minutes_by_block()

# steps by block
st.subheader("Average steps per 4 hour block")
fig, ax = plt.subplots(figsize=(10, 3.5))
bar_colors = ["#e94560" if v == steps["StepTotal"].max() else "#52dee3" for v in steps["StepTotal"]]
bars = ax.bar(range(len(steps)), steps["StepTotal"], color=bar_colors, edgecolor="none")
ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
ax.set_xticks(range(len(steps)))
ax.set_xticklabels(blocks, fontsize=9)
ax.set_ylabel("Avg steps")
ax.set_ylim(0, steps["StepTotal"].max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# calories and sleep side by side
col1, col2 = st.columns(2)

fig, ax = plt.subplots(figsize=(5, 3.5))
bar_colors = ["#e94560" if v == calories["Calories"].max() else "#52dee3" for v in calories["Calories"]]
bars = ax.bar(range(len(calories)), calories["Calories"], color=bar_colors, edgecolor="none")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
ax.set_xticks(range(len(calories)))
ax.set_xticklabels(blocks, fontsize=8)
ax.set_ylabel("Avg calories")
ax.set_title("Calories burned by block", fontweight="bold")
ax.set_ylim(0, calories["Calories"].max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col1.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.5))
bar_colors = ["#e94560" if v == sleep["SleepMinutes"].max() else "#52dee3" for v in sleep["SleepMinutes"]]
bars = ax.bar(range(len(sleep)), sleep["SleepMinutes"], color=bar_colors, edgecolor="none")
ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
ax.set_xticks(range(len(sleep)))
ax.set_xticklabels(blocks, fontsize=8)
ax.set_ylabel("Sleep minutes")
ax.set_title("Sleep minutes by block", fontweight="bold")
ax.set_ylim(0, sleep["SleepMinutes"].max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
col2.pyplot(fig)
plt.close(fig)

# steps distribution by hour
st.subheader("Average steps by hour of day")
steps_hourly = data.get_hourly_steps_all()
hourly_avg   = steps_hourly.groupby("HourOfDay")["StepTotal"].mean()

fig, ax = plt.subplots(figsize=(10, 3.5))
bar_colors = ["#e94560" if v == hourly_avg.max() else "#52dee3" for v in hourly_avg]
bars = ax.bar(hourly_avg.index, hourly_avg.values, color=bar_colors, edgecolor="none")
ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=7)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Avg steps")
ax.set_title("Average steps by hour of day", fontweight="bold")
ax.set_xticks(range(24))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Normalised comparison across blocks")
def normalise(array):
    min, max = array.min(), array.max()
    return (array - min) / (max - min)

x = np.arange(len(blocks))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, normalise(steps["StepTotal"].values),marker="o", linewidth=2, label="Steps", color="#52dee3")
ax.plot(x, normalise(calories["Calories"].values),marker="o", linewidth=2, label="Calories", color="#1a1a2e")
ax.plot(x, normalise(sleep["SleepMinutes"].values),marker="o", linewidth=2, label="Sleep", color="#e94560")
ax.set_xticks(x)
ax.set_xticklabels(blocks, fontsize=9)
ax.set_ylabel("Normalised value (0-1)")
ax.set_title("Activity and Sleep chart line normalised", fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)