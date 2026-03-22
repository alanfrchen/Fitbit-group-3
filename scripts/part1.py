from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from datetime import timedelta
from numpy import random
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt 


database = pd.read_csv("daily_activity.csv")
unique_users = database["Id"].nunique()
print("Number of unique users:", unique_users)

total_distance_by_user = database.groupby("Id", as_index=False)["TotalDistance"].sum()
total_distance_by_user = total_distance_by_user.sort_values(by="TotalDistance")
print(total_distance_by_user)
plt.bar(total_distance_by_user["Id"].astype(str),
        total_distance_by_user["TotalDistance"])

plt.xlabel("User ID")
plt.ylabel("Total Distance")
plt.title("Total distance per user")

plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

database["ActivityDate"] = pd.to_datetime(database["ActivityDate"], errors="coerce")

def user_friendly(user_id, start_date=None, end_date=None):
    df = database.loc[database["Id"] == user_id, ["ActivityDate", "Calories"]].dropna()

    if df.empty:
        print(f"No data found for user {user_id}")
        return

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df["ActivityDate"] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df["ActivityDate"] <= end_date]

    if df.empty:
        print(f"No data for user {user_id} in the selected date range.")
        return

    df = df.sort_values("ActivityDate")

    plt.figure(figsize=(10, 5))
    plt.plot(df["ActivityDate"], df["Calories"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Calories burnt")
    plt.title(f"Calories burnt per day by user {user_id}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def lazy_final_day():
    ids = database["Id"].unique()
    lasts = []
    for id in ids:
        last = database.loc[database["Id"] == id, "Calories"].to_list()[-1]
        lasts.append(last)
    lasts_averages = sum(lasts)/len(lasts) 
    overall_average = database["Calories"].mean()
    plt.figure(figsize=(10,5))
    plt.bar(ids.astype(str), lasts)
    plt.axhline(y=lasts_averages, color='red', linestyle=':', label = f"average calories burnt last day ({round(lasts_averages,1)})")
    plt.axhline(y=overall_average, color='green', linestyle=':', label = f"average calories burnt in general ({round(overall_average,1)})")
    plt.xlabel("User Id")
    plt.ylabel("Calories burnt")
    plt.title("Calories burnt final day of user activity")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()

lazy_final_day()
user_friendly(1927972279, start_date="2016-04-01", end_date="2016-04-11")

database = pd.read_csv("daily_activity.csv")
unique_users = database["Id"].nunique()
print("Number of unique users:", unique_users)

def calc_cal(database):
    y = database['Calories']
    TotalSteps = database[['TotalSteps']].copy()
    Id_placeholder = pd.get_dummies(database['Id'], prefix='Id', drop_first=True) #1 if user 0 if not
    TotalSteps = pd.concat([TotalSteps, Id_placeholder], axis = 1)
    TotalSteps = sm.add_constant(TotalSteps)
    TotalSteps = TotalSteps.astype(float)
    model = sm.OLS(y , TotalSteps).fit()
    print(model.summary())
    return model
calc_cal(database)
    
