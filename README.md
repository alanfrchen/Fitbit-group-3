# Fitbit-group-3
## Part 1
The primary goal of this analysis is to get acquainted with survey data reported daily by Fitbit users. We perform computations and visualizations to understand user behavior and the effectiveness of tracking steps to estimate calories burnt.

## Data Inspection
The script performs an initial scan of the dataset to establish the scope of the survey:

* **User Identification**: Calculates the total number of unique participants in the study using .nunique().

* **Distance Tracking**: Aggregates the TotalDistance registered for every user to compare activity levels across the cohort.

## Key Functions
The implementation includes specific functions to visualize data at both the individual and group levels:

### `user_friendly(user_id)`
Generates a line graph showing the daily calories burnt for a specific user.

Visuals: Plots ActivityDate on the x-axis against Calories on the y-axis.

### `lazy_sunday()`
A custom analysis that isolates the final recorded activity day for every user in the dataset.

Visuals: Displays a bar chart comparing each user's last-day caloric burn against the group average (marked with a red dashed line).
