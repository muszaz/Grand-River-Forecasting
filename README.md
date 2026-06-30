# Grand-River-Forecasting
This project forecasts monthly streamflow for the Grand River at the Galt monitoring station using a SARIMAX (Seasonal Autoregressional Moving Average with Exogynous Variables) Model.

The objective was to determine whether incorporating upstream river discharge improves forecasting accuracy compared with a standard SARIMA model that relies only on historical observations.

Using monthly discharge data from Environment and Climate Change Canada's HYDAT database, the SARIMAX model significantly outperformed the baseline SARIMA model by incorporating measurements from an upstream monitoring station.

Key Results entage Error)
| Model | MAE (Mean Absolute Error) | MAPE (Mean Absolute Percentage Error) |
|------|---------------------------:|--------------------------------------:|
| SARIMA | 33 m³/s | 227% |
| SARIMAX | 4.7 m³/s | 31% |

Adding upstream discharge as an exogenous variable reduced forecasting error substantially, demonstrating that incorporating physically meaningful predictors can greatly improve statistical forecasts.

Dataset

Source

Environment and Climate Change Canada – HYDAT Database

Stations

02GA034 – Grand River at West Montrose (upstream)
02GA003 – Grand River at Galt (downstream)

Time Period

2000–2026
Monthly mean discharge (m³/s)

Missing monthly values were interpolated using time-based interpolation before the data were aligned and resampled into continuous monthly time series.

## Methodology
The forecasting pipeline consists of the following steps:

Query monthly discharge data from the HYDAT SQLite database.
Clean and interpolate missing observations.
Align upstream and downstream time series.
Test stationarity using the Augmented Dickey-Fuller (ADF) test.
Automatically select SARIMAX parameters using pmdarima.auto_arima().
Train the model using upstream discharge as an exogenous predictor.
Validate performance using a 12-month holdout dataset.
Generate a 12-month forecast with confidence intervals.

The model predicts downstream flow at Galt while using simultaneous upstream discharge measurements from West Montrose as additional explanatory information.
## Python Libraries Used
pandas
NumPy
Matplotlib
statsmodels
pmdarima
scikit-learn
SQLite

## Validation

To evaluate performance, the final twelve months of observations were withheld from training.

The model was trained on the remaining historical record and then used to predict the unseen period. Forecast accuracy was evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Mean Absolute Percentage Error (MAPE)

This provides an unbiased estimate of how the model performs on future observations rather than simply measuring how well it fits historical data.

## Future Plans
In the future, I plan on adding other exogynous variables, like precipitaion and temperature to the model to hopefully further reduce the error. I also will add a time lag between the two stations. My end goal of the project will be to have an interactive map of sorts, where you can click on a station on a river, predict the streamflow for each one, and compare it to realtime data.

## How to Run

Install the required packages:

pandas numpy matplotlib statsmodels pmdarima scikit-learn

Download the HYDAT SQLite database from Environment and Climate Change Canada and place Hydat.sqlite3 in the project directory.

Run the notebook sequentially to reproduce the analysis, validation, and forecasts.

##Author
Mustafa Zazai
Water Resources Engineering Student at the University of Guelph
June 2026
