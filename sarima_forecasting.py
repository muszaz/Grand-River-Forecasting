#!/usr/bin/env python
# coding: utf-8

# In[52]:


import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ## Grand River SARIMAX Forecast
# 
# SARIMAX = SARIMA + eXogenous variables
# 
# Instead of just looking at Galt's historical flow, we use Montrose (upstream) 
# as an exogenous input. Water flowing through Montrose today helps predict 
# Galt's flow downstream.
# 
# This should perform better than plain SARIMA because it captures the actual 
# physical relationship in the river system.

# In[53]:


conn = sqlite3.connect('Hydat.sqlite3')

def get_station_flow(conn, station_number, start_date='2000-01-01'):
    """
    Pull monthly mean flow for a single HYDAT station.
    """
    query = """
        SELECT YEAR, MONTH, MONTHLY_MEAN
        FROM DLY_FLOWS
        WHERE STATION_NUMBER = ?
    """
    station_df = pd.read_sql_query(query, conn, params=(station_number,))

    station_df['date'] = pd.to_datetime(station_df[['YEAR', 'MONTH']].assign(DAY=1))
    station_df = station_df.set_index('date').sort_index()

    station_df['MONTHLY_MEAN'] = station_df['MONTHLY_MEAN'].interpolate(method='time')

    series = station_df.loc[start_date:, 'MONTHLY_MEAN']
    return series.resample('ME').mean()


# HYDAT stores one row per station per month, with MONTHLY_MEAN already 
# calculated. A handful of months are missing values (NaN). Since this is 
# a small fraction of the data, missing values are interpolated using 
# neighbouring months rather than dropped, which keeps the time series 
# continuous for SARIMA.

# In[54]:


montrose = get_station_flow(conn, '02GA034')
galt = get_station_flow(conn, '02GA003')

print(f"Montrose: {len(montrose)} months")
print(f"Galt: {len(galt)} months")


# In[55]:


# Load climate data
climate = pd.read_csv('en_climate_daily_ON_6143092_2026_P1D.csv')

# Parse date and create proper index
climate['date'] = pd.to_datetime(climate[['Year', 'Month', 'Day']])
climate = climate.set_index('date')

# Convert columns to numeric (some might be 'M' for missing)
climate['Mean Temp (°C)'] = pd.to_numeric(climate['Mean Temp (°C)'], errors='coerce')
climate['Total Precip (mm)'] = pd.to_numeric(climate['Total Precip (mm)'], errors='coerce')

# Aggregate to monthly: avg temp, total precip
monthly_climate = climate.resample('ME').agg({
    'Mean Temp (°C)': 'mean',
    'Total Precip (mm)': 'sum'
})

monthly_climate = monthly_climate.dropna()

print(f"Climate data: {len(monthly_climate)} months, from {monthly_climate.index[0]} to {monthly_climate.index[-1]}")
monthly_climate.head()


# In[39]:


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

montrose.plot(ax=axes[0], title='Upstream: Grand River at Montrose', color='blue', linewidth=2)
galt.plot(ax=axes[1], title='Downstream: Grand River at Galt', color='green', linewidth=2)

for ax in axes:
    ax.set_ylabel('Flow (m³/s)')

plt.tight_layout()
plt.show()


# 

# In[40]:


result_montrose = adfuller(montrose.dropna())
result_galt = adfuller(galt.dropna())

print(f"Montrose ADF p-value: {result_montrose[1]:.4f}")
print(f"Galt ADF p-value: {result_galt[1]:.4f}")

if result_galt[1] < 0.05:
    print("Galt is stationary")
else:
    print("Galt needs differencing")


# SARIMA assumes the series is stationary — its statistical properties
# don't drift over time. The Augmented Dickey-Fuller test checks this.
# A p-value below 0.05 means the series is stationary as-is; above that,
# differencing (handled by the `d` parameter later) is needed.

# In[60]:


fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(galt.dropna(), lags=40, ax=axes[0], title='ACF')
plot_pacf(galt.dropna(), lags=40, ax=axes[1], title='PACF')
plt.tight_layout()
plt.show()


# These plots help identify candidate AR and MA parameters by showing how
# correlated the series is with its own past values. They're used as a
# starting point — the parameter search below explores a range
# automatically.

# In[42]:


print("Searching for optimal SARIMAX parameters...")

auto_model = pm.auto_arima(
    galt,
    exogenous=montrose,
    start_p=0, start_q=0, max_p=3, max_q=3, d=1,
    start_P=0, start_Q=0, max_P=2, max_Q=2, D=1,
    m=12, seasonal=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(f"\nBest order: {auto_model.order}")
print(f"Best seasonal order: {auto_model.seasonal_order}")


# In[43]:


# Drop any rows with NaN in either series
aligned_data = pd.DataFrame({
    'galt': galt,
    'montrose': montrose
}).dropna()

aligned_galt = aligned_data['galt']
aligned_montrose = aligned_data['montrose']

print(f"After removing NaNs: {len(aligned_galt)} months")

# Split into train/test
train_galt = aligned_galt.iloc[:-12]
test_galt = aligned_galt.iloc[-12:]

train_montrose = aligned_montrose.iloc[:-12]
test_montrose = aligned_montrose.iloc[-12:]

val_model = SARIMAX(train_galt, exog=train_montrose, 
                    order=auto_model.order, seasonal_order=auto_model.seasonal_order)
val_results = val_model.fit(disp=False)

val_forecast = val_results.get_forecast(steps=12, exog=test_montrose).summary_frame()

rmse = np.sqrt(mean_squared_error(test_galt, val_forecast['mean']))
mae = mean_absolute_error(test_galt, val_forecast['mean'])

print(f"Validation MAE: {mae:.2f} m³/s")
print(f"Validation RMSE: {rmse:.2f} m³/s")
print(f"Historical average Galt flow: {aligned_galt.mean():.2f} m³/s")


# Rather than manually testing every combination of SARIMA parameters,
# `auto_arima` searches a range of (p, d, q) and seasonal (P, D, Q, m)
# values and selects the combination with the best AIC score. The seasonal
# period `m=12` reflects the yearly cycle seen in the seasonality plot.
# 

# Before trusting the forecast, the model needs to be validated against
# data it hasn't seen. The last 12 months are held out as a test set; the
# model is trained on everything before that and asked to "predict" the
# withheld months. Comparing the prediction to what actually happened
# gives an honest error estimate.

# In[51]:


production_model = SARIMAX(aligned_galt, exog=aligned_montrose, 
                          order=auto_model.order, seasonal_order=auto_model.seasonal_order)
production_results = production_model.fit(disp=False)

recent_avg = aligned_montrose.tail(12).mean()
montrose_future = pd.DataFrame({'montrose': [recent_avg] * 12})

forecast_output = production_results.get_forecast(steps=12, exog=montrose_future.values)
forecast_df = forecast_output.summary_frame()
forecast_df['mean_ci_lower_clipped'] = forecast_df['mean_ci_lower'].clip(lower=0)

forecast_df[['mean', 'mean_ci_lower_clipped', 'mean_ci_upper']]


# With the parameters validated, the model is refit on the full historical
# series to produce the actual forecast. Confidence interval lower
# bounds are clipped at zero since negative streamflow isn't physically
# possible, the raw statistical interval can occasionally dip below zero,
# which this corrects.

# In[46]:


production_results.plot_diagnostics(figsize=(14, 8))
plt.show()


# To check how the model performs against reality, Environment Canada
# publishes near-real-time daily flow readings for active stations. This
# pulls the live feed for Galt and resamples it to daily means.

# This compares the SARIMAX forecast for the most recent months against the
# actual observed flow over the same period

# In[63]:


fig, ax = plt.subplots(figsize=(14, 6))

# 1. Grab the last observed date and value to bridge the gap
last_date = aligned_galt.index[-1]
last_value = aligned_galt.iloc[-1]

# 2. Stitch that last point to a temporary forecast dataframe for plotting
plot_forecast = forecast_df.copy()
plot_forecast.loc[last_date] = {
    'mean': last_value, 
    'mean_ci_lower_clipped': last_value, 
    'mean_ci_upper': last_value
}
plot_forecast = plot_forecast.sort_index()

# 3. Plot the historical data
aligned_galt.iloc[-36:].plot(ax=ax, label='Observed', linewidth=2, color='black')

# 4. Plot the newly stitched forecast data
plot_forecast['mean'].plot(ax=ax, label='Forecast', linewidth=2, color='red')

# 5. Fill the confidence interval (now starting from a point of 0 uncertainty)
ax.fill_between(
    plot_forecast.index,
    plot_forecast['mean_ci_lower_clipped'],
    plot_forecast['mean_ci_upper'],
    alpha=0.3, color='red', label='95% confidence interval'
)

ax.set_title('Grand River at Galt SARIMAX Forecast')
ax.set_ylabel('Flow (m³/s)')
ax.set_xlabel('Date')
ax.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[64]:


recent_galt = aligned_galt.iloc[-6:]
recent_montrose = aligned_montrose.iloc[-6:]

retrospective_forecast = production_results.get_forecast(steps=6, exog=recent_montrose.values.reshape(-1, 1)).summary_frame()

comparison_df = pd.DataFrame({
    'Actual': recent_galt.values,
    'Predicted': retrospective_forecast['mean'].values[:len(recent_galt)]
})

comparison_df['Error'] = abs(comparison_df['Actual'] - comparison_df['Predicted'])
comparison_df['Error_Percent'] = (comparison_df['Error'] / comparison_df['Actual'] * 100).round(2)

print("Last 6 months: Actual vs. SARIMAX Forecast")
print(comparison_df.to_string())

print(f"\nMean absolute error (MAE): {comparison_df['Error'].mean():.2f} m³/s")
print(f"Mean absolute percent error (MAPE): {comparison_df['Error_Percent'].mean():.2f}%")


# In[65]:


fig, ax = plt.subplots(figsize=(14, 6))

# Actual observations
test_galt.plot(
    ax=ax,
    color='green',
    linewidth=2,
    marker='o',
    markersize=6,
    label='Actual Flow'
)

# Predicted observations
val_forecast['mean'].plot(
    ax=ax,
    color='red',
    linewidth=2,
    marker='s',
    markersize=6,
    label='Predicted Flow'
)

ax.set_title('SARIMAX Validation: Actual vs. Predicted Monthly Streamflow')
ax.set_xlabel('Date')
ax.set_ylabel('Flow (m³/s)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[50]:


conn.close()


