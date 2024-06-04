import pandas as pd
from prophet import Prophet # type: ignore
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('timestamps.csv')
print(df.head())
df.columns = ['ds','y']

df['ds'] = pd.to_datetime(df['ds'])
print(df.head())

# Train the Prophet model
prophet_model = Prophet(changepoint_prior_scale=0.01)  # Adjust changepoint_prior_scale as needed
# prophet_model.add_seasonality(name='daily', period=1, fourier_order=3)  # Add daily seasonality
prophet_model.fit(df)

# Extract ground truth data from the original DataFrame
ground_truth_data = df[['ds', 'y']].copy()
# Plot the ground truth data
plt.figure()
plt.plot(ground_truth_data['ds'], ground_truth_data['y'], label='Ground Truth')
plt.xlabel('Time')
plt.ylabel('Action ID')
plt.title('Ground Truth Action IDs')
plt.legend()

plt.show()


# Make future predictions (if needed)
# future = prophet_model.make_future_dataframe(periods=1*60, freq='T', include_history=False)
future = prophet_model.make_future_dataframe(periods=600, freq='5T')
print(future.head())
forecast = prophet_model.predict(future)

# Print the forecasted values
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
# fig = prophet_model.plot(forecast)
# plt.xlabel('Time')
# plt.ylabel('Action ID')
# plt.title('Forecasted Action IDs')
# plt.show()
# # plt.show()

# Plot the forecast
fig = prophet_model.plot(forecast)
plt.xlabel('Time')
plt.ylabel('Action ID')
plt.title('Forecasted Action IDs')
plt.legend()

# # Show the plot
# plt.show()
plt.show()


# Save the trained model
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(prophet_model, f)