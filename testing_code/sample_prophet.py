import pandas as pd
import pymongo
from prophet import Prophet
from matplotlib import pyplot as plt
import numpy as np
import pickle
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["siddhant1"]
collection = db["face_data"]

# Query the database to retrieve timestamps and action IDs
cursor = collection.find({}, {"_id": 0, "actions": 1})
timestamps = []
action_ids = []
for doc in cursor:
    for timestamp, action_data in doc["actions"].items():
        timestamps.append(timestamp)
        action_ids.append(action_data["id"])

# Prepare the data for the Prophet model
data = {
    "ds": timestamps,
    "y": action_ids
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)
df.head()


prophet_model = Prophet()
try:
    prophet_model.fit(df)  # 'df' should contain your prepared data for training
except Exception as e:
    print("Error training the Prophet model:", e)
    exit(1)
# Generate future timestamps for forecasting
future = prophet_model.make_future_dataframe(periods=300, freq='h')  # Adjust periods and frequency as needed

# Make predictions
forecast = prophet_model.predict(future)

# Print the forecasted values
# You can access forecasted values from 'forecast' DataFrame

# Plot the forecast
fig = prophet_model.plot(forecast)
plt.xlabel('Time')
plt.ylabel('Action ID')
plt.title('Forecasted Action IDs')

# Modify plotting code to use np.array (already addressed)
ax = fig.gca()
ax.plot(
    np.array(prophet_model.history['ds']),
    prophet_model.history['y'],
    'k.',
    label='Actual'
)

# Display the plot
plt.show()


# Save the trained model
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(prophet_model, f)

