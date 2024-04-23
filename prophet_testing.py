import pandas as pd
from prophet import Prophet
import pymongo
import random
from datetime import datetime, timedelta

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

# Get the length of the original timestamps list
original_length = len(timestamps)

# Modify timestamps with different offsets
timestamps = [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i, ts in enumerate(timestamps)]

for i in range(1,10):
    # Create a new set of timestamps continuing from where the first set left off
    new_timestamps = [(datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S") + timedelta(minutes=i+1)).strftime("%Y-%m-%d %H:%M:%S") for i in range(original_length)]
    new_action_ids = action_ids[:original_length]  # Use the same action IDs as the first set

    # Extend the original timestamps and action IDs with the new timestamps and action IDs
    timestamps.extend(new_timestamps)
    action_ids.extend(new_action_ids)

# print(timestamps)

# Create a DataFrame with the timestamps and action IDs
data = {'ds': timestamps, 'y': action_ids}
df = pd.DataFrame(data)

# Save the data to a CSV file
df.to_csv('data_with_random.csv', index=False)
print("Data saved to 'data_with_random.csv'")