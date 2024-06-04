import pandas as pd
from datetime import datetime, timedelta

# Define the start date and the number of periods (rows)
start_date = '2024-05-20 00:00:00'
# num_periods = 8928  # Number of timestamps you want
num_periods = 2880 # Number of timestamps you want
pattern_length = 288  # Pattern length

# Generate the date range with 5-minute intervals
date_range = pd.date_range(start=start_date, periods=num_periods, freq='5min')

# Create a DataFrame with the generated timestamps
df_timestamps = pd.DataFrame(date_range, columns=['ds'])

# Format the timestamp to the desired format
df_timestamps['ds'] = df_timestamps['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Define your y values pattern for the first cycle (288 rows)
y_values_pattern_cycle1 = [1,1,2,3,1,1,1,1,3,3,3,3,3,3,3,3,1,1,1,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,3,3,3,3,3,3,3,3,3,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,8,11,11,11,11,11,11,11,11,11,11,11,11,11,11,13,13,13,13,13,13,13,13,13,13,13,13,13,13,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,14,12,12,12,12,12,12,12,12,12,12,12,12,12,12,10,10,10,10,10,10,10,10,10,10,10,10,8,8,8,8,8,8,8,8,8,8,8,8,6,6,6,6,6,6,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7]
# Repeat the pattern for the first cycle
y_values_cycle1 = y_values_pattern_cycle1

# Repeat the first cycle's pattern for the next 30 cycles
y_values = y_values_cycle1 * 10

# Ensure the length of y_values matches num_periods
if len(y_values) != num_periods:
    raise ValueError("The length of y_values must match num_periods")

# Add the 'y' column to the DataFrame
df_timestamps['y'] = y_values

# Save the DataFrame to a CSV file
df_timestamps.to_csv('timestamps1.csv', index=False)

# Confirm the file is saved
print('Timestamps with y values saved to timestamps_with_y_values.csv')


# print("CSV file created successfully.")


# import csv
# from datetime import datetime, timedelta

# # Function to generate data for one cycle
# def generate_cycle(start_time):
#     data = []
#     for i in range(3590):
#         y_value = 1 + (i // 359)
#         ds_value = start_time + timedelta(minutes=i)
#         data.append({'ds': ds_value.strftime('%Y-%m-%d %H:%M:%S'), 'y': y_value})
#     for i in range(3590, 7180):
#         y_value = 10 - (i - 3590) // 359
#         ds_value = start_time + timedelta(minutes=i)
#         data.append({'ds': ds_value.strftime('%Y-%m-%d %H:%M:%S'), 'y': y_value})
#     return data

# # Start timestamp
# start_time = datetime.now()

# # Initialize the data list
# data = []

# # Generate data for 100 cycles
# for _ in range(3):
#     data += generate_cycle(start_time)
#     start_time += timedelta(minutes=7180)  # Update start_time for the next cycle

# # Write the data to a CSV file
# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=['ds', 'y'])
#     writer.writeheader()
#     for row in data:
#         writer.writerow(row)

# print("CSV file created successfully.")
