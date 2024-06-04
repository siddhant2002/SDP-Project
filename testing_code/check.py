import csv
from datetime import datetime, timedelta

# Initialize the data list
data = []

# Start timestamp
start_time = datetime.now()

# Define the number of rows and the repeating pattern
num_rows = 35900  # Total rows
repeat_pattern = 359  # Repeat pattern for 'y' values

# Generate the data
for i in range(num_rows):
    # Calculate the value for the 'y' column
    y_value = 1 + (i // repeat_pattern) % 10

    # Calculate the timestamp for the 'ds' column
    ds_value = start_time + timedelta(minutes=i)

    # Append the row to the data list
    data.append({'ds': ds_value.strftime('%Y-%m-%d %H:%M:%S'), 'y': y_value})

# Write the data to a CSV file
with open('output.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['ds', 'y'])

    writer.writeheader()
    for row in data:
        writer.writerow(row)

print("CSV file created successfully.")
