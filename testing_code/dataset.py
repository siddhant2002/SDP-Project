import pandas as pd
import numpy as np

# Generate date range
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

# Generate random numbers between 1 and 7
random_numbers = np.random.randint(1, 8, len(date_range))

# Create a DataFrame
df = pd.DataFrame({'ds': date_range, 'y': random_numbers})

# Write the DataFrame to a CSV file
df.to_csv('random_numbers.csv', index=False)

print("DataFrame written to 'random_numbers.csv'")
