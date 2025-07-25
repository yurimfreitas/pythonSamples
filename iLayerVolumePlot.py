import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.ticker import FuncFormatter
import numpy as np

# Define the path to the CSV file
file_path = os.path.expanduser('~/Desktop/iLayerVolume.csv')

# Read the CSV file
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')

# Sort the dataframe by date
df = df.sort_values(by='Date')

# Function to format numbers for y-axis
def format_number_axis(x, pos):
    if x >= 1_000_000_000:
        return f'{x / 1_000_000_000:.1f}B'
    elif x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.1f}k'
    else:
        return str(int(x))

# Function to format numbers for bar labels
def format_number(x):
    if x >= 1_000_000_000:
        return f'{x / 1_000_000_000:.1f}B'
    elif x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.1f}k'
    else:
        return str(x)

# Plot the data
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Date'].dt.strftime('%d/%m/%Y'), df['TOTAL_RECORDS'], color='skyblue')

# Add the text labels inside the bars with 90-degree rotation
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2, format_number(height), 
             ha='center', va='center', fontsize=10, color='black', rotation=90)

# Add trend line
# Convert dates to ordinal format for trend line computation
x = df['Date'].map(pd.Timestamp.toordinal)
y = df['TOTAL_RECORDS']
# Perform linear regression
coefficients = np.polyfit(x, y, 1)
trendline = np.poly1d(coefficients)
plt.plot(df['Date'].dt.strftime('%d/%m/%Y'), trendline(x), color='red', linewidth=2, label='Trend Line')

plt.xlabel('Date')
plt.ylabel('Total Records')
plt.title('Total Records Over Time')
plt.xticks(rotation=45, ha='right')  # Rotate dates for better readability
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number_axis))  # Set y-axis formatter
plt.legend()
plt.tight_layout()  # Adjust layout to fit labels
plt.show()