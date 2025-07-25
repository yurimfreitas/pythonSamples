import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTzV_K-DeS1Ox4BLqUMgo3YNmmIH8rJC9yCjujCIaHNMeqjEa3m5RUQG-82JIbIyDpcc-aYn0NEzgq7/pub?output=csv')
profile = ProfileReport(df, title="Profiling Report")

profile