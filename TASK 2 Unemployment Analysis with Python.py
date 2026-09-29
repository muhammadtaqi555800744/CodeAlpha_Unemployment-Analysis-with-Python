# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the path to the file you'd like to load
file_path = "Unemployment_Rate_upto_11_2020.csv"

# Load the latest version
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "gokulrajkmv/unemployment-in-india",
    file_path
)

print("First 5 records:")
print(df.head())

# ==========================================
# 1. DATA CLEANING & PREPROCESSING
# ==========================================
# Clean column names (strip trailing spaces if any)
df.columns = df.columns.str.strip()

# Rename columns for simpler access
df.rename(columns={
    'Region': 'States',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'].str.strip(), dayfirst=True)

# Extract Month and Year for trend analysis
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\nDataset Summary Statistics:")
print(df.describe())

# Group by Region/Zone to look at broader geographic trends
region_stats = df.groupby('Region.1')['Unemployment_Rate'].mean().reset_index()
print("\nAverage Unemployment Rate by Region:")
print(region_stats)

# ==========================================
# 3. COVID-19 IMPACT INVESTIGATION
# ==========================================
# India implemented a strict nationwide lockdown starting March 25, 2020.
# Let's compare pre-lockdown (Jan-Feb 2020) vs peak lockdown period (Apr-May 2020).
df['Period'] = df['Date'].apply(lambda x: 'Pre-Lockdown' if x.month in [1, 2] else ('Lockdown Peak' if x.month in [4, 5] else 'Other'))

covid_impact = df.groupby('Period')['Unemployment_Rate'].mean().reset_index()
print("\nCOVID-19 Impact (Pre-Lockdown vs Lockdown Peak):")
print(covid_impact)

# ==========================================
# 4. VISUALIZATION OF TRENDS
# ==========================================
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Unemployment_Rate', hue='Region.1', marker='o')
plt.title('Unemployment Rate Trends Across Indian Regions (2020)')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.xlabel('Date')
plt.grid(True)
plt.legend(title='Zone')
plt.tight_layout()
plt.show()