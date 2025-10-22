import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#TASK 2a

# Load the dataset
file_path = ''
data = pd.read_csv(file_path)

# !!!Initial Inspection
print("1. Initial Data Inspection")
print("First 5 rows:")
print(data.head())
print("\nData Info (Types & Not Null Counts:")
data.info()

# !!!Data Cleaning
# Drop the first column if it is an unnecessary index ('Unnamed: 0')
if 'Unnamed: 0' in data.columns:
  data = data.drop(columns=['Unnamed: 0'])
  print("\n'Unnamed:0' column dropped!")
print("\n2. Data Cleaning and Statistics")

# Check for Missing Value (Null Check)
print("Missing Values per Column:")
print(data.isnull().sum())


