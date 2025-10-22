import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#TASK 2a

# Load the dataset
file_path = ''
data = pd.read_csv(file_path)

# Initial Inspection
print("1. Initial Data Inspection")
print("First 5 rows:")
print(data.head())
print("Data Info (Types & Not Null Counts:")
data.info()

