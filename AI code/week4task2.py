import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(data.describe())

# 1. Define Features (Independent Variables - X)
# X includes all the columns you use to make the prediction: TV, Radio, Newspaper.
X = data[['TV', 'Radio', 'Newspaper']]

# 2. Define Target (Dependent Variable - Y)
# Y is the column you are trying to predict: Sales.
Y = data['Sales']

print(f"X (Features) columns: {X.columns.tolist()}")
print(f"Y (Target) defined.")

#-----------------------------------------------------

#TASK 2b

#!!!Split the data (e.g., 80% for training, 20% for testing)
X_train, X_test, Y_train, Y_test = train_test_split(
    X,           # All features
    Y,           # All target values (Sales)
    test_size=0.2, # 20% for testing
    random_state=42 # Ensures the split is the same every time for reproducibility
)

print(f"Training set size: {X_train.shape[0]} samples") # Should be 160 if total is 200
print(f"Testing set size: {X_test.shape[0]} samples")   # Should be 40 if total is 200

#!!!Initializing the Model
# Create an instance (an object) of the Linear Regression model
model = LinearRegression()
print("Linear Regression model initialized successfully!)

#---------------------------------------------------------

#TASK 2c
#!!!Train the model 
model.fit(X_train, Y_train)
print("Model training complete!")
      

#---------------------------------------------------------

#TASK 2d
#1. TESTING
#Use the trained model to predict Sales values for the test data
Y_pred = model.predict(X_test)

#2. EVALUATING
# Calculate the metrics
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (R²): {r2:.4f}")

