import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# TASK 2a

# Load the dataset
file_path = 'C:/Users/ADMIN/PycharmProjects/PythonProject4/Week 4 - dataset_2_advertising.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

print("1. Initial Data Inspection")
print("First 5 rows:")
print(data.head())
print("\nData Info (Types & Not Null Counts:")
data.info()

# Data Cleaning
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])
    print("\n'Unnamed:0' column dropped!")
print("\n2. Data Cleaning and Statistics")

print("Missing Values per Column:")
print(data.isnull().sum())

print("\nDescriptive Statistics:")
print(data.describe())

print("Correlation Matrix:")
corr = data.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, linecolor="black", fmt=".2f", annot_kws={"size": 10})
plt.show()

X = data[['TV', 'radio', 'newspaper']]
Y = data['sales']

# TASK 2b

model = LinearRegression()

# TASK 2c

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

model.fit(X_train, Y_train)

# TASK 2d
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (R²): {r2:.4f}")

# Features vs. Sales Scatter Plot
print("\n3. Visualizing Feature vs. Sales (EDA)")

sns.pairplot(
    data,
    x_vars=['TV', 'radio', 'newspaper'],  # Features to plot on the x-axis
    y_vars=['sales'],                     # Target to plot on the y-axis
    height=4,                             # Makes each plot 4 inches high
    aspect=1,                             # Makes the plots square
    kind='scatter'                        # Specifies a scatter plot
)
plt.suptitle('Relationship of Each Advertising Channel vs. Sales', y=1.02, fontsize=14)
plt.show()

#---------------------------------------------------------



