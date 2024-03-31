# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch stock market data using yfinance
ticker_symbol = "AAPL"  # Example ticker symbol (replace with your desired symbol)
stock_data = yf.download(ticker_symbol, start="2020-01-01", end="2021-01-01")  # Adjust dates as needed

# Perform Exploratory Data Analysis (EDA)
# Basic statistics
print("Basic Statistics:")
print(stock_data.describe())

# Visualize closing prices distribution
plt.figure(figsize=(10, 6))
sns.histplot(stock_data["Close"], bins=30, kde=True)
plt.title("Distribution of Closing Prices")
plt.xlabel("Closing Price")
plt.ylabel("Frequency")
plt.show()

# Feature Engineering
# Adding lag features
for lag in [1, 3, 5]:  # Example lag values
    stock_data[f"Close_Lag_{lag}"] = stock_data["Close"].shift(lag)

# Adding rolling statistics
window_sizes = [5, 10, 20]  # Example window sizes
for window in window_sizes:
    stock_data[f"Rolling_Mean_{window}"] = stock_data["Close"].rolling(window=window).mean()
    stock_data[f"Rolling_Std_{window}"] = stock_data["Close"].rolling(window=window).std()

# Adding technical indicators (e.g., Moving Average Convergence Divergence - MACD)
# You can use libraries like TA-Lib for technical indicators

# Drop NaN values resulting from feature engineering
stock_data.dropna(inplace=True)

# Split the dataset into features and target variable
X = stock_data.drop("Close", axis=1)  # Features
y = stock_data["Close"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the base model (Random Forest Regressor)
base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Train the base model
base_model.fit(X_train_scaled, y_train)

# Make predictions
predictions = base_model.predict(X_test_scaled)

# Evaluate the model
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)
