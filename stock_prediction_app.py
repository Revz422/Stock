pip install matplotlib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function for EDA and feature engineering
def perform_eda(stock_data):
    st.subheader("Exploratory Data Analysis:")
    st.write(stock_data.head())
    st.write(stock_data.describe())
    st.write("Data Types:")
    st.write(stock_data.dtypes)

    st.subheader("Visualization:")
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Close Price Over Time')
    plt.legend()
    st.pyplot()

# Function for PCA feature engineering
def perform_pca(stock_data):
    st.subheader("Feature Engineering using PCA:")
    X = stock_data[['Open', 'High', 'Low', 'Volume']] # Select relevant features
    y = stock_data['Close'] # Target variable

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    st.write("Explained Variance Ratio:")
    st.write(pca.explained_variance_ratio_)

# Function for DTAML model training
def train_model(X_train, y_train):
    st.subheader("Training DTAML Model:")
    # Define DTAML model
    class DTAML(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(DTAML, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Prepare data for training
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 1

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    # Initialize model, criterion, optimizer
    model = DTAML(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    st.write("Model training completed.")

# Streamlit app
def main():
    st.title("Stock Prediction with DTAML")
    st.sidebar.title("Options")

    # Sidebar inputs
    symbol = st.sidebar.text_input("Enter the stock symbol (e.g., TSLA for Tesla):")
    start_date = st.sidebar.text_input("Enter the start date (YYYY-MM-DD):")
    end_date = st.sidebar.text_input("Enter the end date (YYYY-MM-DD):")

    # Fetch stock data
    if symbol and start_date and end_date:
        stock_data = fetch_stock_data(symbol, start_date, end_date)

        # EDA
        perform_eda(stock_data)

        # PCA
        perform_pca(stock_data)

        # Feature Engineering
        X = stock_data[['Open', 'High', 'Low', 'Volume']] # Select relevant features
        y = stock_data['Close'] # Target variable

        # Train model
        train_model(X, y)

if __name__ == "__main__":
    main()
