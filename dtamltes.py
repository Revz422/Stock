import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import yfinance as yf

class DTAMLAlgorithm:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = None

    def load_data(self):
        """
        Fetch historical stock data using yfinance.
        """
        data = yf.download(self.ticker, start="2020-01-01", end="2022-01-01")
        return data['Close'].values.reshape(-1, 1)

    def dynamic_task_assignment(self, data):
        """
        Dynamic task assignment based on historical stock price trends.
        """
        # Compute price changes
        price_changes = np.diff(data, axis=0)

        # Calculate average price change
        avg_change = np.mean(price_changes)

        # Identify market conditions based on average price change
        if avg_change > 0:
            market_condition = "Bullish"
        elif avg_change < 0:
            market_condition = "Bearish"
        else:
            market_condition = "Neutral"

        print("Market Condition:", market_condition)

        # Additional logic based on market condition can be added here
        pass

    def task_aware_model_adaptation(self, data):
        """
        Task-aware model adaptation logic.
        """
        # Calculate statistics of the data
        data_mean = np.mean(data)
        data_std = np.std(data)

        # Adjust model hyperparameters based on data statistics
        if data_std < 1.0:
            # If data volatility is low, use a simpler model
            model = LinearRegression(normalize=True)
        else:
            # If data volatility is high, use a more complex model
            model = Ridge(alpha=0.1)

        # Train the adapted model
        X = np.arange(len(data)).reshape(-1, 1)
        y = data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        self.model = model

    def train_model(self, data):
        """
        Train the DTAML model using linear regression.
        """
        X = np.arange(len(data)).reshape(-1, 1)
        y = data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        self.model = model

    def predict(self, new_data):
        """
        Make predictions using the trained DTAML model.
        """
        predictions = self.model.predict(np.arange(len(new_data)).reshape(-1, 1))
        return predictions

    def evaluate_model(self, data):
        """
        Evaluate the DTAML model using mean squared error.
        """
        predictions = self.model.predict(np.arange(len(data)).reshape(-1, 1))
        mse = mean_squared_error(data, predictions)
        return mse


def get_available_tickers():
    """
    Fetch available tickers from a predefined list or alternative source.
    """
    # Define a list of predefined tickers
    predefined_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN" , "TSLA"]  # Add more tickers as needed
    
    return predefined_tickers

def select_company(available_tickers):
    """
    Prompt the user to enter the company ticker symbol.
    """
    print("Available tickers:")
    for ticker in available_tickers:
        print(ticker)
    selected_ticker = input("Enter the ticker symbol of the company you want to predict stocks for: ").strip().upper()
    return selected_ticker

# Fetch available tickers
available_tickers = get_available_tickers()

# Prompt the user for the companyAM ticker symbol
ticker = select_company(available_tickers)



# Initialize the DTAMLAlgorithm with the selected ticker
dtaml = DTAMLAlgorithm(ticker)

# Load historical stock data
data = dtaml.load_data()

# Perform dynamic task assignment
dtaml.dynamic_task_assignment(data)

# Adapt the model based on the data
dtaml.task_aware_model_adaptation(data)

# Train the model
dtaml.train_model(data)

# Fetch new data for prediction (real-time data)
new_data = yf.download(ticker, start="2022-01-01", end="2022-02-01")['Close'].values

# Make predictions
predictions = dtaml.predict(new_data)
print("Predictions:", predictions)

# Evaluate the model
mse = dtaml.evaluate_model(data)
print("Mean Squared Error:", mse)
