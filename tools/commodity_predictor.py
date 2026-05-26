import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ============================
# COMMODITY TICKER MAPPING
# ============================

COMMODITY_TICKERS = {
    "gold": "GC=F",
    "crude_oil": "CL=F",
    "natural_gas": "NG=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "wheat": "ZWH=F",
    "corn": "ZCH=F",
    "soybeans": "ZSH=F",
}

# ============================
# COMMODITY PREDICTOR CLASS
# ============================

class CommodityPredictor:
    def __init__(self, commodity_name="gold", lookback_days=365):
        """
        Initialize commodity predictor
        
        Args:
            commodity_name: Name of commodity (e.g., 'gold', 'crude_oil')
            lookback_days: Historical data to fetch
        """
        self.commodity_name = commodity_name.lower()
        self.ticker = COMMODITY_TICKERS.get(self.commodity_name, "GC=F")
        self.lookback_days = lookback_days
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.predictions = {}
        
        print(f"[Predictor] Initializing for {commodity_name} ({self.ticker})")
    
    def fetch_data(self):
        """Fetch historical commodity data from Yahoo Finance"""
        try:
            print(f"[Predictor] Fetching historical data for {self.commodity_name}...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Primary attempt: explicit start/end
            self.data = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            # Fallback: sometimes Yahoo returns empty for start/end—try period-based download
            if self.data is None or self.data.empty:
                print(f"[Predictor] Primary download empty, retrying with period parameter...")
                self.data = yf.download(self.ticker, period=f"{self.lookback_days}d", interval="1d", progress=False)

            # Debug: show returned object type and columns
            print(f"[Predictor] Download result type: {type(self.data)}")
            if hasattr(self.data, 'columns'):
                print(f"[Predictor] Columns returned: {list(self.data.columns)}")

            # Flatten MultiIndex columns from yfinance if necessary
            if hasattr(self.data, 'columns') and hasattr(self.data.columns, 'nlevels') and self.data.columns.nlevels > 1:
                self.data.columns = ["_".join([str(x) for x in col if x is not None]).strip() for col in self.data.columns]
                print(f"[Predictor] Flattened columns: {list(self.data.columns)}")

            # Determine which column to use for price
            price_col = None
            if self.data is not None and hasattr(self.data, 'columns'):
                cols = [c for c in self.data.columns]
                candidates = ['Close', 'Adj Close']
                for cand in candidates:
                    if cand in cols:
                        price_col = cand
                        break
                if price_col is None:
                    for c in cols:
                        if str(c).lower().startswith('close') or str(c).lower().startswith('adj close'):
                            price_col = c
                            break

            if price_col is None:
                print(f"[Predictor] Could not find a price column in downloaded data")
                return False

            # Normalize to 'Close' column name for downstream code
            print(f"[Predictor] Selected price column: {price_col}")
            if price_col != 'Close':
                self.data = self.data.rename(columns={price_col: 'Close'})
            print(f"[Predictor] Columns after normalization: {list(self.data.columns)}")
            self.data = self.data.dropna(subset=['Close'])
            
            if self.data is None or self.data.empty:
                print(f"[Predictor] No data found for {self.ticker} after retries")
                return False

            # Require a minimum number of data points for sequence length (default 60)
            min_required = 61
            if len(self.data) < min_required:
                print(f"[Predictor] Not enough data for {self.ticker}: {len(self.data)} rows (need >= {min_required})")
                return False
            
            print(f"[Predictor] Fetched {len(self.data)} days of data")
            print(f"[Predictor] Price range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}")
            
            return True
        except Exception as e:
            print(f"[Predictor] Error fetching data: {e}")
            return False
    
    def prepare_data(self, sequence_length=60):
        """Prepare data for LSTM model"""
        try:
            if self.data is None:
                print("[Predictor] No data loaded")
                return False
            
            # Use closing prices
            prices = self.data['Close'].values.reshape(-1, 1)
            scaled_prices = self.scaler.fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_prices) - sequence_length):
                X.append(scaled_prices[i:i + sequence_length])
                y.append(scaled_prices[i + sequence_length])
            
            X, y = np.array(X), np.array(y)
            
            # Split into train/test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            print(f"[Predictor] Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"[Predictor] Error preparing data: {e}")
            return False, False, False, False
    
    def build_lstm_model(self, sequence_length=60):
        """Build LSTM model for time series prediction"""
        try:
            print("[Predictor] Building LSTM model...")
            
            model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            print("[Predictor] Model built successfully")
            
            return model
        except Exception as e:
            print(f"[Predictor] Error building model: {e}")
            return None
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=50):
        """Train the LSTM model"""
        try:
            print("[Predictor] Training model...")
            
            self.model = self.build_lstm_model(sequence_length=X_train.shape[1])
            
            if self.model is None:
                return False
            
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            
            print(f"[Predictor] Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            return True
        except Exception as e:
            print(f"[Predictor] Error training model: {e}")
            return False
    
    def predict_future(self, days_ahead=30):
        """Predict future commodity prices"""
        try:
            if self.model is None or self.data is None:
                print("[Predictor] Model or data not available")
                return None
            
            print(f"[Predictor] Predicting {days_ahead} days ahead...")
            
            # Get last sequence
            prices = self.data['Close'].values.reshape(-1, 1)
            scaled_prices = self.scaler.transform(prices)
            last_sequence = scaled_prices[-60:].reshape(1, 60, 1)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                next_pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                           next_pred.reshape(1, 1, 1), axis=1)
            
            # Inverse transform
            predictions = np.array(predictions).reshape(-1, 1)
            actual_predictions = self.scaler.inverse_transform(predictions)
            
            return actual_predictions.flatten()
        except Exception as e:
            print(f"[Predictor] Error predicting: {e}")
            return None
    
    def get_predictions_summary(self):
        """Get predictions for different timeframes"""
        try:
            current_price = self.data['Close'].iloc[-1]
            
            pred_1m = self.predict_future(days_ahead=30)
            pred_6m = self.predict_future(days_ahead=180)
            pred_12m = self.predict_future(days_ahead=365)
            
            if pred_1m is None or pred_6m is None or pred_12m is None:
                return None
            
            summary = {
                "commodity": self.commodity_name,
                "ticker": self.ticker,
                "current_price": float(current_price),
                "last_date": str(self.data.index[-1].date()),
                "predictions": {
                    "1_month": {
                        "price": float(pred_1m[-1]),
                        "change": float((pred_1m[-1] - current_price) / current_price * 100),
                        "min": float(pred_1m.min()),
                        "max": float(pred_1m.max()),
                        "days": 30
                    },
                    "6_months": {
                        "price": float(pred_6m[-1]),
                        "change": float((pred_6m[-1] - current_price) / current_price * 100),
                        "min": float(pred_6m.min()),
                        "max": float(pred_6m.max()),
                        "days": 180
                    },
                    "12_months": {
                        "price": float(pred_12m[-1]),
                        "change": float((pred_12m[-1] - current_price) / current_price * 100),
                        "min": float(pred_12m.min()),
                        "max": float(pred_12m.max()),
                        "days": 365
                    }
                }
            }
            
            return summary
        except Exception as e:
            print(f"[Predictor] Error generating summary: {e}")
            return None
    
    def full_pipeline(self):
        """Run complete prediction pipeline"""
        print(f"\n{'='*60}")
        print(f"RUNNING PREDICTION PIPELINE FOR {self.commodity_name.upper()}")
        print(f"{'='*60}\n")
        
        # Fetch data
        if not self.fetch_data():
            return None
        
        # Prepare data
        result = self.prepare_data(sequence_length=60)
        if result is False or len(result) != 4:
            return None
        X_train, X_test, y_train, y_test = result
        
        # Train model
        if not self.train_model(X_train, y_train, X_test, y_test, epochs=50):
            return None
        
        # Get predictions
        summary = self.get_predictions_summary()
        
        return summary


# ============================
# BATCH PREDICTION FUNCTION
# ============================

def predict_commodities(commodities=None, lookback_days=365):
    """
    Predict multiple commodities at once
    
    Args:
        commodities: List of commodity names (default: ['gold', 'crude_oil', 'silver'])
        lookback_days: Historical data to fetch
        
    Returns:
        Dictionary with predictions for each commodity
    """
    if commodities is None:
        commodities = ["gold", "crude_oil", "silver"]
    
    results = {}
    
    for commodity in commodities:
        try:
            predictor = CommodityPredictor(commodity, lookback_days)
            summary = predictor.full_pipeline()
            if summary:
                results[commodity] = summary
        except Exception as e:
            print(f"Error predicting {commodity}: {e}")
            results[commodity] = None
    
    return results


if __name__ == "__main__":
    # Test predictions
    commodities = ["gold", "crude_oil"]
    predictions = predict_commodities(commodities)
    
    for commodity, pred in predictions.items():
        if pred:
            print(f"\n{commodity.upper()}:")
            print(f"  Current Price: ${pred['current_price']:.2f}")
            print(f"  1-Month Prediction: ${pred['predictions']['1_month']['price']:.2f} "
                  f"({pred['predictions']['1_month']['change']:+.2f}%)")
            print(f"  6-Month Prediction: ${pred['predictions']['6_months']['price']:.2f} "
                  f"({pred['predictions']['6_months']['change']:+.2f}%)")
            print(f"  12-Month Prediction: ${pred['predictions']['12_months']['price']:.2f} "
                  f"({pred['predictions']['12_months']['change']:+.2f}%)")
