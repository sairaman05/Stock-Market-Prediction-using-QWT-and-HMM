import numpy as np
import pandas as pd
import yfinance as yf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def fetch_stock_data(ticker, start_date, end_date):
    try:
        # Download data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker} between {start_date} and {end_date}. Check ticker or date range.")
        
        # Reset index to make 'Date' a column
        df = df.reset_index()
        df = pd.DataFrame({
            'ds': pd.to_datetime(df['Date']),
            'open': df['Open'].to_numpy().flatten(),
            'high': df['High'].to_numpy().flatten(),
            'low': df['Low'].to_numpy().flatten(),
            'y': df['Close'].to_numpy().flatten(),  # Use Close as 'y' for consistency
            'volume': df['Volume'].to_numpy().flatten()
        })
        df['returns'] = df['y'].pct_change().fillna(0)  # Calculate returns from Close
        print(f"Fetched {len(df)} days of OHLCV data for {ticker} from {start_date} to {end_date}")
        return df
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"Error fetching data: {e}. Possibly a network issue or invalid ticker.")
        return None
        
def quantum_wavelet_transform(data, n_qubits, plot_circuit=False):
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(normalized_data[i % len(normalized_data)], i)
    qc.h(range(n_qubits))
    qc.measure_all()
    
    # Only plot the circuit if requested
    if plot_circuit:
        print("\nFinal Quantum Circuit:")
        print(qc.draw())
    
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1024).result()
    counts = result.get_counts()
    
    probabilities = np.zeros(2**n_qubits)
    for state, count in counts.items():
        probabilities[int(state, 2)] = count/1024
    
    mid_point = len(probabilities)//2
    return {
        'coeffs': probabilities[mid_point-8:mid_point+8].tolist(),
        'energy': np.sum(probabilities**2),
        'entropy': -np.sum(probabilities * np.log(probabilities + 1e-10)),
        'entanglement': np.var(probabilities)
    }

def create_features(df, window_size=128):
    df['returns'] = df['y'].pct_change()
    features = []
    
    # Only plot circuit for the last window
    plot_circuit = False
    
    for i in range(window_size, len(df)):
        window_data = df['y'].iloc[i-window_size:i].values
        try:
            # Only plot circuit for the last window
            if i == len(df) - 1:
                plot_circuit = True
            quantum_features = quantum_wavelet_transform(window_data, n_qubits=7, plot_circuit=plot_circuit)
            features.append({
                'ds': df['ds'].iloc[i],
                **quantum_features,
                'returns': df['returns'].iloc[i]
            })
        except Exception as e:
            print(f"Error processing window {i}: {str(e)}")
            continue
    
    return pd.DataFrame(features).dropna()

def train_hmm(features, n_states=3):
    scaler = StandardScaler()
    X = scaler.fit_transform(features[['returns', 'energy', 'entropy', 'entanglement']])
    
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )
    model.fit(X)
    return model, scaler

def predict_future_prices(model, scaler, features, last_price, days=100):
    latest_features = scaler.transform(features[['returns', 'energy', 'entropy', 'entanglement']].tail(1))
    random_state = np.random.RandomState(42)
    return_std = features['returns'].std()
    return_mean = features['returns'].mean()
    
    future_prices = []
    current_price = last_price
    
    for _ in range(days):
        try:
            state = model.predict(latest_features)[0]
            new_features = model._generate_sample_from_state(state, random_state=random_state)
            raw_return = scaler.inverse_transform(new_features.reshape(1, -1))[0][0]
            
            # Constrain returns to reasonable values
            constrained_return = np.clip(raw_return, 
                                       return_mean - 2*return_std,
                                       return_mean + 2*return_std)
            
            current_price *= (1 + constrained_return)
            future_prices.append(current_price)
            latest_features = scaler.transform([[
                constrained_return,
                new_features[1],
                new_features[2],
                new_features[3]
            ]])
        except Exception as e:
            print(f"Prediction error at step {_}: {str(e)}")
            break
    
    return future_prices

def calculate_metrics(actual, predicted):
    """Calculate RMSE and MAE metrics"""
    from sklearn.metrics import r2_score
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mae = np.mean(np.abs(actual - predicted))
    r2 = r2_score(actual, predicted)
    
    return rmse, mae, r2

def main():
    try:
        # Configuration
        ticker = "AAPL"
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        prediction_days = 100
        
        # Fetch data
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is None:
            raise ValueError("Failed to fetch data")
        print(f"Fetched {len(df)} historical data points")
        
        # Generate features
        feature_df = create_features(df)
        if feature_df is None or len(feature_df) < 10:
            raise ValueError("Insufficient features generated")
        print(f"\nGenerated {len(feature_df)} feature rows")
        
        # Split data into training and validation sets
        split_idx = int(len(feature_df) * 0.8)  # 80% train, 20% validation
        train_df = feature_df.iloc[:split_idx]
        test_df = feature_df.iloc[split_idx:]
        
        # Train model
        hmm_model, scaler = train_hmm(train_df)
        
        # Validate on test set
        X_test = scaler.transform(test_df[['returns', 'energy', 'entropy', 'entanglement']])
        test_predictions = hmm_model.predict(X_test)
        
        # Backtest predictions
        test_dates = test_df['ds'].values
        test_prices = df[df['ds'].isin(test_dates)]['y'].values
        
        # Reconstruct prices from predicted returns
        predicted_prices = [test_prices[0]]
        for i in range(1, len(test_prices)):
            predicted_return = test_df['returns'].iloc[i-1]  # Using actual returns for reconstruction
            predicted_prices.append(predicted_prices[-1] * (1 + predicted_return))
        
        # Calculate metrics
        rmse, mae, r2 = calculate_metrics(test_prices, predicted_prices)
        print(f"\nValidation Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        
        # Future prediction
        last_historical_date = df['ds'].iloc[-1]
        future_dates = pd.date_range(
            start=last_historical_date + timedelta(days=1),
            periods=prediction_days,
            freq='B'
        )
        
        future_prices = predict_future_prices(
            hmm_model, scaler, feature_df,
            last_price=df['y'].iloc[-1],
            days=prediction_days
        )
        
        # Create combined dataframe
        future_df = pd.DataFrame({
            'ds': future_dates,
            'y': future_prices,
            'type': 'predicted'
        })
        
        historical_df = pd.DataFrame({
            'ds': df['ds'],
            'y': df['y'],
            'type': 'historical'
        })
        
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)
        
        # Visualizations
        plt.figure(figsize=(14, 7))
        plt.plot(historical_df['ds'], historical_df['y'], label='Historical Price', color='blue')
        plt.plot(future_df['ds'], future_prices, label='Predicted Price', color='orange', linestyle='--')
        plt.axvline(x=last_historical_date, color='red', linestyle='--', label='Prediction Start')
        plt.title(f"{ticker} Stock Price Prediction\nValidation RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    main()
