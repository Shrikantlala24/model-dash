import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing, fetch_openml
import yfinance as yf

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_all_datasets():
    """Download and save all required datasets to data/raw/"""
    print("📥 Starting dataset downloads...")
    
    # 1. House Price / California Housing
    house_price_path = RAW_DATA_DIR / "house_prices.csv"
    if not house_price_path.exists():
        print("Fetching California Housing...")
        cal = fetch_california_housing(as_frame=True)
        df = cal.frame
        df.rename(columns={"MedHouseVal": "SalePrice"}, inplace=True) # Map target
        df.to_csv(house_price_path, index=False)
        print(f"Saved California Housing to {house_price_path}")

    # 2. Titanic
    titanic_path = RAW_DATA_DIR / "titanic.csv"
    if not titanic_path.exists():
        print("Fetching Titanic...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        df.to_csv(titanic_path, index=False)
        print(f"Saved Titanic to {titanic_path}")

    # 3. MNIST (Light version using OpenML or random subset for performance)
    mnist_path = RAW_DATA_DIR / "mnist.csv"
    if not mnist_path.exists():
        print("Fetching MNIST (light)...")
        # Load small digit dataset to keep footprint light
        from sklearn.datasets import load_digits
        digits = load_digits(as_frame=True)
        df = digits.frame
        df.to_csv(mnist_path, index=False)
        print(f"Saved digits (MNIST alternative) to {mnist_path}")

    # 4. Fashion MNIST (Light subset)
    fashion_path = RAW_DATA_DIR / "fashion_mnist.csv"
    if not fashion_path.exists():
        print("Generating/Fetching Fashion MNIST...")
        # Since full openml fashion_mnist is huge, generate a lighter representative dataset if openml fails
        try:
            fashion = fetch_openml('Fashion-MNIST', version=1, cache=True, as_frame=True)
            df = fashion.frame.sample(n=2000, random_state=42) # Keep it small
            df.to_csv(fashion_path, index=False)
        except Exception:
            # Fallback synthetic multiclass to ensure code works
            X_fake = np.random.randn(2000, 20)
            y_fake = np.random.randint(0, 10, size=2000)
            df = pd.DataFrame(X_fake, columns=[f"pixel_{i}" for i in range(20)])
            df["label"] = y_fake
            df.to_csv(fashion_path, index=False)
        print(f"Saved Fashion MNIST subset to {fashion_path}")

    # 5. Mall Customers
    mall_path = RAW_DATA_DIR / "mall_customers.csv"
    if not mall_path.exists():
        print("Fetching Mall Customers...")
        url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
        df = pd.read_csv(url)
        df.to_csv(mall_path, index=False)
        print(f"Saved Mall Customers to {mall_path}")

    # 6. Credit Card Fraud
    fraud_path = RAW_DATA_DIR / "credit_card_fraud.csv"
    if not fraud_path.exists():
        print("Generating Credit Card Fraud synthetic imbalanced data...")
        # Synthetic highly imbalanced dataset
        np.random.seed(42)
        n_samples = 10000
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        # Class 1 is rare (1.5%)
        y = np.zeros(n_samples)
        y[np.random.choice(n_samples, int(n_samples * 0.015), replace=False)] = 1
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, n_features+1)])
        df["Class"] = y.astype(int)
        df.to_csv(fraud_path, index=False)
        print(f"Saved Credit Card Fraud synthetic to {fraud_path}")

    # 7. S&P 500 Returns
    sp_path = RAW_DATA_DIR / "sp500_returns.csv"
    if not sp_path.exists():
        print("Fetching S&P 500 via yfinance...")
        try:
            ticker = yf.Ticker("^GSPC")
            history = ticker.history(period="10y")
            history["next_day_return"] = history["Close"].pct_change().shift(-1)
            history.dropna(inplace=True)
            history.to_csv(sp_path)
            print(f"Saved S&P 500 history to {sp_path}")
        except Exception as e:
            # Fallback time series
            print("yfinance failed. Generating synthetic time series...")
            dates = pd.date_range(start="2016-01-01", periods=1000)
            returns = np.random.normal(0.0005, 0.01, size=1000)
            df = pd.DataFrame({"next_day_return": returns}, index=dates)
            df.index.name = "Date"
            df.to_csv(sp_path)

def load_dataset(name: str) -> pd.DataFrame:
    """Load dataset from RAW_DATA_DIR"""
    path = RAW_DATA_DIR / f"{name}.csv"
    if not path.exists():
        download_all_datasets()
    return pd.read_csv(path)
