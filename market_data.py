import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict

class MarketDataFetcher:
    def __init__(self, symbols: List[str] = ['BTC-USD', 'ETH-USD']):
        self.symbols = symbols
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
    def fetch_data(self, period: str = '7d', interval: str = '1h') -> np.ndarray:
        data = []
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            data.append(self._process_ticker_data(hist))
            
        return np.concatenate(data, axis=1)
    
    def _process_ticker_data(self, df: pd.DataFrame) -> np.ndarray:
        # Normalize data
        for feature in self.features:
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
        return df[self.features].values 