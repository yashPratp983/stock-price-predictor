import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.stock_data = None
        self.scaler = StandardScaler()

    def load_data(self):
        # Load CSV file and sort by date
        self.stock_data = pd.read_csv(self.file_path)


    def rename_columns(self):
        # Rename Japanese column names to English
        english_columns = {
            '日付け': 'Date', '出来高': 'Volume', '変化率 %': 'Change_Rate',
            '終値': 'Closing_Price', '始値': 'Opening_Price',
            '高値': 'High_Price', '安値': 'Low_Price',
        }
        self.stock_data.rename(columns=english_columns, inplace=True)
        self.stock_data.drop(columns=['Change_Rate'], inplace=True)
        self.stock_data = self.stock_data.sort_values('Date')

    def convert_volume(self):
        # Convert volume from string (with M or B suffix) to float
        def convert(volume_str):
            if 'M' in volume_str:
                return float(volume_str.replace('M', '')) * 1e6
            elif 'B' in volume_str:
                return float(volume_str.replace('B', '')) * 1e9
            else:
                return float(volume_str)
        self.stock_data['Volume'] = self.stock_data['Volume'].apply(convert)

    def add_technical_indicators(self):
        # Add Simple Moving Averages (SMA) and Relative Strength Index (RSI)
        self.stock_data['SMA_20'] = self.stock_data['Closing_Price'].rolling(window=20).mean()
        self.stock_data['SMA_50'] = self.stock_data['Closing_Price'].rolling(window=50).mean()
        self.stock_data['RSI'] = self.calculate_rsi(self.stock_data['Closing_Price'])

    @staticmethod
    def calculate_rsi(prices, period=14):
        # Calculate Relative Strength Index
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_data(self, window_size=500):
        # Prepare data for LSTM model (sliding window approach)
        features = ['Closing_Price', 'Opening_Price', 'High_Price', 'Low_Price', 'Volume',
                    'SMA_20', 'SMA_50', 'RSI']

        self.stock_data = self.stock_data.dropna()
        scaled_data = self.scaler.fit_transform(self.stock_data[features])

        X, y = [], []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:(i + window_size)])
            y.append(scaled_data[i + window_size, :5])

        return np.array(X), np.array(y), features

    def preprocess(self):
        # Run all preprocessing steps
        self.load_data()
        self.rename_columns()
        self.convert_volume()
        self.add_technical_indicators()
        self.stock_data.to_csv('./dataset/modified_stock_data.csv')
        return self.prepare_data()