import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2

class Training:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None

    def split_data(self, test_size=0.2):
        # Split data into training and testing sets
        return train_test_split(self.X, self.y, test_size=test_size, shuffle=False)

    def build_model(self, input_shape):
        # Build LSTM model architecture
        model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-6)), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-6))),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-6))),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu', kernel_regularizer=l2(1e-6)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(5)
    ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train(self, epochs=100, batch_size=32):
        # Train the model
        X_train, X_test, y_train, y_test = self.split_data()
        self.model = self.build_model((self.X.shape[1], self.X.shape[2]))
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                 epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
        self.model.save('./model/stock_predictor.h5')
        return history, X_test, y_test

    def evaluate_model(self, X_test, y_test):
        # Evaluate model performance
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Calculate directional accuracy
        direction_true = np.sign(np.diff(y_test[:, 0]))
        direction_pred = np.sign(np.diff(y_pred[:, 0]))
        directional_accuracy = np.mean(direction_true == direction_pred)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Directional Accuracy": directional_accuracy
        }