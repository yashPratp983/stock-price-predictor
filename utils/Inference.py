import numpy as np
import datetime

class Inferencing:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.features = features

    def predict_next_day(self, current_window):
        # Predict the next day's stock data
        input_data = np.expand_dims(current_window, axis=0)
        prediction = self.model.predict(input_data)[0]

        # Prepare for inverse transform
        full_featured_prediction = np.zeros((1, len(self.features)))
        full_featured_prediction[0, :5] = prediction

        # Inverse transform to get actual values
        inverse_prediction = self.scaler.inverse_transform(full_featured_prediction)[0]

        return inverse_prediction[:5]

    def update_window(self, window, new_data):
        # Update the window with new data for the next prediction
        window = window[1:]  # Remove the oldest day

        # Add the new day's data
        new_row = np.zeros((1, len(self.features)))
        new_row[0, :5] = new_data

        # Update technical indicators (simplified)
        new_row[0, 5] = np.mean(window[-19:, 0].tolist() + [new_data[0]])  # SMA_20
        new_row[0, 6] = np.mean(window[-49:, 0].tolist() + [new_data[0]])  # SMA_50
        new_row[0, 7] = 50  # Placeholder RSI (you might want to implement a proper RSI calculation)

        return np.vstack((window, new_row))

    def iterative_predict(self, initial_data, start_date, end_date):
        # Perform iterative prediction from start_date to end_date
        current_date = start_date
        current_window = initial_data[-500:]  # Assuming 500-day window
        predictions = []

        while current_date <= end_date:
            # Predict next day
            next_day_prediction = self.predict_next_day(current_window)

            # Store prediction
            predictions.append((current_date, next_day_prediction))

            # Update window
            current_window = self.update_window(current_window, next_day_prediction)

            # Move to next day
            current_date += datetime.timedelta(days=1)

        return predictions

    def predict_future(self, initial_data, future_steps):
        # This method remains for backward compatibility
        predictions = []
        current_window = initial_data.copy()

        for _ in range(future_steps):
            predicted_scaled = self.predict_non_negative(np.expand_dims(current_window, axis=0))[0]
            predictions.append(predicted_scaled)

            new_row = np.zeros((1, len(self.features)))
            new_row[0, :5] = predicted_scaled
            new_row[0, 5] = np.mean(current_window[-20:, 0])  # SMA_20
            new_row[0, 6] = np.mean(current_window[-50:, 0])  # SMA_50
            new_row[0, 7] = 50  # Placeholder RSI

            current_window = np.vstack((current_window[1:], new_row))

        predictions = np.array(predictions)
        full_featured_predictions = np.zeros((len(predictions), len(self.features)))
        full_featured_predictions[:, :5] = predictions

        inverse_predictions = self.scaler.inverse_transform(full_featured_predictions)
        return inverse_predictions[:, :5]

    def predict_non_negative(self, X):
        # Ensure non-negative predictions
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)