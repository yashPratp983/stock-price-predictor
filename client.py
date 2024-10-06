import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import datetime
from utils.Inference import Inferencing

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('model\stock_predictor.h5', custom_objects={'mse': MeanSquaredError()})
    return model
# Load and prepare data
@st.cache_data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    
    features = ['Closing_Price', 'Opening_Price', 'High_Price', 'Low_Price', 'Volume',
                'SMA_20', 'SMA_50', 'RSI']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    return df, scaled_data, scaler, features


# Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # Load pre-existing data
    file_path = "./dataset/modified_stock_data.csv"  # Make sure this file exists in your directory
    df, scaled_data, scaler, features = load_and_prepare_data(file_path)

    # Get the last date in the dataset
    last_date = df['Date'].max().date()

    # Display information about the dataset
    st.subheader("Dataset Information")
    st.write(f"Data range: {df['Date'].min().date()} to {last_date}")
    st.write(f"Total days: {len(df)}")

    # Date input for prediction
    target_date = st.date_input("Select target date for prediction", 
                                min_value=last_date + datetime.timedelta(days=1),
                                value=last_date + datetime.timedelta(days=30))


    if st.button("Predict"):
        # Load the model
        model = load_model('model\stock_predictor.h5', custom_objects={'mse': MeanSquaredError()})

        # Create Inferencing object
        inferencer = Inferencing(model, scaler, features)

        # Perform iterative prediction
        predictions = inferencer.iterative_predict(scaled_data, last_date + datetime.timedelta(days=1), target_date)

        # Display predictions
        st.subheader("Predictions")
        date, pred = predictions[-1]
        st.write(f"Predictions for {date}:")
        st.write(f"  Closing Price: {pred[0]:.2f}")
        st.write(f"  Opening Price: {pred[1]:.2f}")
        st.write(f"  High Price: {pred[2]:.2f}")
        st.write(f"  Low Price: {pred[3]:.2f}")
        st.write(f"  Volume: {pred[4]:.0f}")
        st.write("---")

        # Plot the predictions
        st.subheader("Prediction Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical data
        historical_dates = df['Date'].dt.date
        ax.plot(historical_dates, df['Closing_Price'], label='Historical Closing Price')

        # Plot predicted data
        predicted_dates = [pred[0] for pred in predictions]
        predicted_prices = [pred[1][0] for pred in predictions]
        ax.plot(predicted_dates, predicted_prices, label='Predicted Closing Price', linestyle='--')

        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.set_title('Stock Price Prediction')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()