# Stock Predictor

## Prerequisites

- Python 3.11
- Conda for environment management

## Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yashPratp983/stock-price-predictor.git
    cd stock_predictor
    ```

2. Create a Conda environment with Python 3.11:
    ```bash
    conda create --name env python=3.11.4
    conda activate env
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Inferncing

1. Navigate to the project directory:
    ```bash
    cd stock_predictor
    ```

2. Run the streamlit app:
    ```bash
    streamlit run client.py
    ```
![Stock Predictor App Screenshot](./public/Screenshot%202024-10-07%20003749.png)

## Training
1. Navigate to the project directory:
    ```bash
    cd stock_predictor
    ```

2. Run the training script:
    ```bash
    python train_script.py
    ```