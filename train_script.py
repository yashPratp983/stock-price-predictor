from utils.preprocess import Preprocessing
from utils.training import Training
file_path = "./dataset/stock_price.csv"

# Preprocessing
preprocessor = Preprocessing(file_path)
X, y, features = preprocessor.preprocess()

# Training
trainer = Training(X, y)
history, X_test, y_test = trainer.train()
evaluation_results = trainer.evaluate_model(X_test, y_test)
print("Model Evaluation Results:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value}")
