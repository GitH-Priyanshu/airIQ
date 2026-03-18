import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_DIR, "data", "clean_air_quality.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.pkl")

def train_and_save():
    print(f"Loading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df_clean = df.dropna().copy()
    
    # Define features and target
    X = df_clean[["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]]
    y = df_clean["AQI"]

    print("Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X, y)
    
    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save()
