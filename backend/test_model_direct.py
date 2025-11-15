import sys
import os
sys.path.insert(0, "C:\\Users\\harno\\lakers-win-project\\backend")

# Test the model directly without needing the server
import joblib
import numpy as np

base_dir = "C:\\Users\\harno\\lakers-win-project\\backend"
model_path = os.path.join(base_dir, "data", "lakers_win_model.pkl")
scaler_path = os.path.join(base_dir, "data", "lakers_scaler.pkl")

print("Testing model directly...")
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(f"Model type: {type(model)}")
print(f"Scaler type: {type(scaler)}")

# Create sample features (15 features based on our training dataset)
sample_features = np.array([
    1,      # HOME
    100,    # L_PTS_ROLL5
    50,     # L_REB_ROLL5
    20,     # L_AST_ROLL5
    8,      # L_STL_ROLL5
    5,      # L_BLK_ROLL5
    95,     # O_PTS_ROLL5
    48,     # O_REB_ROLL5
    19,     # O_AST_ROLL5
    7,      # O_STL_ROLL5
    4,      # O_BLK_ROLL5
    0,      # O_BACK_TO_BACK
    3,      # O_DAYS_REST
    0,      # L_BACK_TO_BACK
    2,      # L_DAYS_REST
]).reshape(1, -1)

print(f"\nSample features shape: {sample_features.shape}")

# Scale features
sample_scaled = scaler.transform(sample_features)

# Make prediction
prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

print(f"\nPrediction: {'Win' if prediction == 1 else 'Loss'}")
print(f"Win Probability: {probability:.4f}")
print("\nSUCCESS: Model works correctly!")
