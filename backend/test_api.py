import requests
import json

# Test health endpoint
print("Testing /health endpoint...")
health_response = requests.get("http://localhost:5000/health")
print(f"Status: {health_response.status_code}")
print(f"Response: {health_response.json()}\n")

# Test predict endpoint with sample features
print("Testing /predict endpoint...")
# Create sample features (must match the number of features used in training)
sample_features = [
    1,      # HOME
    100,    # L_PTS_ROLL5
    50,     # L_REB_ROLL5
    20,     # L_AST_ROLL5
    8,      # L_STL_ROLL5
    5,      # L_BLK_ROLL5
    0,      # O_PTS_ROLL5
    0,      # O_REB_ROLL5
    0,      # O_AST_ROLL5
    0,      # O_STL_ROLL5
    0,      # O_BLK_ROLL5
    0,      # O_BACK_TO_BACK
    999,    # O_DAYS_REST
    0,      # L_BACK_TO_BACK
    999,    # L_DAYS_REST
]

predict_data = {"features": sample_features}
predict_response = requests.post(
    "http://localhost:5000/predict",
    json=predict_data,
    headers={"Content-Type": "application/json"}
)
print(f"Status: {predict_response.status_code}")
print(f"Response: {predict_response.json()}")
