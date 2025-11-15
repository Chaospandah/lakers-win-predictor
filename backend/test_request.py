import requests

# IMPORTANT: replace these values with real numbers from your dataset
# Send a dict keyed by the feature column names from training
test_features = {
    "HOME": 1,
    "L_BACK_TO_BACK": 0,
    "L_DAYS_REST": 2,
    "L_PTS_ROLL5": 110.2,
    "L_REB_ROLL5": 44.5,
    "L_AST_ROLL5": 25.1,
    "L_STL_ROLL5": 6.3,
    "L_BLK_ROLL5": 5.7,
    "O_PTS_ROLL5": 108.0,
    "O_REB_ROLL5": 45.0,
    "O_AST_ROLL5": 24.0,
    "O_STL_ROLL5": 7.0,
    "O_BLK_ROLL5": 5.0,
    "O_BACK_TO_BACK": 0,
    "O_DAYS_REST": 3,
}

url = "http://127.0.0.1:5000/predict"

response = requests.post(url, json={"features": test_features})

print(response.json())
