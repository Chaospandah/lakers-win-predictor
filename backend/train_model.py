import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Load cleaned data
print("Loading Lakers matchup data...")
df = pd.read_csv("data/lakers_matchup_dataset.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Prepare features and target
# Target: WL (1 = Win, 0 = Loss)
# Features: All stats except GAME_DATE, SEASON, WL

exclude_cols = ['GAME_DATE', 'SEASON', 'WL']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].fillna(0)
y = df['WL'].fillna(0)

print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save model and scaler
os.makedirs("data", exist_ok=True)
joblib.dump(model, "data/lakers_win_model.pkl")
joblib.dump(scaler, "data/lakers_scaler.pkl")
joblib.dump(feature_cols, "data/lakers_feature_cols.pkl")

print("\nModel saved to data/lakers_win_model.pkl")
print("Scaler saved to data/lakers_scaler.pkl")
print("Feature columns saved to data/lakers_feature_cols.pkl")
