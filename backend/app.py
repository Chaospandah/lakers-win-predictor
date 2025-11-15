"""Flask backend exposing health, generic predict, and next-game prediction endpoints."""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams as nba_teams

from feature_builder import build_features_for_matchup

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Paths & static resources
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(DATA_DIR, "lakers_win_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "lakers_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(DATA_DIR, "lakers_feature_cols.pkl")
ROLLING_DATA_PATH = os.path.join(DATA_DIR, "all_teams_past_seasons_with_rolling.csv")

LAKERS_TEAM_ID = 1610612747

# Map TEAM_ID -> team abbreviation so we can describe the opponent
TEAMS_LIST = nba_teams.get_teams()
TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in TEAMS_LIST}


def _load_joblib(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    print(f"Warning: file not found at {path}")
    return None


model = _load_joblib(MODEL_PATH)
scaler = _load_joblib(SCALER_PATH)
feature_columns = _load_joblib(FEATURE_COLS_PATH)

if os.path.exists(ROLLING_DATA_PATH):
    all_games_df = pd.read_csv(ROLLING_DATA_PATH)
    all_games_df["GAME_DATE"] = pd.to_datetime(all_games_df["GAME_DATE"])
else:
    all_games_df = None
    print(f"Warning: Rolling dataset not found at {ROLLING_DATA_PATH}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepare_feature_array(raw_features) -> np.ndarray:
    """Validate and order incoming features, returning the scaled array."""
    if feature_columns:
        if isinstance(raw_features, dict):
            missing = [col for col in feature_columns if col not in raw_features]
            if missing:
                raise ValueError(f"Missing feature values for: {missing}")
            ordered_values = [raw_features[col] for col in feature_columns]
        else:
            ordered_values = list(raw_features)
            if len(ordered_values) != len(feature_columns):
                raise ValueError(
                    f"Expected {len(feature_columns)} feature values but received {len(ordered_values)}"
                )
    else:
        ordered_values = np.array(raw_features).reshape(-1).tolist()

    features = np.array(ordered_values, dtype=float).reshape(1, -1)

    if scaler is not None and feature_columns:
        features_df = pd.DataFrame(features, columns=feature_columns)
        features = scaler.transform(features_df)

    return features


def _predict_from_features(features: np.ndarray):
    """Return (prediction_int, win_probability_float)."""
    if model is None:
        raise RuntimeError("Model not loaded")

    prediction_raw = model.predict(features)[0]
    class_probs = model.predict_proba(features)[0]
    model_classes = list(getattr(model, "classes_", []))

    if model_classes:
        if 1 in model_classes:
            target_class = 1
        elif "W" in model_classes:
            target_class = "W"
        else:
            target_class = model_classes[-1]
        prob_index = model_classes.index(target_class)
        probability = float(class_probs[prob_index])
    else:
        probability = float(class_probs[-1])

    prediction = 1 if prediction_raw in (1, "W", True) else 0
    return prediction, probability


def find_next_lakers_game(max_days_ahead: int = 30):
    """Use nba_api scoreboard endpoint to find the next scheduled Lakers game."""
    today = datetime.today().date()

    for i in range(max_days_ahead):
        target_date = today + timedelta(days=i)
        date_str = target_date.strftime("%m/%d/%Y")

        try:
            sb = ScoreboardV2(game_date=date_str)
            games = sb.game_header.get_data_frame()
        except Exception:
            continue

        mask = (games["HOME_TEAM_ID"] == LAKERS_TEAM_ID) | (games["VISITOR_TEAM_ID"] == LAKERS_TEAM_ID)
        lal_games = games[mask]

        if not lal_games.empty:
            game = lal_games.iloc[0]
            home_id = int(game["HOME_TEAM_ID"])
            visitor_id = int(game["VISITOR_TEAM_ID"])

            if home_id == LAKERS_TEAM_ID:
                home_flag = 1
                opponent_id = visitor_id
            else:
                home_flag = 0
                opponent_id = home_id

            return target_date, home_flag, opponent_id

    raise RuntimeError("No upcoming Lakers game found in next 30 days.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Lakers win predictor backend running"}), 200


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({"status": "error", "error": "Model not loaded"}), 500
    if all_games_df is None:
        return jsonify({"status": "error", "error": "Rolling dataset not loaded"}), 500
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    raw_features = data.get("features")
    if raw_features is None:
        return jsonify({"error": "Missing 'features' in request"}), 400

    try:
        features = _prepare_feature_array(raw_features)
        prediction, probability = _predict_from_features(features)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({
        "prediction": prediction,
        "probability": probability
    }), 200


@app.route("/next-game-prediction", methods=["GET"])
def next_game_prediction():
    if all_games_df is None:
        return jsonify({"error": "Rolling dataset not loaded"}), 500

    try:
        game_date, home_flag, opponent_id = find_next_lakers_game()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    features_vector = build_features_for_matchup(
        all_games_df=all_games_df,
        game_date=game_date,
        lakers_team_id=LAKERS_TEAM_ID,
        opponent_team_id=opponent_id,
        home_flag=home_flag,
    )

    if feature_columns:
        raw_features = dict(zip(feature_columns, features_vector))
    else:
        raw_features = features_vector

    try:
        features = _prepare_feature_array(raw_features)
        prediction, probability = _predict_from_features(features)
    except Exception as exc:
        return jsonify({"error": f"Failed to score next game: {exc}"}), 500

    opponent_abbr = TEAM_ID_TO_ABBR.get(opponent_id, "UNKNOWN")

    return jsonify({
        "opponent": opponent_abbr,
        "opponent_id": opponent_id,
        "game_date": game_date.isoformat(),
        "home": bool(home_flag),
        "prediction": prediction,
        "win_probability": probability
    }), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
