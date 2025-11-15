# backend/feature_builder.py

import pandas as pd
from datetime import datetime

# Stats we will average over the last 5 games
STATS = ["PTS", "REB", "AST", "STL", "BLK"]


def _compute_team_last5_features(all_games_df: pd.DataFrame, team_id: int, game_date: pd.Timestamp):
    """
    Given the full all-teams DataFrame, a team_id, and a game_date,
    compute:
      - last 5-game averages for PTS, REB, AST, STL, BLK
      - days of rest before game_date
      - back-to-back flag
    """
    # Ensure datetime
    if not isinstance(game_date, pd.Timestamp):
        game_date = pd.to_datetime(game_date)

    # Filter for this team's games BEFORE the game_date
    team_games = all_games_df[
        (all_games_df["TEAM_ID"] == team_id) &
        (all_games_df["GAME_DATE"] < game_date)
    ].sort_values("GAME_DATE")

    features = {}

    if team_games.empty:
        # No prior games (start of season case) -> default zeros and rest=7 days
        for stat in STATS:
            features[f"{stat}_ROLL5"] = 0.0
        features["DAYS_REST"] = 7
        features["BACK_TO_BACK"] = 0
        return features

    # Last 5 games
    last5 = team_games.tail(5)

    for stat in STATS:
        if stat in last5.columns:
            features[f"{stat}_ROLL5"] = float(last5[stat].mean())
        else:
            features[f"{stat}_ROLL5"] = 0.0

    # Days rest = difference between game_date and last game date
    last_game_date = team_games["GAME_DATE"].iloc[-1]
    days_rest = (game_date - last_game_date).days
    features["DAYS_REST"] = int(days_rest)
    features["BACK_TO_BACK"] = 1 if days_rest == 1 else 0

    return features


def build_features_for_matchup(
    all_games_df: pd.DataFrame,
    game_date,
    lakers_team_id: int,
    opponent_team_id: int,
    home_flag: int,
):
    """
    Build the 15-element feature vector for the model:

    [HOME,
     L_BACK_TO_BACK,
     L_DAYS_REST,
     L_PTS_ROLL5,
     L_REB_ROLL5,
     L_AST_ROLL5,
     L_STL_ROLL5,
     L_BLK_ROLL5,
     O_PTS_ROLL5,
     O_REB_ROLL5,
     O_AST_ROLL5,
     O_STL_ROLL5,
     O_BLK_ROLL5,
     O_BACK_TO_BACK,
     O_DAYS_REST]
    """

    if not isinstance(game_date, pd.Timestamp):
        game_date = pd.to_datetime(game_date)

    # Compute Lakers last-5 stats
    lakers_feats = _compute_team_last5_features(all_games_df, lakers_team_id, game_date)
    # Compute Opponent last-5 stats
    opp_feats = _compute_team_last5_features(all_games_df, opponent_team_id, game_date)

    # Map into the exact feature order your model expects
    HOME = int(home_flag)

    L_BACK_TO_BACK = int(lakers_feats["BACK_TO_BACK"])
    L_DAYS_REST = int(lakers_feats["DAYS_REST"])
    L_PTS_ROLL5 = float(lakers_feats["PTS_ROLL5"])
    L_REB_ROLL5 = float(lakers_feats["REB_ROLL5"])
    L_AST_ROLL5 = float(lakers_feats["AST_ROLL5"])
    L_STL_ROLL5 = float(lakers_feats["STL_ROLL5"])
    L_BLK_ROLL5 = float(lakers_feats["BLK_ROLL5"])

    O_PTS_ROLL5 = float(opp_feats["PTS_ROLL5"])
    O_REB_ROLL5 = float(opp_feats["REB_ROLL5"])
    O_AST_ROLL5 = float(opp_feats["AST_ROLL5"])
    O_STL_ROLL5 = float(opp_feats["STL_ROLL5"])
    O_BLK_ROLL5 = float(opp_feats["BLK_ROLL5"])
    O_BACK_TO_BACK = int(opp_feats["BACK_TO_BACK"])
    O_DAYS_REST = int(opp_feats["DAYS_REST"])

    feature_vector = [
        HOME,
        L_BACK_TO_BACK,
        L_DAYS_REST,
        L_PTS_ROLL5,
        L_REB_ROLL5,
        L_AST_ROLL5,
        L_STL_ROLL5,
        L_BLK_ROLL5,
        O_PTS_ROLL5,
        O_REB_ROLL5,
        O_AST_ROLL5,
        O_STL_ROLL5,
        O_BLK_ROLL5,
        O_BACK_TO_BACK,
        O_DAYS_REST,
    ]

    return feature_vector
