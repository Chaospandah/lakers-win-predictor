import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
import time
import os

# Seasons you want to collect
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# Make sure data folder exists
if not os.path.exists("data"):
    os.makedirs("data")

def get_season_data(season):
    """Fetch game logs for one season."""
    print(f"Fetching season {season}...")

    logs = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season"
    ).get_data_frames()[0]

    logs["SEASON"] = season
    return logs

all_data = []

# Loop through each season
for season in SEASONS:
    df = get_season_data(season)
    all_data.append(df)
    time.sleep(1.2)  # to prevent API rate-limits

# Combine all seasons
full_df = pd.concat(all_data, ignore_index=True)

# Sort by team and date
full_df["GAME_DATE"] = pd.to_datetime(full_df["GAME_DATE"])
full_df = full_df.sort_values(["TEAM_ID", "GAME_DATE"])

# Save
output_path = "data/all_teams_past_seasons.csv"
full_df.to_csv(output_path, index=False)

print(f"\nSaved all team data to {output_path}")
