from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd
import os
import time

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Lakers Team ID
lakers_id = 1610612747  

# List of seasons to fetch (adjust as needed)
seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

all_data = []

print("Fetching Lakers game data...")

for season in seasons:
    print(f"Getting season {season}...")
    games = LeagueGameLog(season=season)
    df = games.get_data_frames()[0]
    # Filter for Lakers games only
    df = df[df['TEAM_ID'] == lakers_id]
    df["SEASON"] = season
    all_data.append(df)
    
    # Avoid hitting API too fast
    time.sleep(1)

# Combine all seasons into one DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Save the data as a CSV
csv_path = "data/lakers_past_seasons.csv"
combined_df.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")
print("First 5 rows:")
print(combined_df.head())
