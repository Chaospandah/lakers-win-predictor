
import pandas as pd

# Load your Lakers data
df = pd.read_csv("data/lakers_past_seasons.csv")

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- Basic cleanup ---
# Drop duplicate rows if any
df = df.drop_duplicates()

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

# Create a SEASON column (based on year)
df['SEASON'] = df['GAME_DATE'].apply(lambda x: f"{x.year}-{x.year+1}" if x.month >= 10 else f"{x.year-1}-{x.year}")

# --- Feature Engineering ---

# 1. Create binary column for Home/Away
df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

# 2. Extract opponent team name
df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split()[-1])

# 3. Convert Win/Loss to numeric
df['WL'] = df['WL'].map({'W': 1, 'L': 0})

# 4. Sort games by date
df = df.sort_values('GAME_DATE').reset_index(drop=True)

# 5. Create rolling averages (last 5 games for some key stats)
for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
    if stat in df.columns:
        df[f'{stat}_ROLL5'] = df[stat].rolling(window=5, min_periods=1).mean()

# 6. Create "Back-to-Back" indicator
df['BACK_TO_BACK'] = (df['GAME_DATE'].diff().dt.days == 1).astype(int)

# --- Drop unnecessary columns ---
cols_to_drop = ['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

# --- Handle missing values ---
df = df.fillna(0)

# --- Save cleaned dataset ---
output_path = "data/lakers_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned data saved to {output_path}")
print("Final shape:", df.shape)
print(df.head())
