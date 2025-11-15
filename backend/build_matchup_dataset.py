
import os
import re
import time
from collections import defaultdict

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.static import teams as nba_teams

# ---------- CONFIG ----------
INPUT_CSV = "data/lakers_past_seasons.csv"         # your combined file
OUTPUT_CSV = "data/lakers_matchup_dataset.csv"
CACHE_DIR = "team_game_logs"                  # caches each team's season logs
ROLL_WINDOW = 5                               # last N games to use for rolling features
SLEEP_BETWEEN_API_CALLS = 0.8                 # seconds - polite to NBA API
# ----------------------------

os.makedirs(CACHE_DIR, exist_ok=True)

def load_lakers_df(path):
    df = pd.read_csv(path)
    # Ensure datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    return df

def build_abbr_to_id_map():
    """Use nba_api static teams list to build mapping from abbreviation -> team id"""
    teams_list = nba_teams.get_teams()
    abbr_to_id = {t['abbreviation'].upper(): t['id'] for t in teams_list}
    # Sometimes abbreviations vary; ensure keys exist
    return abbr_to_id

def parse_opponent_abbr(matchup):
    """
    Expect matchup strings like:
      'LAL vs. BOS'  or 'LAL @ BOS'  or 'LAL vs BOS'
    We'll extract the final token which should be the opponent abbreviation.
    """
    # last token, strip punctuation
    try:
        token = str(matchup).split()[-1]
        # remove punctuation like '.', ',' etc
        token = re.sub(r'[^A-Z0-9]', '', token.upper())
        return token
    except Exception:
        return None

def get_team_season_log(team_id, season):
    """
    Return a DataFrame for team_id-season.
    Caches to CSV as <CACHE_DIR>/<team_id>_<season>.csv
    """
    safe_season = season.replace('-', '_')
    fn = os.path.join(CACHE_DIR, f"{team_id}_{safe_season}.csv")
    if os.path.exists(fn):
        df = pd.read_csv(fn)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        return df

    # Fetch from nba_api
    print(f"Fetching team {team_id} season {season} from API...")
    try:
        gamelog = LeagueGameLog(season=season)
        df = gamelog.get_data_frames()[0]
        df = df[df['TEAM_ID'] == team_id]
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df.to_csv(fn, index=False)
        # be polite
        time.sleep(SLEEP_BETWEEN_API_CALLS)
        return df
    except Exception as e:
        print(f"Error fetching team {team_id} season {season}: {e}")
        return pd.DataFrame()  # empty DataFrame fallback

def compute_rolling_stats_from_log(df_log, current_date, stats, window=ROLL_WINDOW):
    """
    Given a team's game log DataFrame, compute rolling means for 'stats' using
    all games with GAME_DATE < current_date. Returns dictionary stat -> value.
    If there are no prior games, returns None (or 0).
    """
    prior = df_log[df_log['GAME_DATE'] < current_date].sort_values('GAME_DATE')
    if prior.empty:
        return {stat: 0.0 for stat in stats}
    # take last `window` rows
    lastN = prior.tail(window)
    means = {}
    for stat in stats:
        if stat in lastN.columns:
            means[stat] = lastN[stat].astype(float).mean()
        else:
            means[stat] = 0.0
    return means

def main():
    # Load Lakers data
    print("Loading Lakers data...")
    ldf = load_lakers_df(INPUT_CSV)
    print("Initial rows:", len(ldf))

    # Build abbreviation -> team id mapping
    abbr_to_id = build_abbr_to_id_map()

    # Standard stat columns to pull rolling means for
    candidate_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV']
    # Keep only those present in Lakers df and in opponent logs later
    candidate_stats = [s for s in candidate_stats if s in ldf.columns]

    # Add basic columns: HOME, OPP_ABBR, OPP_TEAM_ID
    ldf['HOME'] = ldf['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) or 'vs ' in str(x) else 0)
    ldf['OPP_ABBR'] = ldf['MATCHUP'].apply(parse_opponent_abbr)
    ldf['OPP_ABBR'] = ldf['OPP_ABBR'].fillna('').str.upper()

    # Map opponent abbreviation to team id (if unknown, will be NaN)
    ldf['OPP_TEAM_ID'] = ldf['OPP_ABBR'].map(lambda a: abbr_to_id.get(a, pd.NA))

    # Check for any unknown abbreviations
    unknown_abbrs = sorted(set(ldf[ldf['OPP_TEAM_ID'].isna()]['OPP_ABBR'].unique()))
    if unknown_abbrs:
        print("Warning: unknown opponent abbreviations found:", unknown_abbrs)
        print("You may need to inspect MATCHUP formatting. Unknown teams will get missing opponent features set to 0.")

    # Compute Lakers rolling features (exclude current game by shifting)
    stats_for_rolling = [s for s in ['PTS', 'REB', 'AST', 'STL', 'BLK'] if s in ldf.columns]
    for stat in stats_for_rolling:
        # shift by 1 to exclude current game, then rolling
        ldf[f'L_{stat}_ROLL{ROLL_WINDOW}'] = ldf[stat].shift(1).rolling(window=ROLL_WINDOW, min_periods=1).mean().fillna(0)

    # Create back-to-back and rest days for Lakers
    ldf['L_PREV_DATE'] = ldf['GAME_DATE'].shift(1)
    ldf['L_DAYS_REST'] = (ldf['GAME_DATE'] - ldf['L_PREV_DATE']).dt.days.fillna(999).astype(int)
    ldf['L_BACK_TO_BACK'] = (ldf['L_DAYS_REST'] == 1).astype(int)

    # Prepare output rows
    out_rows = []
    # cache for opponent season logs: {(team_id, season): df}
    opp_log_cache = {}

    total = len(ldf)
    print("Building matchup rows for each Lakers game...")
    for idx, row in ldf.iterrows():
        season = row.get('SEASON')
        game_date = row['GAME_DATE']
        opp_id = row['OPP_TEAM_ID']

        # Construct base features from Lakers rolling ones
        base = {
            'GAME_DATE': game_date,
            'SEASON': season,
            'HOME': row['HOME'],
            'L_BACK_TO_BACK': int(row['L_BACK_TO_BACK']),
            'L_DAYS_REST': int(row['L_DAYS_REST']) if 'L_DAYS_REST' in row else 999,
            'WL': row['WL'] if 'WL' in row else (1 if str(row.get('WL')).upper() == 'W' else 0)
        }
        # Add Lakers rolling stats
        for stat in stats_for_rolling:
            base[f'L_{stat}_ROLL{ROLL_WINDOW}'] = row.get(f'L_{stat}_ROLL{ROLL_WINDOW}', 0.0)

        # Default opponent stats (if unknown or no prior games)
        opp_features = {f'O_{stat}_ROLL{ROLL_WINDOW}': 0.0 for stat in stats_for_rolling}
        opp_features.update({
            'O_BACK_TO_BACK': 0,
            'O_DAYS_REST': 999
        })

        if pd.notna(opp_id):
            key = (int(opp_id), season)
            if key not in opp_log_cache:
                opp_df = get_team_season_log(int(opp_id), season)
                opp_log_cache[key] = opp_df
            else:
                opp_df = opp_log_cache[key]

            if not opp_df.empty:
                # compute opponent rolling stats before current game_date
                opp_means = compute_rolling_stats_from_log(opp_df, game_date, stats_for_rolling, window=ROLL_WINDOW)
                # compute opponent rest days/back-to-back using their game dates
                prior = opp_df[opp_df['GAME_DATE'] < game_date].sort_values('GAME_DATE')
                if prior.empty:
                    opp_last_date = None
                    opp_days_rest = 999
                    opp_b2b = 0
                else:
                    opp_last_date = prior['GAME_DATE'].iloc[-1]
                    opp_days_rest = (game_date - opp_last_date).days
                    opp_b2b = 1 if opp_days_rest == 1 else 0

                # fill opp_features
                for stat in stats_for_rolling:
                    opp_features[f'O_{stat}_ROLL{ROLL_WINDOW}'] = opp_means.get(stat, 0.0)
                opp_features['O_BACK_TO_BACK'] = int(opp_b2b)
                opp_features['O_DAYS_REST'] = int(opp_days_rest if opp_days_rest is not None else 999)

        # Merge base + opp_features
        combined = {**base, **opp_features}
        out_rows.append(combined)

        # Progress indicator occasionally
        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(f"Processed {idx+1}/{total} games...")

    # Build DataFrame and save
    out_df = pd.DataFrame(out_rows)
    # Drop rows where GAME_DATE is NaT (if any)
    out_df = out_df.dropna(subset=['GAME_DATE']).reset_index(drop=True)

    # Save
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMatchup dataset saved to {OUTPUT_CSV}")
    print("Columns:", out_df.columns.tolist())
    print("Rows:", len(out_df))

if __name__ == "__main__":
    main()
