"""
College Stats Scraper for NFL Draft Value Model
================================================
Scrapes career and final season college stats from Sports Reference CFB
for all players in the merged dataset that have a cfb_id.

URL pattern: https://www.sports-reference.com/cfb/players/{cfb_id}.html

Run locally (not Colab) — takes ~4 hours for full dataset.
Saves progress every 50 players so you don't lose work if it crashes.

Usage:
    python src/scrape_college_stats.py

Reads  data/processed/merged_data.csv  (needs a 'cfb_id' column)
Writes data/raw/college_stats.csv
"""

import os
import time

import pandas as pd
import requests

# ============================================================================
# CONFIG
# ============================================================================
# Paths are resolved relative to the repository root so the script can be run
# from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(REPO_ROOT, 'data', 'processed', 'merged_data.csv')
OUTPUT_FILE = os.path.join(REPO_ROOT, 'data', 'raw', 'college_stats.csv')
SAVE_EVERY = 50  # save progress every N players
SLEEP_TIME = 3   # seconds between requests (be polite to Sports Reference)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

BASE_URL = 'https://www.sports-reference.com/cfb/players/{}.html'

# ============================================================================
# TABLE IDENTIFICATION
# ============================================================================
# Each position has stats in a specific table on the page.
# We identify tables by keywords in their column names.

# Which table to look for based on position group
POS_TABLE_MAP = {
    'QB': 'passing',
    'WR': 'receiving',
    'RB': 'rushing',
    'EDGE': 'defense',
    'DL': 'defense',
    'LB': 'defense',
    'CB': 'defense',
    'S': 'defense',
    'TE': 'receiving',
    'OL': None,  # OL don't have meaningful stats tables
}


def identify_table(tables, table_type):
    """
    Given a list of DataFrames (from pd.read_html), find the one
    matching the desired table type.
    
    Returns the matching DataFrame or None.
    """
    for df in tables:
        cols = [str(c).lower() for c in df.columns.get_level_values(-1)]
        all_cols = ' '.join(cols)
        
        if table_type == 'passing':
            if 'cmp' in cols and 'att' in cols and 'int' in cols:
                return df
        elif table_type == 'receiving':
            if 'rec' in cols or 'y/r' in cols:
                return df
        elif table_type == 'rushing':
            # RB table has rushing stats - look for att and y/a but not cmp
            if 'att' in cols and 'y/a' in cols and 'cmp' not in cols:
                return df
        elif table_type == 'defense':
            if 'solo' in cols or 'tfl' in cols or 'sk' in cols:
                return df
    
    return None


def flatten_columns(df):
    """
    Flatten multi-level column headers into single-level.
    e.g., ('Receiving', 'Rec') -> 'receiving_rec'
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # Join non-empty levels
            parts = [str(c).strip() for c in col if str(c).strip() and 'Unnamed' not in str(c)]
            if len(parts) > 1:
                new_cols.append('_'.join(parts).lower().replace(' ', '_').replace('/', '_per_'))
            else:
                new_cols.append(parts[0].lower().replace(' ', '_').replace('/', '_per_'))
        df.columns = new_cols
    else:
        df.columns = [str(c).lower().strip().replace(' ', '_').replace('/', '_per_') for c in df.columns]
    
    return df


def get_career_and_last_season(table):
    """
    Extract the Career totals row and the last regular season row.
    
    Returns: (career_dict, last_season_dict)
    """
    table = flatten_columns(table)
    
    # Find the 'season' or first column to identify rows
    first_col = table.columns[0]
    
    # Find Career row
    career_row = None
    last_season_row = None
    
    for idx, row in table.iterrows():
        val = str(row[first_col]).strip().lower()
        if val == 'career':
            career_row = row
        elif val.replace('*', '').replace('†', '').isdigit() or \
             (len(val) >= 4 and val[:4].replace('*', '').replace('†', '').isdigit()):
            # This is a regular season row (starts with a year like "2019*")
            last_season_row = row
    
    career_dict = {}
    last_dict = {}
    
    if career_row is not None:
        for col in table.columns:
            career_dict[f'career_{col}'] = career_row[col]
    
    if last_season_row is not None:
        for col in table.columns:
            last_dict[f'last_{col}'] = last_season_row[col]
    
    return career_dict, last_dict


def scrape_player(cfb_id, pos_group):
    """
    Scrape college stats for a single player.
    
    Returns: dict with career and last season stats, or empty dict on failure.
    """
    url = BASE_URL.format(cfb_id)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code != 200:
            return {'scrape_status': f'HTTP {response.status_code}'}
        
        # Parse all tables on the page
        try:
            tables = pd.read_html(response.text)
        except ValueError:
            return {'scrape_status': 'no_tables_found'}
        
        if not tables:
            return {'scrape_status': 'no_tables_found'}
        
        # Determine which table to look for
        table_type = POS_TABLE_MAP.get(pos_group, None)
        
        if table_type is None:
            return {'scrape_status': 'no_table_type_for_position'}
        
        # Find the matching table
        table = identify_table(tables, table_type)
        
        if table is None:
            # Some players might have stats under a different table
            # Try all table types as fallback
            for fallback_type in ['passing', 'receiving', 'rushing', 'defense']:
                table = identify_table(tables, fallback_type)
                if table is not None:
                    break
        
        if table is None:
            return {'scrape_status': 'table_not_found'}
        
        # Extract career and last season rows
        career_dict, last_dict = get_career_and_last_season(table)
        
        result = {'scrape_status': 'success'}
        result.update(career_dict)
        result.update(last_dict)
        
        return result
    
    except requests.exceptions.Timeout:
        return {'scrape_status': 'timeout'}
    except requests.exceptions.ConnectionError:
        return {'scrape_status': 'connection_error'}
    except Exception as e:
        return {'scrape_status': f'error: {str(e)[:100]}'}


def main():
    # Load the dataset
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter to players with cfb_id
    players = df[df['cfb_id'].notna()][['pfr_id', 'cfb_id', 'player_name', 'pos', 'pos_group', 'season']].drop_duplicates(subset='cfb_id')
    print(f"Players with cfb_id: {len(players)}")
    
    # Check for existing progress
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        already_scraped = set(existing['cfb_id'].values)
        print(f"Already scraped: {len(already_scraped)} players")
        results = existing.to_dict('records')
    else:
        already_scraped = set()
        results = []
    
    # Filter to remaining players
    remaining = players[~players['cfb_id'].isin(already_scraped)]
    print(f"Remaining to scrape: {len(remaining)}")
    
    if len(remaining) == 0:
        print("All players already scraped!")
        return
    
    # Estimate time
    est_minutes = (len(remaining) * SLEEP_TIME) / 60
    print(f"Estimated time: {est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")
    print(f"Saving progress every {SAVE_EVERY} players to {OUTPUT_FILE}")
    print("=" * 60)
    
    # Scrape
    count = 0
    success = 0
    failed = 0
    
    for idx, row in remaining.iterrows():
        cfb_id = row['cfb_id']
        player_name = row['player_name']
        pos_group = row['pos_group']
        
        count += 1
        
        # Scrape the player
        stats = scrape_player(cfb_id, pos_group)
        
        # Add identifying info
        stats['cfb_id'] = cfb_id
        stats['pfr_id'] = row['pfr_id']
        stats['player_name'] = player_name
        stats['pos_group'] = pos_group
        stats['draft_season'] = row['season']
        
        results.append(stats)
        
        if stats.get('scrape_status') == 'success':
            success += 1
            status_char = '✓'
        else:
            failed += 1
            status_char = '✗'
        
        # Progress update
        pct = count / len(remaining) * 100
        print(f"[{count}/{len(remaining)} ({pct:.1f}%)] {status_char} {player_name} ({pos_group}) - {stats.get('scrape_status', 'unknown')}")
        
        # Save progress periodically
        if count % SAVE_EVERY == 0:
            save_df = pd.DataFrame(results)
            save_df.to_csv(OUTPUT_FILE, index=False)
            print(f"  >> Saved progress: {len(save_df)} total records ({success} success, {failed} failed)")
        
        # Rate limit
        time.sleep(SLEEP_TIME)
    
    # Final save
    save_df = pd.DataFrame(results)
    save_df.to_csv(OUTPUT_FILE, index=False)
    
    print("=" * 60)
    print(f"SCRAPING COMPLETE")
    print(f"Total: {count}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()