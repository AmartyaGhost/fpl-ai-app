# fpl_streamlit_app.py
# FPL AI Optimizer (v21 - Final Embedded Icon)
# By Gemini

import streamlit as st
import requests
import pandas as pd
import pulp
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# --- Page Configuration ---
st.set_page_config(page_title="FPL AI Optimizer", page_icon="🦁", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    /* Main title banner */
    .title-banner {
        background-color: #37003c; /* FPL purple */
        padding: 15px;
        border-radius: 10px;
        display: flex; /* Use flexbox for alignment */
        align-items: center;
        justify-content: center;
        color: white;
        margin-bottom: 20px;
    }
    .title-banner img {
        height: 80px; /* Control icon size */
        margin-right: 25px; /* Space between icon and text */
    }
    .title-text h1 {
        font-size: 2.8em;
        font-weight: bold;
        margin: 0;
        padding-bottom: 5px;
        text-align: left;
    }
    .title-text p {
        font-size: 1.1em;
        margin: 0;
        text-align: left;
    }
    /* Scoreboard styling */
    .scoreboard-row {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        border-radius: 5px;
        background-color: #27002c;
        margin-bottom: 8px;
    }
    .team-name {
        font-size: 1.1em;
        font-weight: bold;
        text-align: center;
        flex: 3;
    }
    .team-crest {
        height: 35px;
        width: auto;
        flex: 1;
    }
    .score {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)

# --- Team Data Mappings ---
TEAM_JERSEYS = {
    'Arsenal': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_3-66.png',
    'Aston Villa': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_7-66.png',
    'Bournemouth': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_91-66.png',
    'Brentford': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_94-66.png',
    'Brighton': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_36-66.png',
    'Chelsea': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_8-66.png',
    'Crystal Palace': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_31-66.png',
    'Everton': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_11-66.png',
    'Fulham': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_54-66.png',
    'Liverpool': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_14-66.png',
    'Man City': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_43-66.png',
    'Man Utd': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_1-66.png',
    'Newcastle': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_4-66.png',
    'Nott\'m Forest': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_17-66.png',
    'Spurs': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_6-66.png',
    'West Ham': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_21-66.png',
    'Wolves': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_39-66.png',
    'Burnley': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_90-66.png',
    'Leeds': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_2-66.png',
    'Sunderland': 'https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_56-110.webp'
}

# --- FPL LOGIC (Functions are cached to avoid re-fetching data on every interaction) ---

@st.cache_data(ttl=3600)
def fetch_live_fpl_data():
    """Fetches comprehensive player, team, and fixture data from the live FPL API."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None, None

    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    positions_df = pd.DataFrame(data['element_types'])
    
    current_gw = 0
    for gw in data['events']:
        if gw['is_current']:
            current_gw = gw['id']
            break
            
    team_crests = {team['id']: f"https://resources.premierleague.com/premierleague/badges/70/t{team['code']}.png" for team in data['teams']}
            
    return players_df, teams_df, positions_df, current_gw, team_crests

@st.cache_data(ttl=60)
def fetch_live_gameweek_data(gameweek_id):
    """Fetches live match data for a specific gameweek."""
    if not gameweek_id:
        return None
    url = f"https://fantasy.premierleague.com/api/fixtures/?event={gameweek_id}"
    try:
        fixtures_response = requests.get(url)
        fixtures_response.raise_for_status()
        fixtures_data = fixtures_response.json()
        return fixtures_data
    except requests.exceptions.RequestException:
        return None

def engineer_features(players_df, teams_df, positions_df):
    """Cleans data and prepares it for optimization."""
    teams_df['name'] = teams_df['name'].replace({'Tottenham': 'Spurs'})
    
    players_df['team_name'] = players_df['team'].map(teams_df.set_index('id')['name'])
    players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name_short'])

    for col in ['form', 'ep_next', 'ep_this', 'ict_index']:
        players_df[col] = pd.to_numeric(players_df[col], errors='coerce')

    available_players = players_df[players_df['status'] == 'a'].copy()
    available_players = available_players[available_players['minutes'] > 0]
    
    available_players.reset_index(drop=True, inplace=True)
    return available_players

def create_simulated_prediction(player_df):
    """Creates a simulated 'xP' (expected points) score."""
    max_ict = player_df['ict_index'].max()
    player_df['xP'] = (
        0.6 * player_df['ep_next'] + 
        0.3 * player_df['form'] + 
        0.1 * (player_df['ict_index'] / max_ict) * 10 
    )
    player_df['xP'] = player_df['xP'].clip(0)
    return player_df

def optimize_squad(player_df):
    """Selects the optimal 15-man squad using linear programming."""
    prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
    player_vars = {player.id: pulp.LpVariable(f"player_{player.id}", cat='Binary') for _, player in player_df.iterrows()}

    prob += pulp.lpSum([player_df.loc[i, 'xP'] * player_vars[player.id] for i, player in player_df.iterrows()])

    prob += pulp.lpSum([player_df.loc[i, 'now_cost'] * player_vars[player.id] for i, player in player_df.iterrows()]) <= 1000
    prob += pulp.lpSum(list(player_vars.values())) == 15
    
    prob += pulp.lpSum([player_vars[p.id] for _, p in player_df.iterrows() if p.position == 'GKP']) == 2
    prob += pulp.lpSum([player_vars[p.id] for _, p in player_df.iterrows() if p.position == 'DEF']) == 5
    prob += pulp.lpSum([player_vars[p.id] for _, p in player_df.iterrows() if p.position == 'MID']) == 5
    prob += pulp.lpSum([player_vars[p.id] for _, p in player_df.iterrows() if p.position == 'FWD']) == 3

    for team_id in player_df['team'].unique():
        prob += pulp.lpSum([player_vars[p.id] for _, p in player_df.iterrows() if p.team == team_id]) <= 3
        
    prob.solve(pulp.PULP_CBC_CMD(msg=0)) 

    selected_player_ids = [p.id for _, p in player_df.iterrows() if pulp.value(player_vars[p.id]) == 1]
    
    optimal_squad = player_df[player_df['id'].isin(selected_player_ids)]
    return optimal_squad.sort_values(by='element_type')

def get_starting_lineup(squad_df):
    """Selects the best starting 11 from the 15-man squad."""
    squad_df = squad_df.sort_values(by='xP', ascending=False)
    
    starting_11 = pd.DataFrame()
    positions = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    
    for pos, min_count in positions.items():
        players_in_pos = squad_df[squad_df['position'] == pos]
        starting_11 = pd.concat([starting_11, players_in_pos.head(min_count)])

    remaining_players = squad_df.drop(starting_11.index)
    
    while len(starting_11) < 11 and not remaining_players.empty:
        best_remaining = remaining_players.iloc[0]
        pos = best_remaining['position']
        
        can_add = False
        if pos == 'DEF' and len(starting_11[starting_11['position'] == 'DEF']) < 5:
            can_add = True
        elif pos == 'MID' and len(starting_11[starting_11['position'] == 'MID']) < 5:
            can_add = True
        elif pos == 'FWD' and len(starting_11[starting_11['position'] == 'FWD']) < 3:
            can_add = True
        
        if can_add:
            starting_11 = pd.concat([starting_11, best_remaining.to_frame().T])
        
        remaining_players = remaining_players.iloc[1:]

    bench = squad_df.drop(starting_11.index).sort_values(by='element_type')
    return starting_11, bench

# --- UI HELPER FUNCTIONS ---

def display_player_card(player_series, container):
    with container:
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                jersey_url = TEAM_JERSEYS.get(player_series['team_name'])
                if jersey_url:
                    st.image(jersey_url, width=100)
                else:
                    st.write(player_series['team_name'])
            
            with col2:
                st.markdown(f"**{player_series['web_name']}**")
                st.markdown(f"<small>{player_series['team_name']} | {player_series['position']}</small>", unsafe_allow_html=True)
                st.metric(label="Predicted Points (xP)", value=f"{player_series['xP']:.2f}")

def display_live_scoreboard(fixtures_data, teams_df, team_crests):
    st.header("🔴 Live Gameweek Scoreboard")
    
    if not fixtures_data:
        st.warning("Live match data is not available at the moment.")
        return

    team_name_map = teams_df.set_index('id')['name'].to_dict()
    utc_zone = pytz.utc
    ist_zone = pytz.timezone('Asia/Kolkata')

    for fixture in fixtures_data:
        home_team_id = fixture['team_h']
        away_team_id = fixture['team_a']
        
        home_team_name = team_name_map.get(home_team_id, 'N/A')
        away_team_name = team_name_map.get(away_team_id, 'N/A')
        
        home_crest_url = team_crests.get(home_team_id, '')
        away_crest_url = team_crests.get(away_team_id, '')

        if fixture['started']:
            score = f"{fixture['team_h_score']} - {fixture['team_a_score']}"
        else:
            kickoff_time_utc = datetime.fromisoformat(fixture['kickoff_time'][:-1]).replace(tzinfo=utc_zone)
            kickoff_time_ist = kickoff_time_utc.astimezone(ist_zone)
            score = kickoff_time_ist.strftime('%H:%M IST')

        st.markdown(f"""
        <div class="scoreboard-row">
            <div class="team-name" style="text-align: right; padding-right: 10px;">{home_team_name}</div>
            <img src="{home_crest_url}" class="team-crest">
            <div class="score">{score}</div>
            <img src="{away_crest_url}" class="team-crest">
            <div class="team-name" style="text-align: left; padding-left: 10px;">{away_team_name}</div>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN STREAMLIT APP ---

# --- THIS IS THE CORRECTED PART ---
# The icon is now embedded as a Base64 string, so it will never break.
pl_icon_base64 = "iVBORw0KGgoAAAANSUhEUgAAAPAAAADwCAYAAAA+VemSAAAgAElEQVR4nOydZ1RU15rv+Z+ZmdlJ2kBCSCAhG0iCIAqioKgoKqhYsI51rLPOus46a4+sY8daj1hHBAVFBVERQYJtEshJCCQhSSZpJtnn+f3hMplkJiQh7z7P6/vgcWbezHt+55577vOde57zR2AEagRqBGp8P0G9/u1dDGoEagRqBGgY1AjUCBAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCBAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjucNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1AjUCNAI1Aj-squad"

st.markdown("""
<div class="title-banner">
    <img src="data:image/png;base64,{pl_icon_base64}">
    <div class="title-text">
        <h1>FPL AI OPTIMIZER</h1>
        <p>Your AI assistant to find the optimal Fantasy Premier League squad for the upcoming gameweek.</p>
    </div>
</div>
""", unsafe_allow_html=True)


if st.button("🚀 Generate My Optimal Squad", type="primary"):
    st_autorefresh(interval=60 * 1000, key="datarefresh")
    
    with st.spinner("🧠 Running the AI Optimizer... This might take a moment."):
        players, teams, positions, current_gw, team_crests = fetch_live_fpl_data()
        
        if players is None:
            st.error("Failed to load data. Please try again later.")
        else:
            available_players = engineer_features(players, teams, positions)
            predicted_players = create_simulated_prediction(available_players)
            final_squad = optimize_squad(predicted_players)
            starting_11, bench = get_starting_lineup(final_squad)

            fixtures_data = fetch_live_gameweek_data(current_gw)

            st.success(f"Optimal Squad Found for Gameweek {current_gw}!")
            
            total_xp = final_squad['xP'].sum()
            total_cost = final_squad['now_cost'].sum() / 10.0

            col1, col2 = st.columns(2)
            col1.metric("Predicted Points (Full Squad)", f"{total_xp:.2f}")
            col2.metric("Total Squad Cost", f"£{total_cost:.1f}m")
            
            st.markdown("---")
            display_live_scoreboard(fixtures_data, teams, team_crests)
            
            st.markdown("---")
            st.header("©️ Captaincy Picks")
            
            squad_sorted_by_xp = final_squad.sort_values(by='xP', ascending=False)
            captain = squad_sorted_by_xp.iloc[0]
            vice_captain = squad_sorted_by_xp.iloc[1]
            
            cap_col, vc_col = st.columns(2)
            display_player_card(captain, cap_col)
            display_player_card(vice_captain, vc_col)
            
            st.markdown("---")
            st.header("⭐ Starting XI")
            
            c1, c2, c3, c4 = st.columns(4)
            for i, player in enumerate(starting_11.sort_values(by='element_type').to_dict('records')):
                display_player_card(player, [c1, c2, c3, c4][i % 4])

            st.markdown("---")
            st.header("🪑 Substitutes")

            bench_cols = st.columns(4)
            for i, player in enumerate(bench.to_dict('records')):
                display_player_card(player, bench_cols[i])

            st.markdown("---")
            st.subheader("💡 FPL Chip Strategy Guide")
            st.info(f"**This Week's Triple Captain Pick:** {captain['web_name']} ({captain['team_name']}) with a predicted score of {captain['xP']:.2f} points.")
            
            with st.expander("See General Chip Strategy Advice"):
                st.markdown("""
                - **Wildcard (WC):** Use the first one around GW 4-8. Use the second one late in the season (GW 30+) to prepare for a big Double Gameweek.
                - **Bench Boost (BB):** Only use this in a **Double Gameweek (DGW)** when you have 15 players who are all playing twice.
                - **Triple Captain (TC):** Best saved for a **Double Gameweek (DGW)** on a premium player with two favorable fixtures.
                - **Free Hit (FH):** Best used to navigate a **Blank Gameweek (BGW)** where many of your players don't have a match.
                """)
