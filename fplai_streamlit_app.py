import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pulp
from typing import Dict, List

# --- Page Configuration ---
st.set_page_config(
    page_title="FPL AI Optimizer",
    page_icon="⚽",
    layout="wide"
)

# --- FPL API & PREDICTION LOGIC ---

class FPLDataManager:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.session = requests.Session()

    @st.cache_data(ttl=3600)
    def get_bootstrap_data(_self):
        """Gets the main FPL bootstrap data."""
        return _self.session.get(f"{_self.base_url}bootstrap-static/").json()

    @st.cache_data(ttl=3600)
    def get_fixtures(_self):
        """Gets all fixture data."""
        return _self.session.get(f"{_self.base_url}fixtures/").json()

    @st.cache_data(ttl=3600)
    def get_live_gameweek_data(_self, gameweek):
        """Gets live data for a specific gameweek."""
        return _self.session.get(f"{_self.base_url}event/{gameweek}/live/").json()

    @st.cache_data
    def get_team_data(_self):
        """Returns a dataframe of team data, including crests."""
        bootstrap = _self.get_bootstrap_data()
        teams_df = pd.DataFrame(bootstrap['teams'])
        teams_df['crest_url'] = teams_df['code'].apply(
            lambda code: f"https://resources.premierleague.com/premierleague/badges/70/t{code}.png"
        )
        return teams_df[['id', 'name', 'short_name', 'crest_url']]

    def get_live_manager_team(self, manager_id, gameweek):
        """Gets a manager's live team and points."""
        live_data = self.get_live_gameweek_data(gameweek)
        
        if 'elements' not in live_data or not live_data['elements']:
            return pd.DataFrame(columns=['Player', 'Points'])

        elements = pd.json_normalize(live_data['elements'])

        bootstrap_data = self.get_bootstrap_data()
        player_names = pd.DataFrame(bootstrap_data['elements'])[['id', 'web_name']]
        elements = elements.merge(player_names, on='id', how='left')
        
        top_players = elements.sort_values(by='stats.total_points', ascending=False).head(15)
        return top_players[['web_name', 'stats.total_points']].rename(columns={'web_name': 'Player', 'stats.total_points': 'Points'})


class FPLPredictor:
    def __init__(self):
        self.dm = FPLDataManager()

    @st.cache_data
    def generate_predictions(_self):
        """Creates a DataFrame of all players with features and predicted points."""
        bootstrap_data = _self.dm.get_bootstrap_data()
        players = pd.DataFrame(bootstrap_data['elements'])
        teams_df = pd.DataFrame(bootstrap_data['teams'])
        teams = teams_df.set_index('id')
        
        players = players[['id', 'web_name', 'team', 'element_type', 'now_cost', 'form', 'points_per_game', 'total_points', 'ict_index', 'team_code']]
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        players['position'] = players['element_type'].map(pos_map)
        
        players['cost'] = players['now_cost'] / 10.0
        for col in ['form', 'points_per_game', 'ict_index']:
            players[col] = pd.to_numeric(players[col], errors='coerce').fillna(0)
        
        players['predicted_points'] = (players['form'] * 0.5) + (players['points_per_game'] * 0.3) + (players['ict_index'] * 0.02)
        
        players['team_name'] = players['team'].map(teams['name'])
        players['team_short_name'] = players['team'].map(teams['short_name'])
        
        # Add mock DGW/BGW data for chip strategy logic
        dgw_teams = ['ARS', 'LIV'] # Mock: Arsenal and Liverpool have a double gameweek
        bgw_teams = ['AVL', 'WHU'] # Mock: Villa and West Ham have a blank gameweek
        players['gameweek_status'] = players['team_short_name'].apply(
            lambda x: 'DGW' if x in dgw_teams else 'BGW' if x in bgw_teams else 'Normal'
        )
        
        return players


class SquadOptimizer:
    def __init__(self):
        self.position_counts = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        self.squad_size = 15
        self.max_players_per_team = 3

    @st.cache_data
    def optimize_squad(_self, players_df: pd.DataFrame, budget: float = 100.0):
        """Uses PuLP to find the optimal 15-player squad."""
        prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
        player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') 
                       for idx in players_df.index}
        
        prob += pulp.lpSum(players_df.loc[idx, 'predicted_points'] * player_vars[idx] for idx in players_df.index)
        
        prob += pulp.lpSum(players_df.loc[idx, 'cost'] * player_vars[idx] for idx in players_df.index) <= budget
        prob += pulp.lpSum(player_vars.values()) == _self.squad_size
        
        for position, count in _self.position_counts.items():
            prob += pulp.lpSum(player_vars[idx] for idx in players_df.index if players_df.loc[idx, 'position'] == position) == count

        for team_id in players_df['team'].unique():
            prob += pulp.lpSum(player_vars[idx] for idx in players_df.index if players_df.loc[idx, 'team'] == team_id) <= _self.max_players_per_team

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        selected_indices = [idx for idx in players_df.index if player_vars[idx].value() == 1]
        squad = players_df.loc[selected_indices]
        
        return squad

# --- UI HELPER FUNCTIONS ---

def get_shirt_url(team_code):
    """Returns a URL for a team's jersey image."""
    return f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{team_code}-220.png"

def display_visual_squad(squad_df, formation):
    """Displays the squad in a pitch layout with jerseys."""
    squad_by_pos = {
        'GK': squad_df[squad_df['position'] == 'GK'],
        'DEF': squad_df[squad_df['position'] == 'DEF'],
        'MID': squad_df[squad_df['position'] == 'MID'],
        'FWD': squad_df[squad_df['position'] == 'FWD']
    }
    
    starters = pd.concat([
        squad_by_pos['GK'].sort_values('predicted_points', ascending=False).head(formation[0]),
        squad_by_pos['DEF'].sort_values('predicted_points', ascending=False).head(formation[1]),
        squad_by_pos['MID'].sort_values('predicted_points', ascending=False).head(formation[2]),
        squad_by_pos['FWD'].sort_values('predicted_points', ascending=False).head(formation[3])
    ])
    
    bench_indices = squad_df.index.difference(starters.index)
    bench = squad_df.loc[bench_indices]

    st.markdown(f"<h4 style='text-align: center;'>Starting XI (Formation: {'-'.join(map(str, formation[1:]))})</h4>", unsafe_allow_html=True)
    
    pitch_css = """
        <style>
        .pitch {
            background-color: #05A368;
            background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1h98v98H1z' fill='none' stroke='%2306925E' stroke-width='3'/%3E%3C/svg%3E");
            border: 2px solid white;
            padding: 20px;
            border-radius: 10px;
        }
        .player-card { text-align: center; margin-bottom: 10px; }
        .player-name { background-color: #37003c; color: #FFFFFF; padding: 2px 5px; border-radius: 3px; font-size: 12px; font-weight: bold; }
        .player-info { background-color: #FFFFFF; color: #000000; padding: 2px 5px; border-radius: 3px; font-size: 11px; }
        </style>
    """
    st.markdown(pitch_css, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='pitch'>", unsafe_allow_html=True)
        # Forwards
        if formation[3] > 0:
            cols = st.columns(formation[3])
            for i, (_, player) in enumerate(starters[starters['position'] == 'FWD'].iterrows()):
                 with cols[i]:
                    st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)

        # Midfielders
        if formation[2] > 0:
            cols = st.columns(formation[2])
            for i, (_, player) in enumerate(starters[starters['position'] == 'MID'].iterrows()):
                with cols[i]:
                    st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)
        
        # Defenders
        if formation[1] > 0:
            cols = st.columns(formation[1])
            for i, (_, player) in enumerate(starters[starters['position'] == 'DEF'].iterrows()):
                with cols[i]:
                    st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)
        
        # Goalkeepers
        if formation[0] > 0:
            cols = st.columns(formation[0])
            for i, (_, player) in enumerate(starters[starters['position'] == 'GK'].iterrows()):
                with cols[i]:
                    st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Substitutes</h4>", unsafe_allow_html=True)
    cols = st.columns(4)
    bench_sorted = bench.sort_values(by=['position', 'predicted_points'], ascending=[True, False])
    for i, (_, player) in enumerate(bench_sorted.iterrows()):
        with cols[i]:
            st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=60><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)

# --- STREAMLIT APP PAGES ---

def optimizer_page():
    st.title("Gameweek Optimizer 🔮")
    st.markdown("This tool analyzes all FPL players to suggest the optimal 15-player squad based on your budget.")
    
    predictor = FPLPredictor()
    optimizer = SquadOptimizer()

    if 'players_df' not in st.session_state:
        with st.spinner("Fetching player data and generating predictions..."):
            st.session_state.players_df = predictor.generate_predictions()

    players_df = st.session_state.players_df
    
    budget = st.sidebar.slider("Set Your Budget (£m)", min_value=80.0, max_value=105.0, value=100.0, step=0.1)
    formation_choice = st.sidebar.selectbox("Choose your starting formation", ["3-4-3", "3-5-2", "4-4-2", "4-3-3", "5-3-2"])
    
    if st.sidebar.button("Optimize My Squad", use_container_width=True, type="primary"):
        with st.spinner("Calculating the optimal squad... this might take a minute!"):
            optimal_squad = optimizer.optimize_squad(players_df, budget=budget)
            st.session_state.optimal_squad = optimal_squad # Save to session state
        
        st.success("Optimal Squad Found!")

    if 'optimal_squad' in st.session_state:
        optimal_squad = st.session_state.optimal_squad
        col1, col2 = st.columns([2, 1])
        with col1:
            formation_map = {"3-4-3": [1, 3, 4, 3], "3-5-2": [1, 3, 5, 2], "4-4-2": [1, 4, 4, 2],
                             "4-3-3": [1, 4, 3, 3], "5-3-2": [1, 5, 3, 2]}
            display_visual_squad(optimal_squad, formation_map[formation_choice])
        
        with col2:
            st.markdown("### Squad Analysis")
            total_cost = optimal_squad['cost'].sum()
            total_points = optimal_squad['predicted_points'].sum()
            st.metric("Total Cost", f"£{total_cost:.1f}m")
            st.metric("Predicted Points (15 players)", f"{total_points:.1f}")
            
            st.markdown("### Captaincy Picks")
            captaincy_candidates = optimal_squad.sort_values('predicted_points', ascending=False).head(5)
            captain = captaincy_candidates.iloc[0]
            vice_captain = captaincy_candidates.iloc[1]
            st.markdown(f"👑 **Captain:** {captain['web_name']} ({captain['team_short_name']}) - *xP: {captain['predicted_points']:.2f}*")
            st.markdown(f"🛡️ **Vice-Captain:** {vice_captain['web_name']} ({vice_captain['team_short_name']}) - *xP: {vice_captain['predicted_points']:.2f}*")
            
            st.markdown("### Top Players in Squad")
            st.dataframe(optimal_squad[['web_name', 'position', 'team_short_name', 'cost', 'predicted_points']]
                         .sort_values('predicted_points', ascending=False).reset_index(drop=True), height=350)

def live_tracker_page():
    st.title("Live Gameweek Tracker 📈")
    st.markdown("Track live points for players and see live Premier League scores.")
    
    dm = FPLDataManager()
    
    bootstrap = dm.get_bootstrap_data()
    teams = dm.get_team_data().set_index('short_name')
    current_gw = next((gw['id'] for gw in bootstrap['events'] if gw['is_current']), None)
    
    if not current_gw:
        st.warning("No active gameweek found. Please check back when the season is live.")
        return
        
    st.info(f"Currently tracking **Gameweek {current_gw}**.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Live Player Points")
        st.markdown("This demo shows the top 15 highest-scoring live players.")
        manager_id = st.text_input("Enter your FPL Manager ID (feature in development)", "")
        
        if st.button("Refresh Live Points"):
            st.cache_data.clear()

        with st.spinner("Fetching live player data..."):
            team_data = dm.get_live_manager_team(manager_id, current_gw)
            st.dataframe(team_data)

    with col2:
        st.subheader("Live Premier League Scores")
        
        st.markdown("""
            <style>
                @keyframes blink { 50% { opacity: 0; } }
                .blinking-dot { height: 8px; width: 8px; background-color: red; border-radius: 50%; display: inline-block; animation: blink 1s linear infinite; margin-right: 5px; }
                .match-row { display: flex; align-items: center; justify-content: space-between; padding: 10px; border-bottom: 1px solid #333; }
                .team { display: flex; align-items: center; width: 120px; }
                .team-name { margin: 0 10px; }
                .score { font-weight: bold; font-size: 1.2em; }
                .status { text-align: center; width: 50px; }
            </style>
        """, unsafe_allow_html=True)
            
        # [FIX] Updated mock data to use current PL teams (e.g., replaced 'LEE' with 'WOL')
        live_scores = [
            {"home": "CHE", "away": "CRY", "score": "0 - 0", "time": "FT", "status": "FINISHED", "date": "Sun 17 Aug"},
            {"home": "NFO", "away": "BRE", "score": "3 - 1", "time": "FT", "status": "FINISHED", "date": "Sun 17 Aug"},
            {"home": "MUN", "away": "ARS", "score": "0 - 1", "time": "73'", "status": "LIVE", "date": "Sun 17 Aug"},
            {"home": "WOL", "away": "EVE", "score": "00:30", "time": "00:30 IST", "status": "SCHEDULED", "date": "Tue 19 Aug"},
        ]
        
        current_date = ""
        for match in live_scores:
            if match['date'] != current_date:
                st.markdown(f"**{match['date']}**")
                current_date = match['date']

            home_crest = teams.loc[match['home']]['crest_url']
            away_crest = teams.loc[match['away']]['crest_url']
            live_indicator = "<span class='blinking-dot'></span>" if match['status'] == 'LIVE' else ''
            
            st.markdown(f"""
                <div class="match-row">
                    <div class="status">{live_indicator}{match['time']}</div>
                    <div class="team" style="justify-content: flex-end;">
                        <span class="team-name">{teams.loc[match['home']]['name']}</span>
                        <img src="{home_crest}" width="25">
                    </div>
                    <div class="score">{match['score']}</div>
                    <div class="team">
                        <img src="{away_crest}" width="25">
                        <span class="team-name">{teams.loc[match['away']]['name']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def get_chip_recommendation(squad_df: pd.DataFrame):
    """Generates a chip recommendation for the current gameweek."""
    if squad_df is None or squad_df.empty:
        return "Optimize a squad first to get a chip recommendation."

    # Sort squad to find top players and bench
    squad_df = squad_df.sort_values('predicted_points', ascending=False).reset_index()
    top_player = squad_df.iloc[0]
    bench = squad_df.tail(4)
    bench_points = bench['predicted_points'].sum()

    # Heuristic for Triple Captain
    if top_player['gameweek_status'] == 'DGW' and top_player['predicted_points'] > 10:
        return f"**Activate Triple Captain ©️ on {top_player['web_name']}!** He has a Double Gameweek and a very high predicted score ({top_player['predicted_points']:.1f} xP), making him an outstanding candidate."

    # Heuristic for Bench Boost
    if 'DGW' in squad_df['gameweek_status'].unique() and bench_points > 15:
        return f"**Activate Bench Boost ⚡!** You have players with a Double Gameweek and your bench is predicted to score a solid {bench_points:.1f} points. This is a great opportunity to maximize your score."

    # Heuristic for Free Hit
    if len(squad_df[squad_df['gameweek_status'] == 'BGW']) > 4:
         return "**Consider Free Hit 🆓!** A significant number of players in the optimal squad have a Blank Gameweek. Playing the Free Hit would allow you to field a full team of players with fixtures."
    
    return "**Save your chips this week.** There isn't a standout opportunity for a chip. The best strategy is to hold them for a future Double or Blank Gameweek."


def strategy_guide_page():
    st.title("Chip Strategy Guide 🧠")
    
    # [NEW] Dynamic Chip Recommendation Section
    st.subheader("This Week's Chip Recommendation")
    
    recommendation = "Optimize a squad on the 'Gameweek Optimizer' page first to get a personalized chip recommendation."
    if 'optimal_squad' in st.session_state:
        recommendation = get_chip_recommendation(st.session_state.optimal_squad)

    st.info(recommendation)
    
    st.image("https://i.ibb.co/68BFxS8/fpl-chips.png", caption="Your FPL Chips: Wildcard, Free Hit, Bench Boost, Triple Captain")
    st.markdown("""
    Here is a general guide on when to consider using your chips throughout the season.
    
    ---
    ### Wildcard (WC) 🃏
    You get two of these. Use them to completely overhaul your team.
    - **First Wildcard (Use before GW20):** The best time is typically between **Gameweeks 4 and 8**. By then, you'll have enough data to see which players and teams are over/under-performing.
    - **Second Wildcard (Use after GW20):** This is best saved for navigating the big "Double Gameweeks" (DGWs) and "Blank Gameweeks" (BGWs) later in the season, usually around **GW28-GW36**.
    
    ---
    ### Bench Boost (BB) ⚡
    All 15 of your players score points for one week.
    - **When to use:** Only use this during a **Double Gameweek (DGW)**. The ideal time is when you can use your Wildcard the week before to build a squad of 15 players who all have two matches.
    
    ---
    ### Triple Captain (TC) ©️
    Your captain's points are tripled instead of doubled.
    - **When to use:** Use this on a star player during a **Double Gameweek (DGW)**. A top-tier attacker with two favorable fixtures is the perfect candidate.
    
    ---
    ### Free Hit (FH) 🆓
    Lets you make unlimited transfers for a single gameweek before your squad reverts back.
    - **When to use:** The best time is during a **Blank Gameweek (BGW)** where many teams don't have a match, and your original squad is decimated.
    """)

# --- Main App Navigation ---
st.sidebar.title("FPL AI Navigator")
page = st.sidebar.radio("Go to", ["Gameweek Optimizer", "Chip Strategy Guide", "Live Tracker"])

if page == "Gameweek Optimizer":
    optimizer_page()
elif page == "Live Tracker":
    live_tracker_page()
elif page == "Chip Strategy Guide":
    strategy_guide_page()
