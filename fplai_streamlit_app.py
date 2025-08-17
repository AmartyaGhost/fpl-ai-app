import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pulp
from typing import Dict, List, Tuple, Optional

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

    def get_live_manager_team(self, manager_id, gameweek):
        """Gets a manager's live team and points."""
        live_data = self.get_live_gameweek_data(gameweek)
        elements = pd.DataFrame(live_data['elements'])
        
        # Add player names from bootstrap data
        bootstrap_data = self.get_bootstrap_data()
        player_names = pd.DataFrame(bootstrap_data['elements'])[['id', 'web_name']]
        elements = elements.merge(player_names, left_on='id', right_on='id', how='left')
        
        # For this demo, we'll show the highest-scoring players of the live gameweek
        top_players = elements.sort_values(by='stats.total_points', ascending=False).head(15)
        return top_players[['web_name', 'stats.total_points']].rename(columns={'web_name': 'Player', 'stats.total_points': 'Points'})


class FPLPredictor:
    def __init__(self):
        self.dm = FPLDataManager()

    def generate_predictions(self):
        """Creates a DataFrame of all players with features and predicted points."""
        bootstrap_data = self.dm.get_bootstrap_data()
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
        
        return players[['id', 'web_name', 'team', 'position', 'cost', 'predicted_points', 'team_name', 'team_short_name', 'team_code']]


class SquadOptimizer:
    def __init__(self):
        self.position_counts = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        self.squad_size = 15
        self.max_players_per_team = 3

    def optimize_squad(self, players_df: pd.DataFrame, budget: float = 100.0):
        """Uses PuLP to find the optimal 15-player squad."""
        prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
        player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') 
                       for idx in players_df.index}
        
        prob += pulp.lpSum(players_df.loc[idx, 'predicted_points'] * player_vars[idx] for idx in players_df.index)
        
        prob += pulp.lpSum(players_df.loc[idx, 'cost'] * player_vars[idx] for idx in players_df.index) <= budget
        prob += pulp.lpSum(player_vars.values()) == self.squad_size
        
        for position, count in self.position_counts.items():
            prob += pulp.lpSum(player_vars[idx] for idx in players_df.index if players_df.loc[idx, 'position'] == position) == count

        for team_id in players_df['team'].unique():
            prob += pulp.lpSum(player_vars[idx] for idx in players_df.index if players_df.loc[idx, 'team'] == team_id) <= self.max_players_per_team

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
        cols = st.columns(formation[3])
        for i, (_, player) in enumerate(starters[starters['position'] == 'FWD'].iterrows()):
             with cols[i]:
                st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)

        # Midfielders
        cols = st.columns(formation[2])
        for i, (_, player) in enumerate(starters[starters['position'] == 'MID'].iterrows()):
            with cols[i]:
                st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)
        
        # Defenders
        cols = st.columns(formation[1])
        for i, (_, player) in enumerate(starters[starters['position'] == 'DEF'].iterrows()):
            with cols[i]:
                st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)
        
        # Goalkeepers
        cols = st.columns(formation[0])
        for i, (_, player) in enumerate(starters[starters['position'] == 'GK'].iterrows()):
            with cols[i]:
                st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=70><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Display Bench
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Substitutes</h4>", unsafe_allow_html=True)
    cols = st.columns(4)
    bench_sorted = bench.sort_values(by=['position', 'predicted_points'], ascending=[True, False])
    for i, (_, player) in enumerate(bench_sorted.iterrows()):
        with cols[i]:
            st.markdown(f"<div class='player-card'><img src='{get_shirt_url(player['team_code'])}' width=60><br><span class='player-name'>{player['web_name']}</span><br><span class='player-info'>{player['team_short_name']} | £{player['cost']:.1f}m</span></div>", unsafe_allow_html=True)


def get_live_scores():
    """Mock function for live scores. Replace with a real API call."""
    matches = [
        {"home": "Arsenal", "away": "Tottenham", "score": "1-0", "time": "65'", "status": "LIVE"},
        {"home": "Man City", "away": "Liverpool", "score": "0-0", "time": "HT", "status": "HALF_TIME"},
        {"home": "Chelsea", "away": "Man Utd", "score": "-", "time": "20:30 IST", "status": "SCHEDULED"},
        {"home": "Everton", "away": "Fulham", "score": "2-1", "time": "FT", "status": "FINISHED"},
    ]
    return matches

# --- STREAMLIT APP PAGES ---

def optimizer_page():
    st.title("Gameweek Optimizer 🔮")
    st.markdown("This tool analyzes all FPL players to suggest the optimal 15-player squad based on your budget.")
    
    predictor = FPLPredictor()
    optimizer = SquadOptimizer()

    with st.spinner("Fetching player data and generating predictions..."):
        players_df = predictor.generate_predictions()

    budget = st.sidebar.slider("Set Your Budget (£m)", min_value=80.0, max_value=105.0, value=100.0, step=0.1)
    formation_choice = st.sidebar.selectbox("Choose your formation", ["3-4-3", "3-5-2", "4-4-2", "4-3-3", "5-3-2"])
    
    if st.sidebar.button("Optimize My Squad", use_container_width=True, type="primary"):
        with st.spinner("Calculating the optimal squad... this might take a minute!"):
            optimal_squad = optimizer.optimize_squad(players_df, budget=budget)
        
        st.success("Optimal Squad Found!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            formation_map = {
                "3-4-3": [1, 3, 4, 3], "3-5-2": [1, 3, 5, 2], "4-4-2": [1, 4, 4, 2],
                "4-3-3": [1, 4, 3, 3], "5-3-2": [1, 5, 3, 2]
            }
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
            # Clear cache for live data to force a refetch
            st.cache_data.clear()

        with st.spinner("Fetching live player data..."):
            team_data = dm.get_live_manager_team(manager_id, current_gw)
            st.dataframe(team_data)

    with col2:
        st.subheader("Live Premier League Scores")
            
        live_scores = get_live_scores()
        for match in live_scores:
            status_color = {'LIVE': 'lightgreen', 'SCHEDULED': 'orange', 'HALF_TIME': 'yellow'}.get(match['status'], 'gray')
            status_icon = {'LIVE': '🟢', 'SCHEDULED': '⏰', 'HALF_TIME': '⏸️', 'FINISHED': '🏁'}.get(match['status'], '')
            
            st.markdown(f"""
            <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #262730;">
                <span style="color: {status_color};">{status_icon} {match['time']}</span><br>
                **{match['home']}** vs **{match['away']}** <span style="float: right; font-weight: bold;">{match['score']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # [FIX] Removed the time.sleep() and experimental_rerun() lines
    # The page will now be responsive and update when the user clicks a button.


def strategy_guide_page():
    st.title("Chip Strategy Guide 🧠")
    st.image("https://i.ibb.co/68BFxS8/fpl-chips.png", caption="Your FPL Chips: Wildcard, Free Hit, Bench Boost, Triple Captain")
    st.markdown("""
    Using your chips at the right time can be the difference between a good season and a great one. Here’s a guide on when to consider using them.
    
    ---
    ### Wildcard (WC) 🃏
    You get two of these. Use them to completely overhaul your team.
    - **First Wildcard (Use before GW20):** The best time is typically between **Gameweeks 4 and 8**. By then, you'll have enough data to see which players and teams are over/under-performing. Don't be afraid to use it early to fix mistakes and jump on bandwagons.
    - **Second Wildcard (Use after GW20):** This is best saved for navigating the big "Double Gameweeks" (DGWs) and "Blank Gameweeks" (BGWs) later in the season, usually around **GW28-GW36**.
    
    ---
    ### Bench Boost (BB) ⚡
    All 15 of your players score points for one week.
    - **When to use:** Only use this during a **Double Gameweek (DGW)**. The ideal time is when you can use your Wildcard the week before to build a squad of 15 players who all have two matches. This maximizes your point potential. Aim for a DGW between **GW34-GW37**.
    
    ---
    ### Triple Captain (TC) ©️
    Your captain's points are tripled instead of doubled.
    - **When to use:** Use this on a star player during a **Double Gameweek (DGW)**. A top-tier attacker (like Haaland or Salah) with two favorable fixtures is the perfect candidate. This gives them two chances to get a massive haul. Look for opportunities in DGWs around **GW25** or **GW34-37**.
    
    ---
    ### Free Hit (FH) 🆓
    Lets you make unlimited transfers for a single gameweek before your squad reverts back.
    - **When to use:** The best time is during a **Blank Gameweek (BGW)** where many teams don't have a match, and your original squad is decimated. The Free Hit allows you to field a full 11 players from the few teams that *are* playing. The biggest BGWs usually happen around **GW29** due to FA Cup clashes.
    """)

# --- Main App Navigation ---
st.sidebar.title("FPL AI Navigator")
page = st.sidebar.radio("Go to", ["Gameweek Optimizer", "Live Tracker", "Chip Strategy Guide"])

if page == "Gameweek Optimizer":
    optimizer_page()
elif page == "Live Tracker":
    live_tracker_page()
elif page == "Chip Strategy Guide":
    strategy_guide_page()
