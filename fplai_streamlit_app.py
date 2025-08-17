# fpl_streamlit_app.py
# FPL AI Optimizer (v5 - Streamlit UI Version)
# By Gemini

# --- Core Libraries ---
import streamlit as st
import requests
import pandas as pd
import pulp

# --- Page Configuration ---
st.set_page_config(page_title="FPL AI Optimizer", page_icon="⚽", layout="wide")

# --- FPL LOGIC (Functions are cached to avoid re-fetching data on every interaction) ---

@st.cache_data(ttl=3600) # Cache the data for 1 hour
def fetch_live_fpl_data():
    """Fetches comprehensive player, team, and fixture data from the live FPL API."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None

    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    positions_df = pd.DataFrame(data['element_types'])
    
    current_gw = 0
    for gw in data['events']:
        if gw['is_current']:
            current_gw = gw['id']
            break
            
    return players_df, teams_df, positions_df, current_gw

def engineer_features(players_df, teams_df, positions_df):
    """Cleans data and prepares it for optimization."""
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

# --- STREAMLIT UI LAYOUT ---

st.title("⚽ FPL AI Optimizer")
st.write("This app uses real-time data to find the optimal Fantasy Premier League squad for the upcoming gameweek.")

if st.button("🚀 Generate My Optimal Squad", type="primary"):
    
    with st.spinner("🧠 Running the AI Optimizer... This might take a moment."):
        players, teams, positions, current_gw = fetch_live_fpl_data()
        
        if players is None:
            st.error("Failed to load data. Please try again later.")
        else:
            available_players = engineer_features(players, teams, positions)
            predicted_players = create_simulated_prediction(available_players)
            final_squad = optimize_squad(predicted_players)

            st.success(f"Optimal Squad Found for Gameweek {current_gw}!")
            
            total_xp = final_squad['xP'].sum()
            total_cost = final_squad['now_cost'].sum() / 10.0

            col1, col2 = st.columns(2)
            col1.metric("Predicted Points", f"{total_xp:.2f}")
            col2.metric("Total Squad Cost", f"£{total_cost:.1f}m")
            
            display_cols = ['web_name', 'position', 'team_name', 'now_cost', 'xP']
            final_squad['now_cost'] = final_squad['now_cost'] / 10.0
            final_squad['xP'] = round(final_squad['xP'], 2)
            
            display_df = final_squad[display_cols].rename(columns={
                'web_name': 'Player',
                'position': 'Pos',
                'team_name': 'Team',
                'now_cost': 'Price (£m)',
                'xP': 'xP'
            }).reset_index(drop=True)
            
            st.dataframe(display_df, use_container_width=True)
            
            st.subheader("💡 FPL Chip Strategy Guide")
            tc_candidate = final_squad.loc[final_squad['xP'].idxmax()]
            st.info(f"**This Week's Triple Captain Pick:** {tc_candidate['web_name']} ({tc_candidate['team_name']}) with a predicted score of {tc_candidate['xP']:.2f} points.")
            
            with st.expander("See General Chip Strategy Advice"):
                st.markdown("""
                - **Wildcard (WC):** Use the first one around GW 4-8 to adapt to early season form. Use the second one late in the season (GW 30+) to prepare for a big Double Gameweek.
                - **Bench Boost (BB):** Only use this in a **Double Gameweek (DGW)** when you have 15 players who are all playing twice.
                - **Triple Captain (TC):** Best saved for a **Double Gameweek (DGW)** on a premium player with two favorable fixtures.
                - **Free Hit (FH):** Best used to navigate a **Blank Gameweek (BGW)** where many of your players don't have a match.
                """)
