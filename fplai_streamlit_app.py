# fpl_streamlit_app.py
# FPL AI Optimizer (v9 - Final UI Fix)
# By Gemini

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
    """Displays a single player in a card format within a specified container."""
    
    with container:
        # Use a container with a border for the card effect
        with st.container(border=True):
            
            # Use columns for layout: image on the left, info on the right
            col1, col2 = st.columns([1, 2])
            
            with col1:
                player_image_url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{player_series['code']}.png"
                st.image(player_image_url, width=100)
            
            with col2:
                st.markdown(f"**{player_series['web_name']}**")
                st.markdown(f"<small>{player_series['team_name']} | {player_series['position']}</small>", unsafe_allow_html=True)
                st.metric(label="Predicted Points (xP)", value=f"{player_series['xP']:.2f}")

# --- MAIN STREAMLIT APP ---

st.title("⚽ FPL AI Optimizer Dashboard")
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
            starting_11, bench = get_starting_lineup(final_squad)

            st.success(f"Optimal Squad Found for Gameweek {current_gw}!")
            
            total_xp = final_squad['xP'].sum()
            total_cost = final_squad['now_cost'].sum() / 10.0

            col1, col2 = st.columns(2)
            col1.metric("Predicted Points (Full Squad)", f"{total_xp:.2f}")
            col2.metric("Total Squad Cost", f"£{total_cost:.1f}m")
            
            # --- Display the Dashboard UI ---
            st.markdown("---")
            st.header("⭐ Starting XI")
            
            c1, c2, c3, c4 = st.columns(4)
            starting_players_list = starting_11.sort_values(by='element_type').to_dict('records')

            for i, player in enumerate(starting_players_list):
                if i % 4 == 0:
                    display_player_card(player, c1)
                elif i % 4 == 1:
                    display_player_card(player, c2)
                elif i % 4 == 2:
                    display_player_card(player, c3)
                else:
                    display_player_card(player, c4)

            st.markdown("---")
            st.header("🪑 Substitutes")

            bench_cols = st.columns(4)
            bench_players_list = bench.to_dict('records')

            for i, player in enumerate(bench_players_list):
                display_player_card(player, bench_cols[i])

            # --- Display Chip Strategy ---
            st.markdown("---")
            st.subheader("💡 FPL Chip Strategy Guide")
            tc_candidate = final_squad.sort_values(by='xP', ascending=False).iloc[0]
            st.info(f"**This Week's Triple Captain Pick:** {tc_candidate['web_name']} ({tc_candidate['team_name']}) with a predicted score of {tc_candidate['xP']:.2f} points.")
            
            with st.expander("See General Chip Strategy Advice"):
                st.markdown("""
                - **Wildcard (WC):** Use the first one around GW 4-8. Use the second one late in the season (GW 30+) to prepare for a big Double Gameweek.
                - **Bench Boost (BB):** Only use this in a **Double Gameweek (DGW)** when you have 15 players who are all playing twice.
                - **Triple Captain (TC):** Best saved for a **Double Gameweek (DGW)** on a premium player with two favorable fixtures.
                - **Free Hit (FH):** Best used to navigate a **Blank Gameweek (BGW)** where many of your players don't have a match.
                """)
