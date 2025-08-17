# fpl_streamlit_app.py
# FPL AI Optimizer (v6 - Final Version with Visual UI)

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
    """
    Selects the best starting 11 from the 15-man squad.
    This is a simplified version; a more complex one would consider fixture difficulty.
    """
    # Sort by predicted points to prioritize high-scoring players
    squad_df = squad_df.sort_values(by='xP', ascending=False)
    
    starting_11 = pd.DataFrame()
    positions = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1} # Minimum players in each position
    
    # Add minimum players for each position first
    for pos, min_count in positions.items():
        players_in_pos = squad_df[squad_df['position'] == pos]
        starting_11 = pd.concat([starting_11, players_in_pos.head(min_count)])

    # Fill remaining spots with the best available players, respecting positional maximums
    remaining_players = squad_df.drop(starting_11.index)
    
    while len(starting_11) < 11:
        best_remaining = remaining_players.iloc[0]
        pos = best_remaining['position']
        
        # Respect formation constraints (e.g., max 5 defenders)
        if pos == 'DEF' and len(starting_11[starting_11['position'] == 'DEF']) < 5:
            starting_11 = pd.concat([starting_11, best_remaining.to_frame().T])
        elif pos == 'MID' and len(starting_11[starting_11['position'] == 'MID']) < 5:
            starting_11 = pd.concat([starting_11, best_remaining.to_frame().T])
        elif pos == 'FWD' and len(starting_11[starting_11['position'] == 'FWD']) < 3:
            starting_11 = pd.concat([starting_11, best_remaining.to_frame().T])
        
        remaining_players = remaining_players.iloc[1:]

    bench = squad_df.drop(starting_11.index).sort_values(by='element_type')
    return starting_11, bench


# --- UI HELPER FUNCTIONS ---

def display_player(player_series):
    """Displays a single player's image and info."""
    player_image_url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{player_series['code']}.png"
    st.image(player_image_url, width=80)
    st.markdown(f"<p style='text-align: center; font-weight: bold; font-size: 12px;'>{player_series['web_name']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 11px;'>xP: {player_series['xP']:.2f}</p>", unsafe_allow_html=True)

def display_pitch(starting_11, bench):
    """Displays the starting 11 on a football pitch and the bench."""
    
    # Custom CSS for the pitch background
    pitch_css = f"""
    <style>
    .pitch-container {{
        background-image: url('https://i.imgur.com/2T5rgU4.png');
        background-size: cover;
        background-repeat: no-repeat;
        padding: 20px;
        border-radius: 10px;
        height: 600px;
        position: relative;
    }}
    </style>
    """
    st.markdown(pitch_css, unsafe_allow_html=True)
    
    st.markdown('<div class="pitch-container">', unsafe_allow_html=True)

    # Filter players by position
    gkp = starting_11[starting_11['position'] == 'GKP']
    defs = starting_11[starting_11['position'] == 'DEF']
    mids = starting_11[starting_11['position'] == 'MID']
    fwds = starting_11[starting_11['position'] == 'FWD']

    # Display players in their positions using columns
    # Goalkeeper Row
    gkp_cols = st.columns(3)
    with gkp_cols[1]:
        for _, player in gkp.iterrows():
            display_player(player)

    # Defender Row
    def_cols = st.columns(len(defs) if len(defs) > 0 else 1)
    for i, (_, player) in enumerate(defs.iterrows()):
        with def_cols[i]:
            display_player(player)
    
    # Midfielder Row
    mid_cols = st.columns(len(mids) if len(mids) > 0 else 1)
    for i, (_, player) in enumerate(mids.iterrows()):
        with mid_cols[i]:
            display_player(player)
            
    # Forward Row
    fwd_cols = st.columns(len(fwds) if len(fwds) > 0 else 1)
    for i, (_, player) in enumerate(fwds.iterrows()):
        with fwd_cols[i]:
            display_player(player)

    st.markdown('</div>', unsafe_allow_html=True)

    # Display the Bench
    st.markdown("---")
    st.subheader("Substitutes")
    bench_cols = st.columns(4)
    for i, (_, player) in enumerate(bench.iterrows()):
        with bench_cols[i]:
            display_player(player)


# --- MAIN STREAMLIT APP ---

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
            starting_11, bench = get_starting_lineup(final_squad)

            st.success(f"Optimal Squad Found for Gameweek {current_gw}!")
            
            total_xp = final_squad['xP'].sum()
            total_cost = final_squad['now_cost'].sum() / 10.0

            col1, col2 = st.columns(2)
            col1.metric("Predicted Points (Full Squad)", f"{total_xp:.2f}")
            col2.metric("Total Squad Cost", f"£{total_cost:.1f}m")
            
            # --- Display the Pitch UI ---
            display_pitch(starting_11, bench)

            # --- Placeholder for Live Tracking Feature ---
            st.markdown("---")
            st.header("🔴 Live Gameweek Tracker")
            st.info("This feature is under development. Check back during a live match to see real-time point updates for your squad!")
            # Note: Implementing the full live feature is an advanced task.
            # It would involve using st_autorefresh and repeatedly calling a live FPL API endpoint.

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
