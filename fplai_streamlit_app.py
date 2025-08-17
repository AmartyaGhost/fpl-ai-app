# FPL AI Optimizer (v4 - Webhook API Version)
# By Gemini

# --- Core Libraries ---
import requests
import pandas as pd
import numpy as np
import pulp

# --- Flask App Initialization ---
# This creates the web server application.
app = Flask(__name__)

# --- FPL LOGIC (Functions remain the same) ---

def fetch_live_fpl_data():
    """Fetches comprehensive player, team, and fixture data from the live FPL API."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
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
        
    prob.solve(pulp.PULP_CBC_CMD(msg=0)) # Suppress console output

    selected_player_ids = [p.id for _, p in player_df.iterrows() if pulp.value(player_vars[p.id]) == 1]
    
    optimal_squad = player_df[player_df['id'].isin(selected_player_ids)]
    return optimal_squad.sort_values(by='element_type')

# --- NEW: Webhook Endpoint ---
# This decorator defines the URL for our webhook.
# Your frontend will send a POST request to 'http://your-server-address/generate-squad'
@app.route('/generate-squad', methods=['POST'])
def generate_squad_endpoint():
    """
    This function is triggered when the webhook is called.
    It runs the entire FPL optimization process and returns the result.
    """
    print("Webhook received! Running FPL AI Optimizer...")
    
    # 1. Get Live Data
    players, teams, positions, current_gw = fetch_live_fpl_data()
    if players is None:
        return jsonify({"error": "Could not retrieve FPL data."}), 500
        
    # 2. Engineer Features
    available_players = engineer_features(players, teams, positions)
    
    # 3. Create a Prediction Score
    predicted_players = create_simulated_prediction(available_players)
    
    # 4. Optimize Squad
    final_squad = optimize_squad(predicted_players)
    
    # 5. Prepare Data for JSON Response
    total_xp = final_squad['xP'].sum()
    total_cost = final_squad['now_cost'].sum() / 10.0
    
    display_cols = ['web_name', 'position', 'team_name', 'now_cost', 'xP']
    final_squad['now_cost'] = final_squad['now_cost'] / 10.0
    final_squad['xP'] = round(final_squad['xP'], 2)
    
    # Convert the DataFrame to a list of dictionaries, which is JSON-friendly
    squad_json = final_squad[display_cols].to_dict(orient='records')
    
    # Create the final JSON response object
    response_data = {
        "gameweek": current_gw,
        "predicted_points": round(total_xp, 2),
        "total_cost": f"£{total_cost:.1f}m",
        "optimal_squad": squad_json
    }
    
    print("Optimization complete. Sending data back to frontend.")
    
    # Send the data back to the frontend
    return jsonify(response_data)

# --- Server Execution ---
# This makes the script runnable.
if __name__ == '__main__':
    # Runs the Flask server on your local machine.
    # 'debug=True' allows for auto-reloading when you save changes.

    app.run(debug=True, port=5001)
