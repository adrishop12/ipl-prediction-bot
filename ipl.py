import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import os
import logging
import streamlit as st

# Step 1: Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Step 2: Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PREF_FILE = os.path.join(BASE_DIR, 'preferences.json')
os.makedirs(MODEL_DIR, exist_ok=True)

# Step 3: Load datasets
try:
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    logger.info("Datasets loaded successfully")
except FileNotFoundError:
    st.error("Error: matches.csv or deliveries.csv not found in the current directory.")
    logger.error("Datasets not found")
    raise

# Step 4: Preprocess matches.csv
matches = matches.dropna(subset=['winner'])
le_team = LabelEncoder()
le_venue = LabelEncoder()
all_teams = pd.concat([matches['team1'], matches['team2'], matches['winner']]).unique()
le_team.fit(all_teams)

matches['team1_encoded'] = le_team.transform(matches['team1'])
matches['team2_encoded'] = le_team.transform(matches['team2'])
matches['venue_encoded'] = le_venue.fit_transform(matches['venue'])

matches['toss_winner_encoded'] = matches.apply(
    lambda x: 1 if x['toss_winner'] == x['team1'] else 0, axis=1
)
le_toss_decision = LabelEncoder()
matches['toss_decision_encoded'] = le_toss_decision.fit_transform(matches['toss_decision'])

matches['team1_matches'] = matches.groupby('team1').cumcount() + 1
matches['team2_matches'] = matches.groupby('team2').cumcount() + 1
matches['team1_wins'] = matches.apply(
    lambda x: 1 if x['winner'] == x['team1'] else 0, axis=1
).groupby(matches['team1']).cumsum()
matches['team2_wins'] = matches.apply(
    lambda x: 1 if x['winner'] == x['team2'] else 0, axis=1
).groupby(matches['team2']).cumsum()

matches['team1_win_pct'] = matches['team1_wins'] / matches['team1_matches']
matches['team2_win_pct'] = matches['team2_wins'] / matches['team2_matches']
matches['team1_win_pct'] = matches['team1_win_pct'].fillna(0.5)
matches['team2_win_pct'] = matches['team2_win_pct'].fillna(0.5)
matches['win_pct_diff'] = matches['team1_win_pct'] - matches['team2_win_pct']

features = [
    'team1_encoded', 'team2_encoded', 'venue_encoded',
    'toss_winner_encoded', 'toss_decision_encoded',
    'team1_win_pct', 'team2_win_pct', 'win_pct_diff'
]
X = matches[features]
matches['winner_encoded'] = matches.apply(
    lambda x: 1 if x['winner'] == x['team1'] else 0, axis=1
)
y = matches['winner_encoded']

# Step 5: Train team win prediction model
if not os.path.exists(os.path.join(MODEL_DIR, 'team_model.pkl')):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    team_model = RandomForestClassifier(n_estimators=100, random_state=42)
    team_model.fit(X_train, y_train)
    y_pred = team_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Team Win Prediction Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(team_model, os.path.join(MODEL_DIR, 'team_model.pkl'))
    joblib.dump(le_team, os.path.join(MODEL_DIR, 'le_team.pkl'))
    joblib.dump(le_venue, os.path.join(MODEL_DIR, 'le_venue.pkl'))
    joblib.dump(le_toss_decision, os.path.join(MODEL_DIR, 'le_toss_decision.pkl'))
    logger.info(f"Team model and encoders saved to {MODEL_DIR}")
else:
    team_model = joblib.load(os.path.join(MODEL_DIR, 'team_model.pkl'))
    le_team = joblib.load(os.path.join(MODEL_DIR, 'le_team.pkl'))
    le_venue = joblib.load(os.path.join(MODEL_DIR, 'le_venue.pkl'))
    le_toss_decision = joblib.load(os.path.join(MODEL_DIR, 'le_toss_decision.pkl'))
    logger.info("Loaded existing team model and encoders")

# Step 6: Preprocess deliveries.csv for player stats
batsman_stats = deliveries.groupby(['match_id', 'batter']).agg({
    'batsman_runs': 'sum',
    'ball': 'count',
    'is_wicket': 'sum'
}).reset_index()

batsman_stats['strike_rate'] = (batsman_stats['batsman_runs'] / batsman_stats['ball']) * 100
batsman_stats['strike_rate'] = batsman_stats['strike_rate'].fillna(0)
batsman_stats['batting_avg'] = batsman_stats['batsman_runs'] / batsman_stats['is_wicket']
batsman_stats['batting_avg'] = batsman_stats['batting_avg'].replace([np.inf, -np.inf], 0).fillna(0)

batsman_form = batsman_stats.sort_values('match_id')
batsman_form = batsman_form.groupby('batter').tail(5).groupby('batter').agg({
    'batsman_runs': 'sum',
    'batting_avg': 'mean',
    'strike_rate': 'mean'
}).reset_index()
batsman_form.columns = ['batter', 'runs_last_5', 'batting_avg', 'strike_rate']

player_teams = deliveries.groupby('batter').last()[['batting_team']].reset_index()
player_teams.columns = ['batter', 'team']
batsman_form = batsman_form.merge(player_teams, on='batter', how='left')

# Define player features used for prediction
player_features = ['runs_last_5', 'batting_avg', 'strike_rate']

# Step 7: Train player performance model
if not os.path.exists(os.path.join(MODEL_DIR, 'player_model.pkl')):
    player_features = ['runs_last_5', 'batting_avg', 'strike_rate']
    X_player = batsman_form[player_features].fillna(0)
    y_player = batsman_form['runs_last_5']

    player_model = LinearRegression()
    player_model.fit(X_player, y_player)
    joblib.dump(player_model, os.path.join(MODEL_DIR, 'player_model.pkl'))
    logger.info(f"Player model saved to {MODEL_DIR}/player_model.pkl")
else:
    player_model = joblib.load(os.path.join(MODEL_DIR, 'player_model.pkl'))
    logger.info("Loaded existing player model")

# Step 8: Scrape team squads from iplt20.com and extract details with BeautifulSoup
def scrape_team_squad(team_url_name):
    url = f"https://www.iplt20.com/teams/{team_url_name}/squad"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            player_cards = soup.find_all('li', class_='ih-pcard1')
            players = {}
            
            for card in player_cards:
                name_elem = card.find('div', class_='ih-p-name')
                if name_elem and name_elem.find('h2'):
                    name = name_elem.find('h2').text.strip()
                    role_elem = card.find('span', class_='d-block w-100 text-center')
                    role = role_elem.text.strip() if role_elem else "unknown"
                    
                    # Check for special icons
                    icons = card.find('div', class_='teams-icon')
                    is_foreign = bool(icons and icons.find('img', src=lambda x: x and 'foreign-player-icon' in x))
                    is_captain = bool(icons and icons.find('img', src=lambda x: x and 'captain-icon' in x))
                    
                    players[name] = {
                        "name": name,
                        "role": role,
                        "is_foreign": is_foreign,
                        "is_captain": is_captain
                    }
            
            logger.info(f"Scraped squad for {team_url_name}: {len(players)} players")
            return players
        else:
            logger.error(f"Failed to scrape {team_url_name} (status {response.status_code})")
            return {}
    except Exception as e:
        logger.error(f"Error scraping {team_url_name}: {e}")
        return {}

# IPL 2025 teams
ipl_teams = {
    "Royal Challengers Bengaluru": "royal-challengers-bengaluru",
    "Delhi Capitals": "delhi-capitals",
    "Mumbai Indians": "mumbai-indians",
    "Chennai Super Kings": "chennai-super-kings",
    "Kolkata Knight Riders": "kolkata-knight-riders",
    "Sunrisers Hyderabad": "sunrisers-hyderabad",
    "Rajasthan Royals": "rajasthan-royals",
    "Punjab Kings": "punjab-kings",
    "Gujarat Titans": "gujarat-titans",
    "Lucknow Super Giants": "lucknow-super-giants"
}

# Load or scrape squads
SQUAD_FILE = os.path.join(BASE_DIR, 'team_squads.json')
if not os.path.exists(SQUAD_FILE):
    team_squads = {}
    for team_name, team_url_name in ipl_teams.items():
        squad = scrape_team_squad(team_url_name)
        if squad:
            team_squads[team_name] = squad
    with open(SQUAD_FILE, 'w') as f:
        json.dump(team_squads, f)
    logger.info("Squads saved to team_squads.json")
else:
    with open(SQUAD_FILE, 'r') as f:
        team_squads = json.load(f)
    logger.info("Loaded existing team squads from team_squads.json")

# Step 9: Streamlit GUI
def save_preferences(team1, team2, venue, toss_winner, toss_decision):
    prefs = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision
    }
    with open(PREF_FILE, 'w') as f:
        json.dump(prefs, f)
    logger.info(f"Saved preferences: {prefs}")

def load_preferences():
    if os.path.exists(PREF_FILE):
        with open(PREF_FILE, 'r') as f:
            return json.load(f)
    return {"team1": "Royal Challengers Bengaluru", "team2": "Delhi Capitals", 
            "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bengaluru", 
            "toss_decision": "bat"}

def search_players(query):
    results = []
    for team, squad in team_squads.items():
        for player, details in squad.items():
            if query.lower() in player.lower():
                results.append({"name": player, "team": team, "role": details["role"]})
    return results

def main():
    st.title("IPL Match Predictor")

    # Sidebar for preferences and search
    st.sidebar.header("Preferences")
    prefs = load_preferences()
    
    team1 = st.sidebar.selectbox("Team 1", list(ipl_teams.keys()), index=list(ipl_teams.keys()).index(prefs["team1"]))
    team2 = st.sidebar.selectbox("Team 2", list(ipl_teams.keys()), index=list(ipl_teams.keys()).index(prefs["team2"]))
    venue = st.sidebar.selectbox("Venue", matches['venue'].unique(), index=list(matches['venue'].unique()).index(prefs["venue"]))
    toss_winner = st.sidebar.selectbox("Toss Winner", list(ipl_teams.keys()), index=list(ipl_teams.keys()).index(prefs["toss_winner"]))
    toss_decision = st.sidebar.selectbox("Toss Decision", ['bat', 'field'], index=['bat', 'field'].index(prefs["toss_decision"]))

    if st.sidebar.button("Save Preferences"):
        save_preferences(team1, team2, venue, toss_winner, toss_decision)
        st.sidebar.success("Preferences saved!")

    st.sidebar.header("Player Search")
    search_query = st.sidebar.text_input("Enter player name")
    if search_query:
        results = search_players(search_query)
        st.sidebar.write("Search Results:")
        for result in results:
            st.sidebar.write(f"{result['name']} ({result['team']}) - {result['role']}")

    # Main prediction
    if st.button("Predict Match"):
        if team1 == team2:
            st.error("Team 1 and Team 2 must be different.")
            logger.error("Same teams selected")
            return

        team1_matches = matches[matches['team1'] == team1].shape[0] + matches[matches['team2'] == team1].shape[0]
        team1_wins = matches[matches['winner'] == team1].shape[0]
        team2_matches = matches[matches['team1'] == team2].shape[0] + matches[matches['team2'] == team2].shape[0]
        team2_wins = matches[matches['winner'] == team2].shape[0]

        team1_win_pct = team1_wins / team1_matches if team1_matches > 0 else 0.5
        team2_win_pct = team2_wins / team2_matches if team2_matches > 0 else 0.5

        today_match = pd.DataFrame({
            'team1': [team1],
            'team2': [team2],
            'venue': [venue],
            'toss_winner': [toss_winner],
            'toss_decision': [toss_decision],
            'team1_win_pct': [team1_win_pct],
            'team2_win_pct': [team2_win_pct]
        })

        try:
            today_match['team1_encoded'] = le_team.transform(today_match['team1'])
            today_match['team2_encoded'] = le_team.transform(today_match['team2'])
        except ValueError as e:
            st.error(f"Error: Team {team1} or {team2} not in training data.")
            logger.error(f"Team encoding error: {e}")
            return

        if venue not in le_venue.classes_:
            st.warning(f"Venue '{venue}' not found. Using first venue: {le_venue.classes_[0]}.")
            today_match['venue'] = le_venue.classes_[0]
        today_match['venue_encoded'] = le_venue.transform(today_match['venue'])

        today_match['toss_winner_encoded'] = today_match.apply(
            lambda x: 1 if x['toss_winner'] == x['team1'] else 0, axis=1
        )
        try:
            today_match['toss_decision_encoded'] = le_toss_decision.transform(today_match['toss_decision'])
        except ValueError:
            st.warning(f"Toss decision '{toss_decision}' not found. Defaulting to 'bat'.")
            today_match['toss_decision_encoded'] = le_toss_decision.transform(['bat'])

        today_match['win_pct_diff'] = today_match['team1_win_pct'] - today_match['team2_win_pct']

        today_X = today_match[features]
        win_prob = team_model.predict_proba(today_X)[0]
        st.write(f"**Selected Match**: {team1} vs {team2}")
        st.write(f"{team1} Win Probability: {win_prob[1] * 100:.2f}%")
        st.write(f"{team2} Win Probability: {win_prob[0] * 100:.2f}%")
        logger.info(f"Predicted match: {team1} vs {team2}, {team1} {win_prob[1] * 100:.2f}%, {team2} {win_prob[0] * 100:.2f}%")

        squad_team1 = [p["name"] for p in team_squads.get(team1, {}).values()]
        squad_team2 = [p["name"] for p in team_squads.get(team2, {}).values()]
        today_players = squad_team1 + squad_team2
        today_batsman_form = batsman_form[batsman_form['batter'].isin(today_players)]

        if today_batsman_form.empty:
            st.warning(f"No batsman data found for {team1} or {team2} players.")
            st.write("Available batsmen:", list(batsman_form['batter'].unique()[:10]))
            logger.warning(f"No batsman data for {team1} vs {team2}")
        else:
            today_batsman_form['predicted_runs'] = player_model.predict(today_batsman_form[player_features].fillna(0))
            player_ranking = today_batsman_form[['batter', 'team', 'predicted_runs']].sort_values(
                by='predicted_runs', ascending=False
            )
            st.write("**Top 5 Batsmen by Predicted Runs**:")
            st.dataframe(player_ranking.head(5))

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='predicted_runs', y='batter', hue='team', data=player_ranking.head(10), ax=ax)
            ax.set_title(f"Top 10 Batsmen for {team1} vs {team2}")
            st.pyplot(fig)

if __name__ == "__main__":
    main()