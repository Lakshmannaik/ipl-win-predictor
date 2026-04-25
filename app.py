import streamlit as st
import pandas as pd
import joblib
import time
import requests
import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(page_title="IPL Live Analytics", page_icon="🏏", layout="wide")
st.title("🏏 IPL Live Match Center")

# --- 2. Load the ML Model ---
@st.cache_resource
def load_model():
    return joblib.load('ipl_xgboost_model.pkl')

model = load_model()

# --- 3. API Configuration ---
API_KEY = st.secrets["CRIC_API_KEY"]

@st.cache_data(ttl=300)
def get_active_ipl_match_id():
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}&offset=0"
    try:
        response = requests.get(url)
        match_list = response.json().get('data', [])
        for match in match_list:
            if "ipl" in match.get('name', '').lower() or "indian premier league" in match.get('series', '').lower():
                if "match ended" not in match.get('status', '').lower():
                    return match.get('id')
        return None
    except:
        return None

def fetch_live_data(match_id):
    url = f"https://api.cricapi.com/v1/match_info?apikey={API_KEY}&id={match_id}"
    try:
        response = requests.get(url)
        return response.json().get('data', {})
    except:
        return {}

# --- 4. The UI Components ---

def display_scoreboard(match_data):
    """Displays a Google-style match summary card."""
    st.subheader(f"🏟️ {match_data.get('name', 'IPL Match')}")
    st.caption(f"📍 {match_data.get('venue', 'Loading venue...')}")
    
    score_list = match_data.get('score', [])
    teams = match_data.get('teams', ["Team A", "Team B"])
    
    # Create the 'Google Card' layout
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        
        # Team 1 (Home)
        with col1:
            st.write(f"### {teams[0]}")
            if len(score_list) > 0:
                s1 = score_list[0]
                st.title(f"{s1['r']}/{s1['w']}")
                st.write(f"({s1['o']} overs)")
        
        with col2:
            st.write("## VS")
            st.write(f"**{match_data.get('status', 'Live')}**")

        # Team 2 (Away)
        with col3:
            st.write(f"### {teams[1]}")
            if len(score_list) > 1:
                s2 = score_list[1]
                st.title(f"{s2['r']}/{s2['w']}")
                st.write(f"({s2['o']} overs)")
            else:
                st.title("Yet to Bat")

def translate_features(match_data):
    """Translates API JSON to XGBoost features for the 2nd Innings."""
    score_list = match_data.get('score', [])
    if len(score_list) < 2:
        return None 
        
    venue = match_data.get('venue', 'Unknown Venue')
    season = match_data.get('date', '2026').split('-')[0]
    is_impact_era = 1 if int(season) >= 2023 else 0

    first_innings = score_list[0]
    second_innings = score_list[1]
    
    target_score = first_innings.get('r', 0) + 1
    current_score = second_innings.get('r', 0)
    wickets_lost = second_innings.get('w', 0)
    overs_bowled = second_innings.get('o', 0.0)
    
    completed_overs = int(overs_bowled)
    balls_in_current_over = int(round((overs_bowled - completed_overs) * 10))
    balls_bowled = (completed_overs * 6) + balls_in_current_over
    
    balls_remaining = max(0, 120 - balls_bowled)
    runs_required = target_score - current_score
    
    crr = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0.0
    rrr = (runs_required * 6) / balls_remaining if balls_remaining > 0 else 0.0
    
    return {
        'venue': venue, 'season': season, 'is_impact_era': is_impact_era,
        'target_score': target_score, 'current_score': current_score,
        'runs_required': runs_required, 'wickets_lost': wickets_lost,
        'balls_remaining': balls_remaining, 'cumulative_dots': int(completed_overs * 3),
        'crr': crr, 'rrr': rrr, 'run_rate_diff': crr - rrr
    }

# --- 5. Main Application Flow ---

match_id = get_active_ipl_match_id()

if not match_id:
    st.info("🔎 Scanning for live IPL matches...")
    st.warning("No live matches found. The dashboard will update automatically when a game begins.")
else:
    # A. Fetch and Display Scoreboard (Always visible)
    live_data = fetch_live_data(match_id)
    display_scoreboard(live_data)
    
    # B. If 2nd Innings has started, show Predictions
    features = translate_features(live_data)
    
    if features:
        st.divider()
        st.markdown("### 🤖 ML Win Prediction")
        
        # Prediction Logic
        df = pd.DataFrame([features])
        df['venue'] = df['venue'].astype('category')
        df['season'] = df['season'].astype('category')
        win_prob = model.predict_proba(df)[0][1] * 100
        
        # Display Prediction Cards
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.metric(label="Chasing Team Win %", value=f"{win_prob:.1f}%")
        with p_col2:
            st.progress(int(win_prob) / 100.0, text=f"Win Probability: {int(win_prob)}%")
            
        # Advanced Metrics Row
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Required RR", f"{features['rrr']:.2f}")
        m_col2.metric("Current RR", f"{features['crr']:.2f}")
        m_col3.metric("Runs Needed", f"{features['runs_required']}")
    else:
        st.divider()
        st.info("🕒 **1st Innings in progress.** Win prediction will activate once the run chase begins.")

# --- 6. The Refresh Loop ---
st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")
time.sleep(15)
st.rerun()