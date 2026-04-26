import streamlit as st
import requests
import time
import pytz
import pandas as pd
import joblib
from datetime import datetime

# --- 1. CONFIGURATION & SETUP ---
# Securely fetch the API key from Streamlit Secrets
API_KEY = st.secrets["CRIC_API_KEY"]
IST = pytz.timezone('Asia/Kolkata')
MODEL_PATH = "ipl_xgboost_model.pkl"

# Configure the page layout
st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="centered")

# --- 2. SMART SCHEDULER ---
def get_current_match_window():
    """Determines if a match is likely live based on IST time and day of week."""
    now = datetime.now(IST)
    weekday = now.weekday() # 0 = Monday, 6 = Sunday
    time_float = now.hour + now.minute / 60.0
    
    # Weekdays (Mon-Fri): Only evening games at 7:30 PM (19.5)
    if weekday < 5:
        return "evening" if time_float >= 19.5 else "off_hours"
    # Weekends (Sat-Sun): Afternoon games at 3:30 PM (15.5) and Evening games at 7:30 PM
    else:
        if 15.5 <= time_float < 19.5: 
            return "afternoon"
        elif time_float >= 19.5: 
            return "evening"
            
    return "off_hours"

# --- 3. CACHED MATCH ID FETCHER ---
# This runs only ONCE every 4 hours, saving massive API hits.
@st.cache_data(ttl=14400) 
def get_active_ipl_match_id(window_type):
    if window_type == "off_hours":
        return None
    
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}&offset=0"
    try:
        response = requests.get(url)
        match_list = response.json().get('data', [])
        for match in match_list:
            name = match.get('name', '').lower()
            series = match.get('series', '').lower()
            if "ipl" in name or "indian premier league" in series:
                # Ensure the match hasn't already concluded
                if "match ended" not in match.get('status', '').lower():
                    return match.get('id')
        return None
    except:
        return None

# --- 4. DATA FETCHING LOGIC ---
def fetch_match_data(match_id):
    """Fetches the detailed live score for the active match ID."""
    url = f"https://api.cricapi.com/v1/match_info?apikey={API_KEY}&id={match_id}"
    try:
        response = requests.get(url)
        return response.json().get('data', {})
    except:
        return {}

# --- 5. MAIN DASHBOARD UI ---
st.title("🏏 IPL Live Win Predictor")

window = get_current_match_window()
match_id = get_active_ipl_match_id(window)

if not match_id:
    # What to show when no game is on
    st.info("🌙 No live matches scheduled right now. The tracker will auto-wake during match hours.")
    st.caption(f"Last checked: {datetime.now(IST).strftime('%I:%M %p')} IST")
    
    # Sleep for 10 minutes during off-hours to preserve credits
    time.sleep(600) 
    st.rerun()

else:
    # What to do when a match IS live
    match_data = fetch_match_data(match_id)
    
    if match_data:
        match_name = match_data.get('name', 'Live Match')
        status = match_data.get('status', 'Status unavailable')
        score_list = match_data.get('score', [])
        
        st.subheader(match_name)
        st.write(f"**Live Status:** {status}")
        
        # Logic to determine match phase based on innings
        if len(score_list) == 0:
            st.warning("Match is about to start. Waiting for the first ball...")
            
        elif len(score_list) == 1:
            st.info("1st Innings in progress. Win probability will activate during the chase.")
            score_1 = score_list[0]
            st.metric(label="Batting First", value=f"{score_1.get('r')}/{score_1.get('w')} ({score_1.get('o')} overs)")
            
        elif len(score_list) >= 2:
            st.success("2nd Innings in progress! Live predictions active.")
            
            score_1 = score_list[0]
            score_2 = score_list[1]
            
            # Feature Extraction
            target_score = score_1.get('r') + 1
            current_score = score_2.get('r')
            wickets_lost = score_2.get('w')
            overs_bowled = score_2.get('o')
            
            runs_required = target_score - current_score
            
            # Convert decimal overs (e.g., 14.3) to balls bowled
            completed_overs = int(overs_bowled)
            extra_balls = int(round((overs_bowled - completed_overs) * 10))
            balls_bowled = (completed_overs * 6) + extra_balls
            balls_remaining = 120 - balls_bowled
            
            crr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0
            rrr = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else 0
            run_rate_diff = crr - rrr 
            
            # Display Live Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Target", target_score)
            col2.metric("Current Score", f"{current_score}/{wickets_lost}")
            col3.metric("Required", f"{runs_required} off {balls_remaining} balls")
            
            # --- 6. MODEL PREDICTION ---
# --- 6. MODEL PREDICTION ---
        try:
            model = joblib.load(MODEL_PATH)
            
            # 1. Create the DataFrame with the exact variables
            input_dict = {
                'venue': [1],                # Placeholder: Update with actual encoded ID if possible
                'season': [2026],            # Current season
                'is_impact_era': [1],        # 1 = True (Impact player era)
                'target_score': [target_score],
                'current_score': [current_score],
                'runs_required': [runs_required],
                'wickets_lost': [wickets_lost],
                'balls_remaining': [balls_remaining],
                'cumulative_dots': [0],      # Placeholder: API rarely gives live dot balls easily
                'crr': [crr],
                'rrr': [rrr],
                'run_rate_diff': [run_rate_diff]
            }
            
            input_features = pd.DataFrame(input_dict)
            
            # 2. Force the exact column order your model expects
            expected_order = [
                'venue', 'season', 'is_impact_era', 'target_score', 'current_score', 
                'runs_required', 'wickets_lost', 'balls_remaining', 'cumulative_dots', 
                'crr', 'rrr', 'run_rate_diff'
            ]
            input_features = input_features[expected_order]
            
            # 3. Predict Probability
            win_prob = model.predict_proba(input_features)[0][1] * 100
            
            # Display Progress Bar and Percentage
            st.markdown(f"### Chasing Team Win Probability: **{win_prob:.1f}%**")
            st.progress(int(win_prob))
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # --- 7. THE REFRESH LOOP ---
    st.divider()
    current_time = datetime.now(IST).strftime('%I:%M:%S %p')
    st.caption(f"🔋 Power Saver Active: Refreshing every 2.5 mins | Last updated: {current_time} IST")
    
    # 150 seconds = 2.5 minutes
    time.sleep(150)
    st.rerun()