import streamlit as st
import requests
import time
import pytz
import pandas as pd
import joblib
from datetime import datetime

# --- 1. CONFIGURATION & SETUP ---
API_KEY = st.secrets["CRIC_API_KEY"]
IST = pytz.timezone('Asia/Kolkata')
MODEL_PATH = "ipl_xgboost_model.pkl"
ENCODER_PATH = "venue_encoder.pkl"

st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="centered")

# --- 2. VENUE TRANSLATOR ---
# --- 2. VENUE TRANSLATOR (UPDATED FOR CLEAN DATA) ---
VENUE_MAPPING = {
    # Chennai
    "m. a. chidambaram stadium": "MA Chidambaram Stadium",
    "m.a. chidambaram stadium": "MA Chidambaram Stadium",
    "m.a.chidambaram stadium": "MA Chidambaram Stadium",
    "m a chidambaram stadium": "MA Chidambaram Stadium",
    "ma chidambaram stadium": "MA Chidambaram Stadium",
    "chepauk": "MA Chidambaram Stadium",
    
    # Mumbai
    "wankhede stadium": "Wankhede Stadium",
    "wankhede stadium, mumbai": "Wankhede Stadium",
    
    # Ahmedabad
    "narendra modi stadium": "Narendra Modi Stadium",
    
    # Kolkata
    "eden gardens": "Eden Gardens",
    "eden gardens, kolkata": "Eden Gardens",
    
    # Delhi
    "arun jaitley stadium": "Arun Jaitley Stadium",
    "feroz shah kotla": "Arun Jaitley Stadium", 
    
    # Bengaluru
    "m. chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m.chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m chinnaswamy stadium": "M Chinnaswamy Stadium",
    
    # Jaipur
    "sawai mansingh stadium": "Sawai Mansingh Stadium",
    "sawai mansingh stadium, jaipur": "Sawai Mansingh Stadium",
    
    # Hyderabad
    "rajiv gandhi international cricket stadium": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium": "Rajiv Gandhi International Stadium",
    
    # Lucknow
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium": "Ekana Cricket Stadium",
    "ekana cricket stadium": "Ekana Cricket Stadium",
    
    # Punjab
    "maharaja yadavindra singh pca stadium": "Punjab Cricket Association IS Bindra Stadium",
    "punjab cricket association is bindra stadium": "Punjab Cricket Association IS Bindra Stadium",
    
    # Guwahati & Dharamshala
    "aca stadium, barsapara": "Barsapara Cricket Stadium, Guwahati",
    "himachal pradesh cricket association stadium": "Himachal Pradesh Cricket Association Stadium",
    "himachal pradesh cricket association stadium, dharamshala": "Himachal Pradesh Cricket Association Stadium",

    # Raipur
    "shaheed veer narayan singh international cricket stadium, new raipur": "Shaheed Veer Narayan Singh International Stadium",
    "shaheed veer narayan singh international stadium": "Shaheed Veer Narayan Singh International Stadium"
}

# --- 3. SMART SCHEDULER ---
def get_current_match_window():
    now = datetime.now(IST)
    weekday = now.weekday() # 0 = Mon, 6 = Sun
    time_float = now.hour + now.minute / 60.0
    
    if weekday < 5:
        return "evening" if time_float >= 19.5 else "off_hours"
    else:
        if 15.5 <= time_float < 19.5: 
            return "afternoon"
        elif time_float >= 19.5: 
            return "evening"
    return "off_hours"

# --- 4. CACHED MATCH ID FETCHER (4 Hours) ---
# --- 4. CACHED MATCH ID FETCHER (UPDATED FOR REAL JSON) ---
@st.cache_data(ttl=14400) 
def get_active_ipl_match_id(window_type):
    if window_type == "off_hours": 
        return None
        
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}&offset=0"
    try:
        match_list = requests.get(url).json().get('data', [])
        for match in match_list:
            name = match.get('name', '').lower()
            match_ended = match.get('matchEnded', False) # Uses the actual boolean from the API
            
            # If it's an IPL match and it has NOT ended
            if ("ipl" in name or "indian premier league" in name) and not match_ended:
                return match.get('id')
                
    except: 
        return None
        
    return None

# --- 5. DATA FETCHING LOGIC ---
def fetch_match_data(match_id):
    url = f"https://api.cricapi.com/v1/match_info?apikey={API_KEY}&id={match_id}"
    try: return requests.get(url).json().get('data', {})
    except: return {}

# --- 6. MAIN DASHBOARD UI ---
st.title("🏏 IPL Live Win Predictor")

window = get_current_match_window()
match_id = get_active_ipl_match_id(window)

if not match_id:
    st.info("🌙 No live matches scheduled right now. The tracker will auto-wake during match hours.")
    st.caption(f"Last checked: {datetime.now(IST).strftime('%I:%M %p')} IST")
    time.sleep(600) 
    st.rerun()
else:
    match_data = fetch_match_data(match_id)
    
    if match_data:
        match_name = match_data.get('name', 'Live Match')
        status = match_data.get('status', 'Status unavailable')
        score_list = match_data.get('score', [])
        
        st.subheader(match_name)
        st.write(f"**Live Status:** {status}")
        
        if len(score_list) == 0:
            st.warning("Match is about to start. Waiting for the first ball...")
            
        elif len(score_list) == 1:
            st.info("1st Innings in progress. Win probability will activate during the chase.")
            score_1 = score_list[0]
            st.metric(label="Batting First", value=f"{score_1.get('r')}/{score_1.get('w')} ({score_1.get('o')} overs)")
            
        elif len(score_list) >= 2:
            st.success("2nd Innings in progress! Live predictions active.")
            score_1, score_2 = score_list[0], score_list[1]
            
            # --- FEATURE EXTRACTION ---
            target_score = score_1.get('r') + 1
            current_score = score_2.get('r')
            wickets_lost = score_2.get('w')
            overs_bowled = score_2.get('o')
            runs_required = target_score - current_score
            
            completed_overs = int(overs_bowled)
            extra_balls = int(round((overs_bowled - completed_overs) * 10))
            balls_bowled = (completed_overs * 6) + extra_balls
            balls_remaining = 120 - balls_bowled
            
            crr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0
            rrr = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else 0
            run_rate_diff = crr - rrr  # Corrected math!
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target", target_score)
            col2.metric("Current Score", f"{current_score}/{wickets_lost}")
            col3.metric("Required", f"{runs_required} off {balls_remaining} balls")
            
            # --- MODEL PREDICTION ---
            # --- 6. MODEL PREDICTION ---
            try:
                # 1. Translate the live API venue to your training data string
                api_venue_raw = match_data.get('venue', '').lower()
                training_venue_string = VENUE_MAPPING.get(api_venue_raw, "MA Chidambaram Stadium, Chepauk")
                
                # Load the model
                model = joblib.load(MODEL_PATH)
                
                # 2. Build the dictionary (Notice venue and season are STRINGS)
                input_dict = {
                    'venue': [training_venue_string],
                    'season': ['2026'],  
                    'is_impact_era': [1],
                    'target_score': [target_score],
                    'current_score': [current_score],
                    'runs_required': [runs_required],
                    'wickets_lost': [wickets_lost],
                    'balls_remaining': [balls_remaining],
                    'cumulative_dots': [0],
                    'crr': [crr],
                    'rrr': [rrr],
                    'run_rate_diff': [run_rate_diff]
                }
                
                input_features = pd.DataFrame(input_dict)
                expected_order = [
                    'venue', 'season', 'is_impact_era', 'target_score', 'current_score', 
                    'runs_required', 'wickets_lost', 'balls_remaining', 'cumulative_dots', 
                    'crr', 'rrr', 'run_rate_diff'
                ]
                input_features = input_features[expected_order]
                
                # 3. CRITICAL STEP: Cast them to categories exactly like training phase
                input_features['venue'] = input_features['venue'].astype('category')
                input_features['season'] = input_features['season'].astype('category')
                
                # 4. Predict
                win_prob = model.predict_proba(input_features)[0][1] * 100
                st.markdown(f"### Chasing Team Win Probability: **{win_prob:.1f}%**")
                st.progress(int(win_prob))
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # --- 7. THE REFRESH LOOP ---
    st.divider()
    current_time = datetime.now(IST).strftime('%I:%M:%S %p')
    st.caption(f"🔋 Power Saver Active: Refreshing every 2.5 mins | Last updated: {current_time} IST")
    time.sleep(150)
    st.rerun()