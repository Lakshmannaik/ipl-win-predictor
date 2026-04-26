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

st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="centered")

# --- 2. VENUE TRANSLATOR (CLEAN DATA VERSION) ---
VENUE_MAPPING = {
    "m. a. chidambaram stadium": "MA Chidambaram Stadium",
    "m.a. chidambaram stadium": "MA Chidambaram Stadium",
    "m.a.chidambaram stadium": "MA Chidambaram Stadium",
    "m a chidambaram stadium": "MA Chidambaram Stadium",
    "ma chidambaram stadium": "MA Chidambaram Stadium",
    "chepauk": "MA Chidambaram Stadium",
    "wankhede stadium": "Wankhede Stadium",
    "wankhede stadium, mumbai": "Wankhede Stadium",
    "narendra modi stadium": "Narendra Modi Stadium",
    "eden gardens": "Eden Gardens",
    "eden gardens, kolkata": "Eden Gardens",
    "arun jaitley stadium": "Arun Jaitley Stadium",
    "feroz shah kotla": "Arun Jaitley Stadium", 
    "m. chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m.chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m chinnaswamy stadium": "M Chinnaswamy Stadium",
    "sawai mansingh stadium": "Sawai Mansingh Stadium",
    "sawai mansingh stadium, jaipur": "Sawai Mansingh Stadium",
    "rajiv gandhi international stadium": "Rajiv Gandhi International Stadium",
    "ekana cricket stadium": "Ekana Cricket Stadium",
    "maharaja yadavindra singh pca stadium": "Punjab Cricket Association IS Bindra Stadium"
}

# --- 3. SMART SCHEDULER (Toss-Aware) ---
def get_current_match_window():
    now = datetime.now(IST)
    weekday = now.weekday() 
    time_float = now.hour + now.minute / 60.0
    
    if weekday < 5:
        return "evening" if time_float >= 19.0 else "off_hours"
    else:
        if 15.0 <= time_float < 19.0: return "afternoon"
        elif time_float >= 19.0: return "evening"
    return "off_hours"

# --- 4. CACHED MATCH ID FETCHER (JSON Boolean Fix) ---
@st.cache_data(ttl=14400) 
def get_active_ipl_match_id(window_type):
    if window_type == "off_hours": return None
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}&offset=0"
    try:
        match_list = requests.get(url).json().get('data', [])
        for match in match_list:
            name = match.get('name', '').lower()
            match_ended = match.get('matchEnded', False)
            if ("ipl" in name or "indian premier league" in name) and not match_ended:
                return match.get('id')
    except: return None
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
    st.info("🌙 No live matches scheduled right now.")
    time.sleep(600) 
    st.rerun()
else:
    match_data = fetch_match_data(match_id)
    if match_data:
        teams = match_data.get('teams', ["Team 1", "Team 2"])
        status = match_data.get('status', '')
        
        # Determine Batting Order
        if teams[1].lower() in status.lower() and "bowl" in status.lower():
            batting_first, chasing_team = teams[0], teams[1]
        elif teams[0].lower() in status.lower() and "bowl" in status.lower():
            batting_first, chasing_team = teams[1], teams[0]
        else:
            batting_first, chasing_team = teams[0], teams[1]

        st.subheader(f"{teams[0]} vs {teams[1]}")
        st.write(f"**Live Status:** {status}")
        
        score_list = match_data.get('score', [])
        
        if len(score_list) == 0:
            st.warning(f"Match starting soon. {batting_first} preparing to bat.")
            
        elif len(score_list) == 1:
            s1 = score_list[0]
            st.info(f"1st Innings: {batting_first} is batting.")
            st.metric(label=f"{batting_first} Score", value=f"{s1.get('r')}/{s1.get('w')} ({s1.get('o')} overs)")
            
        elif len(score_list) >= 2:
            st.success(f"2nd Innings: {chasing_team} is chasing!")
            s1, s2 = score_list[0], score_list[1]
            
            # --- MATH ---
            target_score = s1.get('r') + 1
            current_score = s2.get('r')
            wickets_lost = s2.get('w')
            overs_bowled = s2.get('o')
            runs_required = target_score - current_score
            
            completed_overs = int(overs_bowled)
            extra_balls = int(round((overs_bowled - completed_overs) * 10))
            balls_bowled = (completed_overs * 6) + extra_balls
            balls_remaining = 120 - balls_bowled
            
            crr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0
            rrr = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else 0
            run_rate_diff = crr - rrr 
            if crr > 9:
                estimated_dots = int(balls_bowled * 0.30)
            elif crr > 7:
                estimated_dots = int(balls_bowled * 0.40)
            else:
                estimated_dots = int(balls_bowled * 0.50)
            # --- PREDICTION (Calculated before display) ---
            try:
                model = joblib.load(MODEL_PATH)
                api_venue = match_data.get('venue', '').lower()
                clean_venue = VENUE_MAPPING.get(api_venue, "MA Chidambaram Stadium")
                
                input_df = pd.DataFrame({
                    'venue': [clean_venue], 'season': ['2026'], 'is_impact_era': [1],
                    'target_score': [target_score], 'current_score': [current_score],
                    'runs_required': [runs_required], 'wickets_lost': [wickets_lost],
                    'balls_remaining': [balls_remaining], 'cumulative_dots': [estimated_dots],
                    'crr': [crr], 'rrr': [rrr], 'run_rate_diff': [run_rate_diff]
                })
                # Reorder and Cast
                input_df = input_df[['venue', 'season', 'is_impact_era', 'target_score', 'current_score', 
                                     'runs_required', 'wickets_lost', 'balls_remaining', 'cumulative_dots', 
                                     'crr', 'rrr', 'run_rate_diff']]
                input_df['venue'] = input_df['venue'].astype('category')
                input_df['season'] = input_df['season'].astype('category')
                
                win_prob = model.predict_proba(input_df)[0][1] * 100
            except:
                win_prob = 50.0

            # --- DISPLAY ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Target", target_score)
            col2.metric(f"{chasing_team} Score", f"{current_score}/{wickets_lost}")
            col3.metric("Required", f"{runs_required} off {balls_remaining}")

            st.markdown(f"### {chasing_team} Win Probability: **{win_prob:.1f}%**")
            st.progress(int(win_prob))

    # --- REFRESH ---
    st.divider()
    st.caption(f"🔋 Power Saver Active | Last updated: {datetime.now(IST).strftime('%I:%M:%S %p')} IST")
    time.sleep(150)
    st.rerun()