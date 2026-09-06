import os
import urllib.parse
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from transcriber import transcribe_audio_file
from analyzer import analyze_transcript, ask_meeting_chat

st.set_page_config(
    page_title="Synpact AI",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Integrated UI CSS with Scoped Icon and Status Rules ---
st.markdown("""
<style>
    @import url('[https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap](https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap)');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #0b0c10 !important;
        color: #f3f4f6 !important;
    }

    /* Prevent icon fonts (Material Symbols) from breaking into raw text */
    [class*="material-symbols"],
    [data-testid*="stIcon"],
    [data-testid="stStatusWidget"] i,
    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] [data-testid="stExpanderToggleIcon"] {
        font-family: inherit !important;
        line-height: 1 !important;
    }

    /* Status widget layout fix */
    [data-testid="stStatusWidget"] {
        background-color: #14161d !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }
    [data-testid="stStatusWidget"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    /* Top Hero Navbar */
    .landing-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0 1.5rem 0;
        width: 100%;
    }
    .landing-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
    }
    .landing-brand-dot {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #6366f1;
        box-shadow: 0 0 16px #6366f1;
    }
    .landing-links {
        display: flex;
        gap: 2.2rem;
        color: #94a3b8;
        font-size: 0.92rem;
        font-weight: 500;
    }

    /* Hero Typography */
    .hero-badge-container {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 9999px;
        padding: 0.45rem 1.2rem;
        font-size: 0.82rem;
        color: #c7d2fe;
        font-weight: 600;
    }
    .hero-title {
        font-size: 4.8rem;
        font-weight: 800;
        line-height: 1.05;
        text-align: center;
        letter-spacing: -0.04em;
        color: #ffffff;
        margin: 0 auto 1.6rem auto;
        max-width: 950px;
    }
    .hero-title span.purple-glow {
        background: linear-gradient(135deg, #6366f1 20%, #a855f7 70%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #8b8ea2;
        font-size: 1.15rem;
        text-align: center;
        max-width: 680px;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
    }

    /* Centered Sign-In Box */
    .login-box {
        background: #14161d;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 28px;
        padding: 3rem 2.4rem 2.4rem 2.4rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        text-align: center;
        margin: 0 auto;
    }
    .login-icon {
        width: 64px;
        height: 64px;
        border-radius: 18px;
        background: #645bf6;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.4rem auto;
        box-shadow: 0 8px 24px rgba(100, 91, 246, 0.35);
    }
    .login-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #ffffff;
    }
    .login-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
    }
    .login-subtitle {
        font-size: 0.95rem;
        color: #7b8092;
        margin-bottom: 2rem;
    }

    /* Form Fields */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    [data-testid="stForm"] input {
        background-color: #0b0c10 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        padding: 0.9rem 1.1rem !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stForm"] input:focus {
        border-color: #645bf6 !important;
        box-shadow: 0 0 0 1px #645bf6 !important;
    }

    /* Primary Sign In Button */
    .stButton.signin-btn > button {
        background: #5b5ff5 !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.85rem 1rem !important;
        font-size: 1rem !important;
        width: 100% !important;
        margin-top: 0.6rem !important;
        box-shadow: 0 8px 20px rgba(91, 95, 245, 0.25) !important;
    }
    .stButton.signin-btn > button:hover {
        opacity: 0.92 !important;
    }

    .divider-text {
        color: #55596b;
        font-size: 0.85rem;
        margin: 1.3rem 0 1.1rem 0;
        text-align: center;
    }
    .login-footer-links {
        margin-top: 1.8rem;
        font-size: 0.88rem;
        color: #7b8092;
        text-align: center;
    }
    .login-footer-links a {
        color: #ffffff !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }

    /* Workspace Sidebar */
    [data-testid="stSidebar"] {
        background-color: #08090d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 1.5rem 1rem !important;
    }
    [data-testid="stSidebarNav"] { display: none; }

    .nav-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #55596b;
        margin: 1.2rem 0 0.5rem 0.4rem;
        font-weight: 700;
    }
    .user-footer {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.75rem;
        background: #111218;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 2rem;
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ec4899, #8b5cf6);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        flex-shrink: 0;
    }

    /* Metrics & Dashboard Surfaces */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #111218;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #8b8ea2;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.8rem 0 0.3rem 0;
    }
    .metric-bot { font-size: 0.78rem; color: #636779; }
    .metric-bot.positive { color: #10b981; }
    .metric-bot.negative { color: #ef4444; }

    .surface-card {
        background: #111218;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 380px;
    }
    .surface-title { font-size: 1.1rem; font-weight: 600; color: #ffffff; margin: 0; }
    .surface-subtitle { font-size: 0.82rem; color: #636779; margin-top: 0.25rem; }

    /* Clean Uploader Box */
    [data-testid="stFileUploader"] {
        width: 100% !important;
        margin-top: 0.4rem !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #111218 !important;
        border: 1px dashed rgba(255, 255, 255, 0.16) !important;
        border-radius: 12px !important;
        padding: 1.8rem 1.2rem !important;
        text-align: center !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #5b5ff5 !important;
    }
    [data-testid="stFileUploader"] button {
        background: #1e1f2b !important;
        color: #f3f4f6 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.82rem !important;
    }

    /* Live Transcript Display Box */
    .transcript-container {
        background: #0d0e14;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem;
        font-family: monospace !important;
        font-size: 0.88rem;
        color: #cbd5e1;
        max-height: 280px;
        overflow-y: auto;
        white-space: pre-wrap;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* Meeting Feed & Table */
    .meeting-card-row {
        background: #111218;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
    }
    .meeting-score-pill {
        background: #161720;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
        min-width: 68px;
    }
    .score-number { font-size: 1.35rem; font-weight: 700; color: #ffffff; line-height: 1; }
    .score-caption { font-size: 0.62rem; color: #636779; text-transform: uppercase; margin-top: 2px; }

    .table-header {
        display: grid;
        grid-template-columns: 2.2fr 1.6fr 1fr 1fr 1fr 1fr;
        padding: 0.8rem 1rem;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #55596b;
        font-weight: 700;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .table-row {
        display: grid;
        grid-template-columns: 2.2fr 1.6fr 1fr 1fr 1fr 1fr;
        padding: 1.1rem 1rem;
        align-items: center;
        font-size: 0.85rem;
        color: #cbd5e1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
</style>
""", unsafe_allow_html=True)

# --- Google OAuth Helper Configuration ---
GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", os.getenv("GOOGLE_CLIENT_ID", ""))
GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET", os.getenv("GOOGLE_CLIENT_SECRET", ""))
REDIRECT_URI = st.secrets.get("REDIRECT_URI", "http://localhost:8501")

def get_google_auth_url():
    """Generates the Google OAuth consent URL."""
    base_url = "[https://accounts.google.com/o/oauth2/v2/auth](https://accounts.google.com/o/oauth2/v2/auth)"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account"
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def exchange_code_for_user_info(auth_code):
    """Exchanges authorization code for Google user profile."""
    token_endpoint = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"
    data = {
        "code": auth_code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    try:
        r = requests.post(token_endpoint, data=data)
        tokens = r.json()
        access_token = tokens.get("access_token")
        if not access_token:
            return None
        user_info_endpoint = "[https://www.googleapis.com/oauth2/v3/userinfo](https://www.googleapis.com/oauth2/v3/userinfo)"
        resp = requests.get(user_info_endpoint, headers={"Authorization": f"Bearer {access_token}"})
        return resp.json()
    except Exception:
        return None

# Check OAuth Redirect Query Callback
query_params = st.query_params
if "code" in query_params and not st.session_state.get("authenticated", False):
    auth_code = query_params["code"]
    user_info = exchange_code_for_user_info(auth_code)
    if user_info and "email" in user_info:
        st.session_state.authenticated = True
        st.session_state.user_email = user_info["email"]
        st.session_state.user_name = user_info.get("name", user_info["email"].split("@")[0])
        st.session_state.user_picture = user_info.get("picture", None)
        st.query_params.clear()
        st.rerun()

# --- App State Init ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "show_login_form" not in st.session_state:
    st.session_state.show_login_form = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_picture" not in st.session_state:
    st.session_state.user_picture = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "report" not in st.session_state:
    st.session_state.report = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "meetings_history" not in st.session_state:
    st.session_state.meetings_history = [
        {"title": "Quarterly Strategy Review", "date": "Sep 6, 2026", "duration": "3 min", "participants": 2, "score": 85},
        {"title": "Product Roadmap Sync", "date": "Sep 1, 2026", "duration": "5 min", "participants": 4, "score": 75},
        {"title": "Design Alignment", "date": "Aug 28, 2026", "duration": "2 min", "participants": 3, "score": 60},
    ]
if "default_action_items" not in st.session_state:
    st.session_state.default_action_items = [
        {"task": "Decide on playing another IPL season based on physical fitness", "meeting": "Quarterly Strategy Review", "owner": "Charan", "priority": "Medium", "status": "Open", "due": "Sep 15, 2026"},
        {"task": "Decide on IPL future and retirement status over the next 6-7 months", "meeting": "Quarterly Strategy Review", "owner": "Charan", "priority": "Medium", "status": "Open", "due": "Oct 01, 2026"},
        {"task": "Work hard for 9 months and assess body condition for IPL 2026", "meeting": "Product Roadmap Sync", "owner": "Team", "priority": "High", "status": "In Progress", "due": "Nov 15, 2026"}
    ]

# =============================================================
# MODAL DIALOG: AUDIO INGESTION
# =============================================================
@st.dialog("Record or Ingest Meeting Audio", width="large")
def new_meeting_dialog():
    st.markdown("""
        <p style='color: #717684; font-size: 0.9rem; margin-top: -6px; margin-bottom: 1.2rem;'>
            Upload audio to transcribe speech directly into text and extract structured takeaways.
        </p>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload Audio File", 
        type=["mp3", "wav", "m4a"], 
        help="Supported formats: MP3, WAV, M4A (Max 200MB)"
    )
    
    if uploaded:
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        st.audio(uploaded)
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        
        if st.button("🚀 Transcribe & Generate Insights", type="primary", use_container_width=True):
            with st.status("Running Synpact Intelligence Pipeline...", expanded=True) as status:
                st.write("🎙️ Converting speech to text (Faster-Whisper)...")
                t = transcribe_audio_file(uploaded)
                st.session_state.transcript = t
                
                st.write("🧠 Synthesizing action items & decisions (Gemini Flash)...")
                try:
                    r = analyze_transcript(t)
                    st.session_state.report = r
                    
                    st.session_state.meetings_history.insert(0, {
                        "title": r.meeting_title,
                        "date": datetime.today().strftime("%b %d, %Y"),
                        "duration": "3 min",
                        "participants": 1,
                        "score": 88
                    })
                    for ai in r.action_items:
                        st.session_state.default_action_items.insert(0, {
                            "task": ai.task,
                            "meeting": r.meeting_title,
                            "owner": ai.owner,
                            "priority": "Medium",
                            "status": "Open",
                            "due": ai.due_date
                        })
                    st.session_state.messages = []
                    status.update(label="Complete! Insights Ready.", state="complete", expanded=False)
                    st.rerun()
                except Exception as err:
                    status.update(label="Processing Failed", state="error", expanded=True)
                    st.error(f"Intelligence synthesis error: {err}")

# =============================================================
# ROUTING: LANDING PAGE VS. LOGIN CARD VS. DASHBOARD
# =============================================================
if not st.session_state.authenticated:
    
    # -------------------------------------------------------------
    # VIEW A: THE HERO LANDING PAGE
    # -------------------------------------------------------------
    if not st.session_state.show_login_form:
        c_brand, c_navlinks, c_login, c_cta = st.columns([2.5, 3.5, 0.9, 1.2])
        with c_brand:
            st.markdown("""
                <div class="landing-brand" style="margin-top: 5px;">
                    <div class="landing-brand-dot"></div>
                    <span>Synpact<sup style='font-size: 0.6rem; color:#818cf8; margin-left: 2px;'>AI</sup></span>
                </div>
            """, unsafe_allow_html=True)
        with c_navlinks:
            st.markdown("""
                <div class="landing-links" style="margin-top: 10px; justify-content: center;">
                    <span>Features</span>
                    <span>Workflow</span>
                    <span>Pricing</span>
                </div>
            """, unsafe_allow_html=True)
        with c_login:
            if st.button("Log in", use_container_width=True, key="top_login"):
                st.session_state.show_login_form = True
                st.rerun()
        with c_cta:
            if st.button("Start free", type="primary", use_container_width=True, key="top_start"):
                st.session_state.show_login_form = True
                st.rerun()

        st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

        st.markdown("""
            <div class="hero-badge-container">
                <div class="hero-pill">
                    <span>🪄</span> AI Meeting Intelligence
                </div>
            </div>
            <div class="hero-title">
                Turn every<br>conversation into<br><span class="purple-glow">momentum.</span>
            </div>
            <div class="hero-subtitle">
                Synpact AI transforms meetings into searchable knowledge, clear decisions,
                and accountable action. Stop losing ideas in the transcript.
            </div>
        """, unsafe_allow_html=True)

        _, cta_left, cta_right, _ = st.columns([2.2, 1.4, 1.4, 2.2])
        with cta_left:
            if st.button("Start free →", type="primary", use_container_width=True, key="hero_start_btn"):
                st.session_state.show_login_form = True
                st.rerun()
        with cta_right:
            if st.button("▷ See how it works", use_container_width=True, key="hero_demo_btn"):
                st.session_state.show_login_form = True
                st.rerun()

        st.stop()

    # -------------------------------------------------------------
    # VIEW B: THE LOGIN CARD
    # -------------------------------------------------------------
    else:
        if st.button("← Back to Home"):
            st.session_state.show_login_form = False
            st.rerun()

        _, center_col, _ = st.columns([1, 1.25, 1])
        with center_col:
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            
            st.markdown("""
                <div class="login-box">
                    <div class="login-icon">
                        <div class="login-dot"></div>
                    </div>
                    <h1 class="login-title">Synpact AI</h1>
                    <div class="login-subtitle">Sign in to your workspace</div>
            """, unsafe_allow_html=True)

            with st.form("auth_form"):
                email_in = st.text_input("Email", placeholder="Email address", label_visibility="collapsed")
                pass_in = st.text_input("Password", placeholder="Password", type="password", label_visibility="collapsed")
                
                st.markdown('<div class="stButton signin-btn">', unsafe_allow_html=True)
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if submitted:
                    if email_in and pass_in:
                        st.session_state.authenticated = True
                        st.session_state.user_email = email_in
                        st.session_state.user_name = email_in.split("@")[0].capitalize()
                        st.session_state.show_login_form = False
                        st.rerun()
                    else:
                        st.error("Please enter both email and password.")

            st.markdown('<div class="divider-text">or</div>', unsafe_allow_html=True)

            # Live Google OAuth Link
            google_login_url = get_google_auth_url()
            st.markdown(f"""
                <a href="{google_login_url}" target="_self" style="text-decoration: none;">
                    <div style="background: #0f1015; color: #ffffff; border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.08); font-weight: 500; padding: 0.75rem 1rem; font-size: 0.92rem; width: 100%; text-align: center; display: flex; align-items: center; justify-content: center; gap: 10px; cursor: pointer;">
                        <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/></svg>
                        Continue with Google
                    </div>
                </a>
            """, unsafe_allow_html=True)

            st.markdown("""
                    <div class="login-footer-links">
                        <div style="margin-bottom: 12px;"><a href="#" style="color:#7b8092 !important; font-weight: 500 !important;">Forgot password?</a></div>
                        <div>Don't have an account? <a href="#" style="color:#ffffff !important; font-weight: 600 !important;">Sign up</a></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.stop()

# =============================================================
# WORKSPACE: MAIN DASHBOARD & SCREEN NAVIGATION
# =============================================================
with st.sidebar:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;'>
            <div style='width: 24px; height: 24px; border-radius: 7px; background: #645bf6; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 12px #645bf6;'>
                <div style='width: 7px; height: 7px; border-radius: 50%; background: #ffffff;'></div>
            </div>
            <span style='font-size: 1.2rem; font-weight: 700; color: #ffffff;'>Synpact<sup style='font-size: 0.55rem; color:#818cf8; margin-left:2px;'>AI</sup></span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">NAVIGATION</div>', unsafe_allow_html=True)
    current_nav = st.radio(
        "Navigation",
        ["Dashboard", "All Meetings", "Action Items", "Analytics", "Search", "Ask Synpact"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 140px;'></div>", unsafe_allow_html=True)
    
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.show_login_form = False
        st.session_state.report = None
        st.session_state.transcript = None
        st.rerun()

    disp_email = st.session_state.user_email if st.session_state.user_email else "charanvaygeti@gmail.com"
    disp_name = st.session_state.user_name if st.session_state.user_name else "Charan Vaygeti"
    initials = disp_name[:2].upper()

    avatar_html = (
        f'<img src="{st.session_state.user_picture}" style="width:36px; height:36px; border-radius:50%;">'
        if st.session_state.user_picture
        else f'<div class="user-avatar">{initials}</div>'
    )

    st.markdown(f"""
        <div class="user-footer">
            {avatar_html}
            <div style="overflow: hidden;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #ffffff;">{disp_name}</div>
                <div style="font-size: 0.72rem; color: #636779; text-overflow: ellipsis; overflow: hidden;">{disp_email}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 1: DASHBOARD
# -------------------------------------------------------------
if current_nav == "Dashboard":
    col_head, col_btn = st.columns([3.5, 1])
    with col_head:
        st.markdown(f"""
            <h1 style='font-size: 1.85rem; font-weight: 700; margin: 0; color: #ffffff;'>Good evening, {disp_name.split()[0]}.</h1>
            <p style='color: #717684; font-size: 0.95rem; margin-top: 0.25rem;'>Here's what happened across your meetings.</p>
        """, unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("＋  New Meeting", use_container_width=True):
            new_meeting_dialog()

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-top"><span>Meetings</span> <span>📅</span></div>
                <div class="metric-val">{len(st.session_state.meetings_history)}</div>
                <div class="metric-bot positive">↗ 2 from last week</div>
            </div>
            <div class="metric-card">
                <div class="metric-top"><span>Meeting Time</span> <span>⏱</span></div>
                <div class="metric-val">0h 12m</div>
                <div class="metric-bot negative">-45m from last week</div>
            </div>
            <div class="metric-card">
                <div class="metric-top"><span>Open Actions</span> <span>🕒</span></div>
                <div class="metric-val">{len(st.session_state.default_action_items)}</div>
                <div class="metric-bot">0 completed</div>
            </div>
            <div class="metric-card">
                <div class="metric-top"><span>Decisions</span> <span>📈</span></div>
                <div class="metric-val">3</div>
                <div class="metric-bot">Recorded this month</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.transcript:
        st.markdown("""
            <div class="surface-card" style="margin-bottom: 1.5rem; min-height: auto;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1rem;">
                    <div>
                        <h3 class="surface-title">🎙️ Audio Transcription Log</h3>
                        <div class="surface-subtitle">Converted speech-to-text ready for export and copy.</div>
                    </div>
                </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="transcript-container">{st.session_state.transcript}</div>', unsafe_allow_html=True)
        
        dl_col1, dl_col2, _ = st.columns([1.4, 1.4, 1.8])
        with dl_col1:
            st.download_button(
                label="📥 Download Transcript (.txt)",
                data=st.session_state.transcript,
                file_name="meeting_transcript.txt",
                mime="text/plain",
                use_container_width=True
            )
        with dl_col2:
            if st.session_state.report:
                brief_md = f"# {st.session_state.report.meeting_title}\n\n"
                brief_md += "## Executive Summary\n" + "\n".join([f"- {s}" for s in st.session_state.report.executive_summary]) + "\n\n"
                brief_md += "## Decisions\n" + "\n".join([f"- {d}" for d in st.session_state.report.key_decisions]) + "\n\n"
                brief_md += "## Actions\n" + "\n".join([f"- {a.task} ({a.owner})" for a in st.session_state.report.action_items])
                
                st.download_button(
                    label="📄 Download Briefing (.md)",
                    data=brief_md,
                    file_name="executive_brief.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.markdown("""
            <div class="surface-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                    <div>
                        <h3 class="surface-title">Recent Meetings</h3>
                        <div class="surface-subtitle">Your latest recorded conversations.</div>
                    </div>
                    <span style="color:#8b8ea2; font-size:0.82rem; cursor:pointer;">View all</span>
                </div>
        """, unsafe_allow_html=True)

        for m in st.session_state.meetings_history[:3]:
            st.markdown(f"""
                <div style="padding: 0.9rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <div style="font-size:0.95rem; font-weight:600; color:#ffffff;">{m['title']}</div>
                    <div style="font-size:0.78rem; color:#636779; margin-top:3px;">{m['date']} &nbsp;•&nbsp; {m['duration']}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
            <div class="surface-card">
                <h3 class="surface-title">Action Items</h3>
                <div class="surface-subtitle" style="margin-bottom:1rem;">Tasks needing attention.</div>
        """, unsafe_allow_html=True)

        for item in st.session_state.default_action_items[:3]:
            st.markdown(f"""
                <div style="display:flex; align-items:flex-start; gap:12px; padding: 0.9rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <div style="color:#55596b; font-size:1.1rem; line-height:1;">⭕</div>
                    <div>
                        <div style="font-size:0.92rem; font-weight:500; color:#ffffff;">{item['task']}</div>
                        <div style="font-size:0.75rem; color:#636779; margin-top:3px;">{item['priority']} Priority &nbsp;•&nbsp; {item['due']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 2: ALL MEETINGS
# -------------------------------------------------------------
elif current_nav == "All Meetings":
    col_t, col_s, col_b = st.columns([2.5, 2, 1.2])
    with col_t:
        st.markdown("""
            <h1 style='font-size: 1.85rem; font-weight: 700; margin: 0; color: #ffffff;'>All Meetings</h1>
            <p style='color: #717684; font-size: 0.95rem; margin-top: 0.25rem;'>Browse and search your meeting history.</p>
        """, unsafe_allow_html=True)
    with col_s:
        search_filter = st.text_input("Search meetings...", placeholder="Search meetings...", label_visibility="collapsed")
    with col_b:
        if st.button("＋  New Meeting", use_container_width=True):
            new_meeting_dialog()

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    for m in st.session_state.meetings_history:
        if search_filter and search_filter.lower() not in m["title"].lower():
            continue
        st.markdown(f"""
            <div class="meeting-card-row">
                <div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #ffffff; margin-bottom: 0.35rem;">{m['title']}</div>
                    <div style="font-size: 0.85rem; color: #717684; margin-bottom: 0.6rem;">Summary available • Fully indexed</div>
                    <div style="font-size: 0.78rem; color: #55596b;">{m['date']} &nbsp;•&nbsp; {m['duration']} &nbsp;•&nbsp; {m['participants']} participants</div>
                </div>
                <div class="meeting-score-pill">
                    <div class="score-number">{m['score']}</div>
                    <div class="score-caption">Score</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 3: ACTION ITEMS
# -------------------------------------------------------------
elif current_nav == "Action Items":
    c_title, c_f1, c_f2 = st.columns([3.5, 1.2, 1.2])
    with c_title:
        st.markdown("""
            <h1 style='font-size: 1.85rem; font-weight: 700; margin: 0; color: #ffffff;'>Action Items</h1>
            <p style='color: #717684; font-size: 0.95rem; margin-top: 0.25rem;'>Manage and track your assigned tasks.</p>
        """, unsafe_allow_html=True)
    with c_f1:
        st.selectbox("Filter", ["All Items", "Open", "Completed"], label_visibility="collapsed")
    with c_f2:
        st.selectbox("Sort", ["↑↓ Newest", "Due Date", "Priority"], label_visibility="collapsed")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="table-header">
            <div>Task</div>
            <div>Meeting</div>
            <div>Owner</div>
            <div>Priority</div>
            <div>Status</div>
            <div>Due Date</div>
        </div>
    """, unsafe_allow_html=True)

    for item in st.session_state.default_action_items:
        st.markdown(f"""
            <div class="table-row">
                <div style="color: #ffffff; font-weight:500;">{item['task']}</div>
                <div style="color: #636779;">🔗 {item['meeting']}</div>
                <div style="color: #636779;">{item['owner']}</div>
                <div><span style="background:#171822; padding:3px 8px; border-radius:6px; color:#94a3b8; font-size:0.75rem;">{item['priority']}</span></div>
                <div><span style="background:#171822; padding:3px 8px; border-radius:6px; color:#10b981; font-size:0.75rem;">{item['status']}</span></div>
                <div style="color: #636779; font-size:0.78rem;">🕒 {item['due']}</div>
            </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 4: ANALYTICS
# -------------------------------------------------------------
elif current_nav == "Analytics":
    c_head, c_pills = st.columns([3, 2])
    with c_head:
        st.markdown("""
            <h1 style='font-size: 1.85rem; font-weight: 700; margin: 0; color: #ffffff;'>Analytics Workspace 🪄</h1>
            <p style='color: #717684; font-size: 0.95rem; margin-top: 0.25rem;'>Analyze your meeting health, action items, and decision velocity.</p>
        """, unsafe_allow_html=True)
    with c_pills:
        st.segmented_control("Range", ["7 Days", "30 Days", "90 Days", "All Time"], default="30 Days", label_visibility="collapsed")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-top"><span>Total Meetings</span> <span>📹</span></div>
                <div class="metric-val">27</div>
                <div class="metric-bot">0.2 total hours</div>
            </div>
            <div class="metric-card">
                <div class="metric-top"><span>Avg Intelligence</span> <span>🪄</span></div>
                <div class="metric-val">70</div>
                <div class="metric-bot">Out of 100</div>
            </div>
            <div class="metric-card">
                <div class="metric-top"><span>Actions Completed</span> <span>✅</span></div>
                <div class="metric-val">0<span style="font-size:1.1rem; color:#636779;"> / 3</span></div>
                <div class="metric-bot">0% completion rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-top"><span>Decisions Made</span> <span>🎯</span></div>
                <div class="metric-val">3</div>
                <div class="metric-bot">0 unresolved questions</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2, gap="medium")
    with chart_col1:
        st.markdown("""
            <div class="surface-card">
                <h3 class="surface-title">Meeting Volume</h3>
                <div class="surface-subtitle" style="margin-bottom: 0.5rem;">Number of meetings held over time</div>
        """, unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=["Sep 1", "Sep 3", "Sep 5", "Sep 6"],
            y=[2, 0, 1, 3],
            marker_color="#5b50e6"
        ))
        fig1.update_layout(
            paper_bgcolor="#111218",
            plot_bgcolor="#111218",
            margin=dict(l=20, r=20, t=10, b=20),
            height=250,
            xaxis=dict(showgrid=False, color="#636779"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#636779")
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("""
            <div class="surface-card">
                <h3 class="surface-title">Meeting Intelligence</h3>
                <div class="surface-subtitle" style="margin-bottom: 0.5rem;">Average intelligence score trend</div>
        """, unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=["Sep 1", "Sep 3", "Sep 5", "Sep 6"],
            y=[50, 60, 65, 70],
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=3),
            marker=dict(size=6, color="#ffffff")
        ))
        fig2.update_layout(
            paper_bgcolor="#111218",
            plot_bgcolor="#111218",
            margin=dict(l=20, r=20, t=10, b=20),
            height=250,
            xaxis=dict(showgrid=False, color="#636779"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#636779", range=[0, 100])
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 5: SEARCH
# -------------------------------------------------------------
elif current_nav == "Search":
    st.markdown("""
        <h1 style='font-size: 1.85rem; font-weight: 700; margin: 0; color: #ffffff;'>Workspace Search</h1>
        <p style='color: #717684; font-size: 0.95rem; margin-top: 0.25rem;'>Semantic search across all meeting transcripts.</p>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    _, search_box, _ = st.columns([1, 2.5, 1])
    with search_box:
        q = st.text_input("Search", placeholder="Search meetings, transcripts, decisions...", label_visibility="collapsed")
        st.caption("Search powered by semantic matching • Press Enter to search")
        if q:
            st.markdown("#### Search Results")
            found = False
            for m in st.session_state.meetings_history:
                if q.lower() in m["title"].lower():
                    st.markdown(f"- **{m['title']}** — {m['date']} ({m['duration']})")
                    found = True
            if not found:
                st.info("No matching meetings or transcripts found.")

# -------------------------------------------------------------
# SCREEN 6: ASK SYNPACT (CHATBOT)
# -------------------------------------------------------------
elif current_nav == "Ask Synpact":
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 0.2rem;">
            <div style="width: 14px; height: 14px; border-radius: 50%; background: #6366f1;"></div>
            <h1 style='font-size: 1.85rem; font-weight: 700; margin: 0; color: #ffffff;'>Ask Synpact</h1>
        </div>
        <p style='color: #717684; font-size: 0.95rem;'>Your AI meeting intelligence assistant.</p>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_q := st.chat_input("Ask about your meetings..."):
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            if st.session_state.transcript:
                with st.spinner("Analyzing meeting intelligence..."):
                    reply = ask_meeting_chat(
                        transcript_text=st.session_state.transcript,
                        chat_history=st.session_state.messages[:-1],
                        user_question=user_q
                    )
            else:
                reply = "I don't have an active audio transcript loaded yet. Please ingest a recording using '+ New Meeting' on the Dashboard or All Meetings page."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    st.markdown("""
        <div style="margin-top: 1.5rem;">
            <span class="chat-prompt-pill">Summarize my recent meetings</span>
            <span class="chat-prompt-pill">What decisions were made this week?</span>
            <span class="chat-prompt-pill">Show my open action items</span>
            <span class="chat-prompt-pill">What did we discuss regarding IPL future?</span>
        </div>
    """, unsafe_allow_html=True)