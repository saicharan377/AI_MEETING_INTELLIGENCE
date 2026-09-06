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

# --- Global UI Styling & Isolated Component Tweaks ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background-color: #0b0c10 !important;
        color: #f3f4f6 !important;
    }

    /* Landing Page Navigation */
    .landing-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0 2rem 0;
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

    /* Landing Page Hero Badge & Headings */
    .hero-badge-container {
        display: flex;
        justify-content: center;
        margin-top: 1.5rem;
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

    /* Internal Workspace Sidebar */
    [data-testid="stSidebar"] {
        background-color: #08090d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 1.5rem 1rem !important;
    }
    [data-testid="stSidebarNav"] { display: none; }

    /* Action Buttons (excluding file uploader internal components) */
    .stMainBlockContainer .stButton>button {
        background: #5b50e6 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
        font-size: 0.88rem !important;
        box-shadow: 0 4px 14px rgba(91, 80, 230, 0.25);
        transition: all 0.2s ease;
    }
    .stMainBlockContainer .stButton>button:hover {
        opacity: 0.92;
        transform: translateY(-1px);
    }

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

    /* Cards & KPIs */
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
    .metric-bot {
        font-size: 0.78rem;
        color: #636779;
    }
    .metric-bot.positive { color: #10b981; }
    .metric-bot.negative { color: #ef4444; }

    .surface-card {
        background: #111218;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 380px;
    }
    .surface-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    .surface-subtitle {
        font-size: 0.82rem;
        color: #636779;
        margin-top: 0.25rem;
    }

    /* File Uploader Alignment & Card Design */
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
        border-color: #5b50e6 !important;
    }
    [data-testid="stFileUploader"] button {
        background: #1e1f2b !important;
        color: #f3f4f6 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.82rem !important;
        box-shadow: none !important;
        transform: none !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #272838 !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #8b8ea2 !important;
        font-size: 0.84rem !important;
    }

    /* Interactive Transcript Preview Card */
    .transcript-container {
        background: #0d0e14;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem;
        font-family: monospace !important;
        font-size: 0.85rem;
        color: #cbd5e1;
        max-height: 280px;
        overflow-y: auto;
        white-space: pre-wrap;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* Feed & Rows */
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

    .chat-bubble-assistant {
        background: #111218;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .chat-prompt-pill {
        background: #12131a;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 0.45rem 0.9rem;
        color: #7b7f94;
        font-size: 0.78rem;
        display: inline-block;
        margin-right: 8px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Authentication Logic ---
USERS = {
    "admin": "admin123",
    "charan": "meeting2026",
    "saicharan": "meeting2026"
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

# --- Session State Data ---
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

# --- Sign In Dialog Trigger ---
@st.dialog("Sign in to Synpact AI", width="small")
def login_dialog():
    st.markdown("<p style='color: #8b8ea2; font-size: 0.88rem; margin-top:-6px;'>Enter your credentials to enter your workspace.</p>", unsafe_allow_html=True)
    with st.form("landing_login_form"):
        u = st.text_input("Username", value="charan")
        p = st.text_input("Password", type="password", value="meeting2026")
        submit = st.form_submit_button("Launch Workspace", use_container_width=True)
        if submit:
            if USERS.get(u) == p:
                st.session_state.authenticated = True
                st.session_state.username = u.capitalize()
                st.rerun()
            else:
                st.error("Invalid username or password.")

# --- Modal Dialog: Audio Ingestion ---
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
                
                st.write("🧠 Synthesizing action items & decisions (Gemini 3.6 Flash)...")
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
# GATE: RENDER LANDING PAGE IF NOT LOGGED IN
# =============================================================
if not st.session_state.authenticated:
    # Top Navbar
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
            login_dialog()
    with c_cta:
        if st.button("Start free", type="primary", use_container_width=True, key="top_start"):
            login_dialog()

    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

    # Hero Badge & Headings
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

    # Centered Action Buttons
    _, cta_left, cta_right, _ = st.columns([2.2, 1.4, 1.4, 2.2])
    with cta_left:
        if st.button("Start free →", type="primary", use_container_width=True, key="hero_start_btn"):
            login_dialog()
    with cta_right:
        if st.button("▷ See how it works", use_container_width=True, key="hero_demo_btn"):
            login_dialog()

    st.stop()

# =============================================================
# WORKSPACE: APP SIDEBAR & SCREEN ROUTING
# =============================================================
with st.sidebar:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;'>
            <div style='width: 24px; height: 24px; border-radius: 50%; background: #6366f1; box-shadow: 0 0 12px #6366f1;'></div>
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
        st.session_state.report = None
        st.session_state.transcript = None
        st.rerun()

    st.markdown("""
        <div class="user-footer">
            <div class="user-avatar">CV</div>
            <div style="overflow: hidden;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #ffffff;">Charan Vaygeti</div>
                <div style="font-size: 0.72rem; color: #636779;">charanvaygeti@gmail.com</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 1: DASHBOARD
# -------------------------------------------------------------
if current_nav == "Dashboard":
    col_head, col_btn = st.columns([3.5, 1])
    with col_head:
        st.markdown("""
            <h1 style='font-size: 1.85rem; font-weight: 700; margin: 0; color: #ffffff;'>Good evening, Charan.</h1>
            <p style='color: #717684; font-size: 0.95rem; margin-top: 0.25rem;'>Here's what happened across your meetings.</p>
        """, unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("＋  New Meeting", use_container_width=True):
            new_meeting_dialog()

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 4 Metric Cards
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

    # --- Live Audio-to-Text Review Panel ---
    if st.session_state.transcript:
        st.markdown("""
            <div class="surface-card" style="margin-bottom: 1.5rem; min-height: auto;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1rem;">
                    <div>
                        <h3 class="surface-title">🎙️ Latest Audio Transcription</h3>
                        <div class="surface-subtitle">Extracted speech-to-text log ready for review and download.</div>
                    </div>
                </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="transcript-container">{st.session_state.transcript}</div>', unsafe_allow_html=True)
        
        dl_col1, dl_col2, _ = st.columns([1.3, 1.3, 2])
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

    # 2 Main Columns
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

    st.markdown("""
        <div class="chat-bubble-assistant">
            <div style="display: flex; gap: 12px; align-items: flex-start;">
                <span style="background:#5b50e6; padding: 4px 6px; border-radius: 6px; font-size: 0.75rem;">🤖</span>
                <div>
                    Hello! I'm Synpact AI. I can answer questions about any of your past meetings, summarize topics, or find specific decisions. What would you like to know?
                </div>
            </div>
        </div>
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