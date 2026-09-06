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

# --- Precise Custom CSS (Synpact Dark UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Overall Matte Background */
    .stApp {
        background-color: #0b0c10 !important;
        color: #f3f4f6 !important;
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #08090d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding-top: 1.2rem;
    }
    [data-testid="stSidebarNav"] { display: none; }

    /* Custom Synpact Primary Purple Button */
    .stButton>button {
        background: #5b50e6 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        font-size: 0.88rem !important;
    }
    .stButton>button:hover {
        opacity: 0.92;
    }

    /* Navigation Headers */
    .nav-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #55596b;
        margin: 1.2rem 0 0.5rem 0.5rem;
        font-weight: 700;
    }

    /* User Profile Box */
    .user-footer {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.65rem 0.75rem;
        background: #111218;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 1.5rem;
    }
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #ec4899;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* KPI Metric Cards */
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
        margin: 0.8rem 0 0.4rem 0;
    }
    .metric-bot {
        font-size: 0.78rem;
        color: #636779;
    }
    .metric-bot.positive { color: #10b981; }
    .metric-bot.negative { color: #ef4444; }

    /* Surface Cards */
    .surface-card {
        background: #111218;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1.4rem;
        min-height: 380px;
    }
    .surface-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    .surface-subtitle {
        font-size: 0.82rem;
        color: #636779;
        margin-top: 0.25rem;
    }

    /* Meeting Feed Card */
    .meeting-card-row {
        background: #111218;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
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

    /* Action Items Table Header & Rows */
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

    /* Ask Synpact Chat Bubble */
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
        cursor: pointer;
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

if not st.session_state.authenticated:
    _, c, _ = st.columns([1, 1.2, 1])
    with c:
        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <div style='width: 44px; height: 44px; border-radius: 50%; background: #6366f1; margin: 0 auto 12px; box-shadow: 0 0 20px #6366f1;'></div>
                <h2 style='margin:0; font-weight:700;'>Synpact<sup style='font-size: 0.6rem; color:#818cf8;'>AI</sup></h2>
                <p style='color: #636779; font-size: 0.9rem;'>Sign in to your meeting intelligence workspace</p>
            </div>
        """, unsafe_allow_html=True)
        with st.form("login_form"):
            u = st.text_input("Username", value="charan")
            p = st.text_input("Password", type="password", value="meeting2026")
            if st.form_submit_button("Continue", use_container_width=True):
                if USERS.get(u) == p:
                    st.session_state.authenticated = True
                    st.session_state.username = u.capitalize()
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    st.stop()

# --- Application State ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "report" not in st.session_state:
    st.session_state.report = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "meetings_history" not in st.session_state:
    st.session_state.meetings_history = [
        {"title": "Untitled Meeting", "date": "Sep 6, 2026", "duration": "3 min", "participants": 0, "score": 0},
        {"title": "Untitled Meeting", "date": "Sep 1, 2026", "duration": "0 min", "participants": 0, "score": 0},
        {"title": "Untitled Meeting", "date": "Sep 1, 2026", "duration": "3 min", "participants": 0, "score": 0},
    ]
if "default_action_items" not in st.session_state:
    st.session_state.default_action_items = [
        {"task": "Decide on playing another IPL season based on physical fitness", "meeting": "Untitled Meeting", "owner": "Unassigned", "priority": "Medium", "status": "Open", "due": "No Due Date"},
        {"task": "Decide on IPL future and retirement status over the next 6-7 months", "meeting": "Untitled Meeting", "owner": "Unassigned", "priority": "Medium", "status": "Open", "due": "No Due Date"},
        {"task": "Work hard for 9 months and assess body condition for IPL 2026", "meeting": "Untitled Meeting", "owner": "Unassigned", "priority": "Medium", "status": "Open", "due": "No Due Date"}
    ]

# --- Modal: Ingest New Meeting ---
@st.dialog("Record or Ingest Meeting Audio")
def new_meeting_dialog():
    uploaded = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])
    if uploaded:
        st.audio(uploaded)
        if st.button("Run Intelligence Engine", type="primary", use_container_width=True):
            with st.spinner("Transcribing and synthesizing with Gemini 3.6 Flash..."):
                t = transcribe_audio_file(uploaded)
                r = analyze_transcript(t)
                st.session_state.transcript = t
                st.session_state.report = r
                
                # Append into history
                st.session_state.meetings_history.insert(0, {
                    "title": r.meeting_title,
                    "date": datetime.today().strftime("%b %d, %Y"),
                    "duration": "3 min",
                    "participants": 1,
                    "score": 70
                })
                # Append action items
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
                st.rerun()

# --- Modal: Search Overlay (Screen 5) ---
@st.dialog("Search Workspace")
def search_dialog():
    query = st.text_input("🔍 Search", placeholder="Search meetings, transcripts, decisions...", label_visibility="collapsed")
    st.caption("Search powered by semantic matching • Press ESC to close")
    if query:
        st.write(f"Results for: **{query}**")
        matches = [m for m in st.session_state.meetings_history if query.lower() in m["title"].lower()]
        for m in matches:
            st.markdown(f"- **{m['title']}** ({m['date']})")

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 9px; margin-bottom: 1.5rem;'>
            <div style='width: 22px; height: 22px; border-radius: 50%; background: #6366f1; box-shadow: 0 0 10px #6366f1;'></div>
            <span style='font-size: 1.15rem; font-weight: 700; color: #ffffff;'>Synpact<sup style='font-size: 0.55rem; color:#818cf8; margin-left:2px;'>AI</sup></span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">NAVIGATION</div>', unsafe_allow_html=True)
    
    current_nav = st.radio(
        "Navigation",
        ["Dashboard", "All Meetings", "Action Items", "Analytics", "Search", "Ask Synpact"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 160px;'></div>", unsafe_allow_html=True)
    
    if st.button("Sign Out", use_container_width=True):
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

    # 2 Surface Columns
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
                    <div style="font-size: 0.85rem; color: #717684; margin-bottom: 0.6rem;">Summary generation pending...</div>
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
                <div style="color: #636779;">{item['owner'][:3]}.. ⌄</div>
                <div><span style="background:#171822; padding:3px 8px; border-radius:6px; color:#94a3b8; font-size:0.75rem;">{item['priority']} ⌄</span></div>
                <div><span style="background:#171822; padding:3px 8px; border-radius:6px; color:#94a3b8; font-size:0.75rem;">{item['status']} ⌄</span></div>
                <div style="color: #636779; font-size:0.78rem;">🕒 dd/mm/yyyy</div>
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

    # Charts Row
    chart_col1, chart_col2 = st.columns(2, gap="medium")
    
    # Meeting Volume Chart
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

    # Meeting Intelligence Chart
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
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
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

    # Initial Bot Introduction Bubble (matching Screen 6)
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

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
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

    # Suggestion Chips
    st.markdown("""
        <div style="margin-top: 1.5rem;">
            <span class="chat-prompt-pill">Summarize my recent meetings</span>
            <span class="chat-prompt-pill">What decisions were made this week?</span>
            <span class="chat-prompt-pill">Show my open action items</span>
            <span class="chat-prompt-pill">What did we discuss regarding IPL future?</span>
        </div>
    """, unsafe_allow_html=True)