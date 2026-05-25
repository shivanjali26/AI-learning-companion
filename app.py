import streamlit as st

from utils.pdf_reader import extract_text_from_pdf
from utils.text_cleaner import clean_text
from utils.chunker import chunk_text

from models.embedding_model import create_embeddings
from models.faiss_db import create_faiss_index

from utils.retriever import retrieve_relevant_chunks
from models.llm import generate_answer


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Lumina — AI Learning Companion",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-deep:       #07090f;
    --bg-card:       rgba(255,255,255,0.040);
    --bg-card-hover: rgba(255,255,255,0.068);
    --border:        rgba(255,255,255,0.080);
    --border-glow:   rgba(139,92,246,0.35);
    --text-primary:  #f0f2f8;
    --text-muted:    #8b93a8;
    --text-dim:      #4a5168;
    --accent-1:      #7c5cbf;
    --accent-2:      #4f87e0;
    --accent-grad:   linear-gradient(135deg,#7c5cbf,#4f87e0);
    --glow-purple:   0 0 40px rgba(124,92,191,0.18);
    --radius-lg:     20px;
    --radius-md:     14px;
    --radius-sm:     10px;
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary);
}

/* ── BACKGROUND ── */
.stApp {
    background-color: var(--bg-deep);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(124,92,191,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(79,135,224,0.10) 0%, transparent 55%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.018'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

/* ── STREAMLIT CHROME REMOVAL ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3rem 3rem 3rem;
    max-width: 1100px;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 99px; }

/* =========================================
   HERO
   ========================================= */
.hero-wrap {
    text-align: center;
    padding: 3rem 0 2.5rem;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(124,92,191,0.12);
    border: 1px solid rgba(124,92,191,0.28);
    border-radius: 99px;
    padding: 5px 16px;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #b79ef5;
    font-weight: 500;
    margin-bottom: 20px;
}

.hero-eyebrow-dot {
    width: 6px; height: 6px;
    background: #7c5cbf;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.5; transform:scale(1.4); }
}

.hero-title {
    font-family: 'Instrument Serif', serif;
    font-size: clamp(2.8rem, 5vw, 4.2rem);
    font-weight: 400;
    line-height: 1.12;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0 0 8px;
}

.hero-title em {
    font-style: italic;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 300;
    margin-top: 12px;
}

/* =========================================
   DIVIDER
   ========================================= */
.divider {
    height: 1px;
    background: linear-gradient(to right,transparent,var(--border),transparent);
    margin: 2rem 0;
}

/* =========================================
   UPLOAD CARD
   ========================================= */
.upload-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 30px;
    margin-bottom: 10px;
    transition: border-color .25s;
}

.upload-wrap:hover { border-color: rgba(255,255,255,0.12); }

.upload-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-dim);
    font-weight: 600;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.upload-label::before {
    content:'';display:inline-block;
    width:16px;height:1px;background:var(--text-dim);
}

/* =========================================
   READY BANNER
   ========================================= */
.ready-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(74,222,128,0.06);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: var(--radius-md);
    padding: 14px 20px;
    margin: 18px 0 28px;
}

.ready-dot {
    width: 8px; height: 8px;
    background: #4ade80;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse-dot 2.5s infinite;
}

.ready-text {
    font-size: 13.5px;
    color: #86efac;
    font-weight: 500;
}

.ready-text span {
    color: #4a5168;
    font-weight: 400;
    font-size: 12.5px;
    margin-left: 6px;
}

/* =========================================
   TOOL CARDS (the 3 clickable tiles)
   ========================================= */
.tools-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 10px;
}

.tool-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px 22px;
    cursor: pointer;
    transition: all .22s;
    position: relative;
    overflow: hidden;
}

.tool-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: var(--radius-lg);
    background: var(--accent-grad);
    opacity: 0;
    transition: opacity .22s;
}

.tool-card:hover {
    border-color: rgba(124,92,191,0.38);
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(124,92,191,0.14);
}

.tool-card:hover::before { opacity: 0.04; }

.tool-card.active-card {
    border-color: rgba(124,92,191,0.55);
    background: rgba(124,92,191,0.10);
    box-shadow: 0 0 0 1px rgba(124,92,191,0.22), 0 10px 30px rgba(124,92,191,0.15);
}

.tool-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    margin-bottom: 14px;
    position: relative; z-index: 1;
}

.tool-card.active-card .tool-icon {
    background: rgba(124,92,191,0.20);
    border-color: rgba(124,92,191,0.35);
}

.tool-name {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 5px;
    position: relative; z-index: 1;
}

.tool-desc {
    font-size: 12.5px;
    color: var(--text-muted);
    line-height: 1.55;
    font-weight: 300;
    position: relative; z-index: 1;
}

.tool-arrow {
    position: absolute;
    top: 18px; right: 18px;
    font-size: 16px;
    color: var(--text-dim);
    transition: all .2s;
    z-index: 1;
}

.tool-card:hover .tool-arrow,
.tool-card.active-card .tool-arrow {
    color: #b79ef5;
    transform: translateX(2px);
}

/* =========================================
   ACTIVE PANEL (expands below cards)
   ========================================= */
.panel-wrap {
    background: rgba(124,92,191,0.05);
    border: 1px solid rgba(124,92,191,0.20);
    border-radius: var(--radius-lg);
    padding: 28px 30px;
    margin-top: 14px;
    animation: panel-in .25s ease;
}

@keyframes panel-in {
    from { opacity:0; transform:translateY(-8px); }
    to   { opacity:1; transform:translateY(0); }
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}

.panel-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.35rem;
    font-weight: 400;
    color: var(--text-primary);
}

.panel-hint {
    font-size: 12.5px;
    color: var(--text-dim);
    margin-bottom: 16px;
    font-weight: 300;
}

/* =========================================
   ANSWER BOX
   ========================================= */
.answer-tag {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #b79ef5;
    font-weight: 600;
    margin: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.answer-tag::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(124,92,191,0.3), transparent);
}

.answer-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: var(--radius-md);
    padding: 22px 24px;
    color: var(--text-primary);
    font-size: 15px;
    line-height: 1.85;
    font-weight: 300;
}

/* =========================================
   FILE UPLOADER
   ========================================= */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.025) !important;
    border-radius: var(--radius-md) !important;
    border: 1.5px dashed rgba(255,255,255,0.10) !important;
    padding: 6px !important;
    transition: border-color .2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(124,92,191,0.38) !important;
}

[data-testid="stFileDropzone"] { background: transparent !important; }

/* =========================================
   QUESTION TEXT INPUT
   ========================================= */
.stTextInput > div > div > input {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14.5px !important;
    background: var(--bg-card);
    font-weight: 300 !important;
    padding: 14px 18px !important;
    height: 52px !important;
    transition: border-color .2s, box-shadow .2s !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(124,92,191,0.50) !important;
    box-shadow: 0 0 0 3px rgba(124,92,191,0.09) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    background: var(--bg-card);
    color: rgba(255,255,255,0.03);
    font-style: italic !important;
}

/* Ask › button — gradient accent */
.ask-action-btn > button {
    background: var(--accent-grad) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.03em !important;
}

.ask-action-btn > button:hover {
    box-shadow: 0 6px 20px rgba(124,92,191,0.35) !important;
    color: #fff !important;
    transform: translateY(-2px) !important;
}

/* =========================================
   BUTTONS
   ========================================= */
.stButton > button {
    width: 100%;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    height: 52px !important;
    transition: all .22s !important;
}

.stButton > button:hover {
    background: rgba(124,92,191,0.14) !important;
    border-color: rgba(124,92,191,0.40) !important;
    color: #c4aff7 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,92,191,0.18) !important;
}

/* ── Generate button ── */
.gen-btn > button {
    background: linear-gradient(135deg,#7c5cbf,#4f87e0) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

.gen-btn > button:hover {
    box-shadow: 0 8px 24px rgba(124,92,191,0.35) !important;
    color: white !important;
}

/* =========================================
   ALERT
   ========================================= */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid rgba(74,222,128,0.20) !important;
    background: rgba(74,222,128,0.06) !important;
    color: #86efac !important;
    font-size: 13px !important;
    display: none !important;  /* hide default success, we use custom banner */
}

/* =========================================
   SIDEBAR
   ========================================= */
section[data-testid="stSidebar"] {
    background: #070910 !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem !important;
}

.sidebar-logo {
    display: flex; align-items: center; gap: 10px; margin-bottom: 28px;
}

.sidebar-logo-mark {
    width: 36px; height: 36px;
    background: var(--accent-grad);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}

.sidebar-logo-text {
    font-family: 'Instrument Serif', serif;
    font-size: 1.25rem; font-weight: 400;
    color: var(--text-primary); line-height: 1.2;
}

.sidebar-logo-text small {
    display: block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10.5px; color: var(--text-dim);
    letter-spacing: 0.08em; text-transform: uppercase; font-weight: 500;
}

.sidebar-section { margin-bottom: 24px; }

.sidebar-section-label {
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.16em; color: var(--text-dim);
    font-weight: 600; margin-bottom: 12px;
}

.sidebar-nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: var(--radius-sm);
    font-size: 13.5px; color: var(--text-muted);
    font-weight: 400; margin-bottom: 3px; transition: all .18s;
}

.sidebar-nav-item:hover {
    background: rgba(255,255,255,0.04); color: var(--text-primary);
}

.nav-icon { width: 22px; text-align: center; font-size: 14px; opacity: .75; }

.sidebar-badge {
    margin-left: auto;
    background: rgba(124,92,191,0.18); color: #b79ef5;
    font-size: 10px; padding: 2px 8px; border-radius: 99px;
    font-weight: 500; letter-spacing: 0.06em;
}

.sidebar-info-card {
    background: rgba(124,92,191,0.08);
    border: 1px solid rgba(124,92,191,0.18);
    border-radius: var(--radius-md);
    padding: 14px 15px; font-size: 12.5px;
    color: var(--text-muted); line-height: 1.65;
}

.sidebar-info-card strong { color: #c4aff7; font-weight: 500; }

/* =========================================
   EMPTY STATE
   ========================================= */
.empty-state {
    text-align: center;
    padding: 3rem 0 2rem;
}

.empty-icon {
    width: 72px; height: 72px;
    background: rgba(124,92,191,0.10);
    border: 1px solid rgba(124,92,191,0.22);
    border-radius: 20px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 30px; margin-bottom: 18px;
}

.empty-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.4rem; color: #f0f2f8; margin-bottom: 8px;
}

.empty-sub {
    font-size: 13.5px; color: #8b93a8;
    max-width: 360px; margin: 0 auto; line-height: 1.65;
}

.features-row {
    display: flex; gap: 12px; flex-wrap: wrap;
    margin: 24px 0 0; justify-content: center;
}

.feature-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 18px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 99px; font-size: 13.5px;
    font-weight: 500; color: var(--text-primary);
}

/* =========================================
   FOOTER
   ========================================= */
.footer {
    text-align: center; padding: 2.5rem 0 1rem;
    border-top: 1px solid var(--border); margin-top: 3rem;
}

.footer-text {
    font-size: 12px; color: var(--text-dim); letter-spacing: 0.04em;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "active_tool" not in st.session_state:
    st.session_state.active_tool = None   # "ask" | "summary" | "quiz"

if "qa_answer" not in st.session_state:
    st.session_state.qa_answer = None

if "summary_result" not in st.session_state:
    st.session_state.summary_result = None

if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None

if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False

if "cleaned_text" not in st.session_state:
    st.session_state.cleaned_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-mark">✦</div>
        <div class="sidebar-logo-text">
            Lumina
            <small>AI Learning Companion</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-label">Capabilities</div>
        <div class="sidebar-nav-item"><span class="nav-icon">💬</span> Question Answering <span class="sidebar-badge">AI</span></div>
        <div class="sidebar-nav-item"><span class="nav-icon">📝</span> Smart Summaries</div>
        <div class="sidebar-nav-item"><span class="nav-icon">🧠</span> Quiz Generation</div>
        <div class="sidebar-nav-item"><span class="nav-icon">🔍</span> Semantic Search</div>
        <div class="sidebar-nav-item"><span class="nav-icon">📄</span> PDF Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-label">How It Works</div>
    </div>
    <div class="sidebar-info-card">
        <strong>① Upload</strong> your PDF notes or study material.<br><br>
        <strong>② Choose</strong> a tool — Ask, Summarise, or Quiz.<br><br>
        <strong>③ Get</strong> instant AI-powered results.<br><br>
        Powered by <strong>RAG + Generative AI</strong> for grounded, accurate answers.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11.5px;color:#4a5168;line-height:1.6;padding:0 2px;">
        Lumina processes documents locally. No data is stored beyond your session.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# HERO
# =====================================================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">
        <span class="hero-eyebrow-dot"></span>
        Powered by Generative AI
    </div>
    <div class="hero-title">Study smarter with <em>Lumina</em></div>
    <div class="hero-subtitle">
        Upload your material — then ask, summarise, and quiz with AI that understands your notes
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# =====================================================
# UPLOAD SECTION
# =====================================================

st.markdown("""
<div class="upload-label">① Upload Document</div>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop your PDF here or click to browse",
    type=["pdf"],
    label_visibility="visible"
)
st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# PROCESS PDF
# =====================================================

if uploaded_file and not st.session_state.doc_ready:
    with st.spinner("Analysing your document…"):
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.cleaned_text = clean_text(text)
        st.session_state.chunks       = chunk_text(st.session_state.cleaned_text)
        embeddings                    = create_embeddings(st.session_state.chunks)
        st.session_state.index        = create_faiss_index(embeddings)
        st.session_state.doc_ready    = True
        # Reset tool state on new upload
        st.session_state.active_tool  = None
        st.session_state.qa_answer    = None
        st.session_state.summary_result = None
        st.session_state.quiz_result  = None


if uploaded_file and st.session_state.doc_ready:

    # ── Ready banner ──
    fname = uploaded_file.name
    st.markdown(f"""
    <div class="ready-banner">
        <div class="ready-dot"></div>
        <div class="ready-text">
            Document ready
            <span>· {fname}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # =====================================================
    # TOOL SELECTION CARDS
    # =====================================================

    st.markdown('<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.14em;color:#4a5168;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px;"><span style=\'display:inline-block;width:16px;height:1px;background:#4a5168;\'></span>② Choose a Tool</div>', unsafe_allow_html=True)

    ask_active     = "active-card" if st.session_state.active_tool == "ask"     else ""
    summary_active = "active-card" if st.session_state.active_tool == "summary" else ""
    quiz_active    = "active-card" if st.session_state.active_tool == "quiz"    else ""

    st.markdown(f"""
    <div class="tools-grid">
        <div class="tool-card {ask_active}" id="card-ask">
            <span class="tool-arrow">›</span>
            <div class="tool-icon">💬</div>
            <div class="tool-name">Ask a Question</div>
            <div class="tool-desc">Type any question and get an instant AI answer grounded in your document.</div>
        </div>
        <div class="tool-card {summary_active}" id="card-summary">
            <span class="tool-arrow">›</span>
            <div class="tool-icon">📝</div>
            <div class="tool-name">Summarise</div>
            <div class="tool-desc">Generate a clean, structured summary of your study material.</div>
        </div>
        <div class="tool-card {quiz_active}" id="card-quiz">
            <span class="tool-arrow">›</span>
            <div class="tool-icon">🧠</div>
            <div class="tool-name">Generate Quiz</div>
            <div class="tool-desc">Create 10 MCQ practice questions with answers from your notes.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Invisible Streamlit buttons that drive the state toggle
    _col1, _col2, _col3 = st.columns(3)

    with _col1:
        if st.button("💬  Ask a Question", key="open_ask", use_container_width=True):
            st.session_state.active_tool = "ask" if st.session_state.active_tool != "ask" else None
            st.session_state.qa_answer = None
            st.rerun()

    with _col2:
        if st.button("📝  Summarise", key="open_summary", use_container_width=True):
            st.session_state.active_tool = "summary" if st.session_state.active_tool != "summary" else None
            st.rerun()

    with _col3:
        if st.button("🧠  Generate Quiz", key="open_quiz", use_container_width=True):
            st.session_state.active_tool = "quiz" if st.session_state.active_tool != "quiz" else None
            st.rerun()

    # =====================================================
    # ACTIVE PANEL — ASK
    # =====================================================

    if st.session_state.active_tool == "ask":
        st.markdown("""
        <div class="panel-wrap">
            <div class="panel-header">
                <div class="panel-title">💬 Ask Your Notes</div>
            </div>
            <div class="panel-hint">
                Type your question below and press <strong style="color:#b79ef5;">Ask ›</strong>
                — or press Enter after typing.
            </div>
        </div>
        """, unsafe_allow_html=True)

        q_col, btn_col = st.columns([5, 1], gap="small")
        with q_col:
            question_input = st.text_input(
                label="question",
                placeholder="e.g. What are the key topics in this document?",
                label_visibility="collapsed",
                key="question_input_box"
            )
        with btn_col:
            st.markdown('<div class="ask-action-btn">', unsafe_allow_html=True)
            ask_clicked = st.button("Ask ›", key="ask_submit", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if (ask_clicked or question_input.strip()) and question_input.strip():
            if ask_clicked or (question_input != st.session_state.get("last_question", "")):
                with st.spinner("Generating answer…"):
                    retrieved = retrieve_relevant_chunks(
                        question_input, st.session_state.index, st.session_state.chunks
                    )
                    st.session_state.qa_answer   = generate_answer(question_input, retrieved)
                    st.session_state.qa_question = question_input
                    st.session_state.last_question = question_input

        if st.session_state.qa_answer:
            if st.session_state.get("qa_question"):
                st.markdown(f"""
                <div style="margin:14px 0 8px;">
                    <span style="font-size:12px;color:#4a5168;font-style:italic;">You asked:</span>
                    <span style="font-size:13.5px;color:#c4aff7;margin-left:6px;">{st.session_state.qa_question}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:4px;">
                <div class="answer-tag">✦ &nbsp;AI Answer</div>
                <div class="answer-box">{st.session_state.qa_answer}</div>
            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # ACTIVE PANEL — SUMMARY
    # =====================================================

    elif st.session_state.active_tool == "summary":
        st.markdown("""
        <div class="panel-wrap">
            <div class="panel-header">
                <div class="panel-title">📝 Smart Summary</div>
            </div>
            <div class="panel-hint">AI will generate a clear, structured summary of your document.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.summary_result is None:
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("✦  Generate Summary Now", key="do_summary"):
                with st.spinner("Writing summary…"):
                    prompt = (
                        "Summarize the following study material clearly and concisely "
                        "using well-structured paragraphs:\n\n"
                        + st.session_state.cleaned_text[:4000]
                    )
                    st.session_state.summary_result = generate_answer(prompt, [])
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.summary_result:
            st.markdown(f"""
            <div style="margin-top:6px;">
                <div class="answer-tag">📚 &nbsp;Summary</div>
                <div class="answer-box">{st.session_state.summary_result}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("↺  Regenerate", key="regen_summary"):
                st.session_state.summary_result = None
                st.rerun()

    # =====================================================
    # ACTIVE PANEL — QUIZ
    # =====================================================

    elif st.session_state.active_tool == "quiz":
        st.markdown("""
        <div class="panel-wrap">
            <div class="panel-header">
                <div class="panel-title">🧠 Practice Quiz</div>
            </div>
            <div class="panel-hint">10 multiple-choice questions with 4 options and marked answers — tailored to your material.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.quiz_result is None:
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("✦  Generate Quiz Now", key="do_quiz"):
                with st.spinner("Building quiz…"):
                    prompt = (
                        "Generate exactly 10 multiple-choice questions with 4 options each "
                        "(A, B, C, D) and clearly mark the correct answer after each question. "
                        "Base everything on the following material:\n\n"
                        + st.session_state.cleaned_text[:4000]
                    )
                    st.session_state.quiz_result = generate_answer(prompt, [])
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.quiz_result:
            st.markdown(f"""
            <div style="margin-top:6px;">
                <div class="answer-tag">🧠 &nbsp;Quiz</div>
                <div class="answer-box">{st.session_state.quiz_result}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("↺  Regenerate", key="regen_quiz"):
                st.session_state.quiz_result = None
                st.rerun()


# =====================================================
# EMPTY STATE
# =====================================================

else:
    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📄</div>
            <div class="empty-title">No document uploaded yet</div>
            <div class="empty-sub">Upload a PDF above to unlock AI-powered question answering, summaries, and quiz generation.</div>
        </div>
        <div class="features-row">
            <div class="feature-pill">💬 Ask Questions</div>
            <div class="feature-pill">📝 Summarise</div>
            <div class="feature-pill">🧠 Quiz Generator</div>
            <div class="feature-pill">🔍 Semantic Search</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <div class="footer-text">
        Lumina AI Learning Companion &nbsp;·&nbsp; Powered by Generative AI &nbsp;·&nbsp; Built with ❤ using Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
