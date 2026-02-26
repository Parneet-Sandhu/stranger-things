
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stranger Things: Fan Intelligence",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS  – retro-horror dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Special+Elite&family=Share+Tech+Mono&display=swap');

/* ── palette ── */
:root {
    --red:    #CC1512;
    --amber:  #E87C2B;
    --yellow: #F5C518;
    --green:  #27AE60;
    --dark:   #090909;
    --panel:  #111118;
    --card:   #16161F;
    --border: #2A1A1A;
    --text:   #E8D5C0;
    --muted:  #7A6A5A;
}

/* ── global ── */
html, body, [class*="css"] {
    background-color: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Special Elite', serif;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 2px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem !important; }

/* ── headings ── */
h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.4rem !important;
    color: var(--red) !important;
    letter-spacing: 5px;
    text-shadow: 0 0 24px #CC151266, 0 2px 0 #000;
}
h2 {
    font-family: 'Bebas Neue', sans-serif !important;
    color: var(--amber) !important;
    letter-spacing: 3px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
}
h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    color: var(--yellow) !important;
    letter-spacing: 2px;
}

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 18px !important;
}
[data-testid="stMetricValue"] {
    color: var(--amber) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.4rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── widget labels ── */
label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}

/* ── buttons ── */
.stButton > button {
    background: var(--red) !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 2px;
    border: none;
    border-radius: 5px;
    padding: 8px 24px;
    transition: background 0.2s;
}
.stButton > button:hover { background: var(--amber) !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--yellow) !important;
    border-bottom: 2px solid var(--yellow) !important;
}

/* ── custom components ── */
.hero {
    background: linear-gradient(160deg, #1A0000 0%, #090909 60%);
    border: 1px solid var(--red);
    border-radius: 10px;
    padding: 36px 44px;
    text-align: center;
    margin-bottom: 28px;
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    color: var(--muted);
    font-size: 1rem;
    letter-spacing: 3px;
    margin-top: -10px;
}
.hero-desc {
    color: var(--text);
    font-size: 1.05rem;
    max-width: 700px;
    margin: 16px auto 0;
    line-height: 1.7;
}

.insight-box {
    background: linear-gradient(135deg, #180A0A, #14100A);
    border: 1px solid var(--amber);
    border-radius: 8px;
    padding: 16px 22px;
    margin: 14px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.87rem;
    color: var(--amber);
    line-height: 1.7;
}

.quote-card {
    background: var(--card);
    border-left: 4px solid var(--red);
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-family: 'Special Elite', serif;
    font-size: 0.95rem;
    line-height: 1.65;
}
.qc-pos { border-left-color: var(--green)  !important; }
.qc-neg { border-left-color: var(--red)    !important; }
.qc-neu { border-left-color: var(--yellow) !important; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #fff;
    margin-right: 6px;
}
.badge-pos { background: var(--green); }
.badge-neg { background: var(--red);   }
.badge-neu { background: #7A6A2A;      }

.keyword-pill {
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--amber);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 4px 3px;
}

.section-label {
    font-family: 'Share Tech Mono', monospace;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.empty-box {
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 32px;
    text-align: center;
    color: var(--muted);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    margin: 16px 0;
}

hr { border-color: var(--border) !important; margin: 20px 0 !important; }

.wc-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 20px;
    background: var(--card);
    border-radius: 8px;
    border: 1px solid var(--border);
    min-height: 120px;
    align-content: flex-start;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
SENT_COLORS = {
    "Positive": "#27AE60",
    "Neutral":  "#F5C518",
    "Negative": "#CC1512",
    "Unknown":  "#555566",
}

THEME_ICONS = {
    "Eleven Survival Theories": "⚡",
    "Ending Dissatisfaction":   "😤",
    "Upside Down Lore":         "🌀",
    "Writing Criticism":        "✍️",
    "Nostalgia & Emotion":      "🕯️",
}

THEME_STORIES = {
    "Eleven Survival Theories": (
        "The internet refused to accept Eleven's fate. Fan threads erupted with frame-by-frame "
        "analyses, power theory breakdowns, and hopeful predictions. Hope dies hard in Hawkins."
    ),
    "Ending Dissatisfaction": (
        "A vocal and passionate faction felt the finale dropped the ball. "
        "These fans catalogued plot holes, unanswered questions, and rushed resolutions "
        "with forensic dedication."
    ),
    "Upside Down Lore": (
        "The lore crowd never sleeps. The true origin of the Upside Down, the Mind Flayer's "
        "consciousness, and Vecna's endgame generated some of the deepest theory threads online."
    ),
    "Writing Criticism": (
        "Showrunner hot takes, pacing debates, and character arc breakdowns. These fans "
        "brought their inner film-school critic to every episode — and weren't gentle."
    ),
    "Nostalgia & Emotion": (
        "Tears, childhood memories, and heartfelt goodbyes. This cluster is the emotional "
        "core of the fandom: people who grew up with Stranger Things and felt every moment."
    ),
}

THEME_EMOJIS_FALLBACK = "🔮"

# Stop-words for keyword extraction
_SW = frozenset({
    "the","and","to","of","is","in","a","i","it","for","this","that","was","on","are","with",
    "have","be","as","at","but","they","he","she","we","you","do","not","so","or","from",
    "by","an","had","his","her","their","our","has","been","all","would","could","will",
    "what","which","there","about","more","when","than","just","like","if","also","into",
    "up","out","can","my","one","some","its","after","him","them","no","very","other",
    "stranger","things","season","show","series","netflix","self","strangerthings","redd",
    "reddit","discussion","spoiler","spoilers","http","https","www","discussion","title",
    "discussioni","discussionthe","discussionstranger","discussionwhat","discussionwhy",
    "discussionhow","discussiondoes","discussionis","isntshe","imgredd","imgur",
    "old","new","even","back","much","didn","don","doesn","isn","wasn","couldn","wouldn",
    "still","over","here","your","then","than","through","before","right","first","last",
    "ever","being","really","think","know","make","watch","feel","good","great","love",
    "episode","episodes","character","characters","scene","scenes","final","finale",
})

# ─────────────────────────────────────────────────────────────
#  DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str = "stranger_things_final.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── text ──
    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].astype(str).str.strip()

    # ── theme ──
    if "theme" not in df.columns:
        df["theme"] = "Unknown"
    df["theme"] = df["theme"].astype(str).str.strip()

    # ── sentiment: fill NaN as "Unknown" ──
    if "sentiment" in df.columns:
        df["sentiment"] = (
            df["sentiment"]
            .astype(str)
            .str.strip()
            .str.title()
        )
        valid = {"Positive", "Neutral", "Negative"}
        df["sentiment"] = df["sentiment"].where(df["sentiment"].isin(valid), "Unknown")
    else:
        df["sentiment"] = "Unknown"

    # ── sentiment_score ──
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = float("nan")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    # ── source ──
    if "source" not in df.columns:
        df["source"] = "Unknown"
    df["source"] = df["source"].astype(str).str.strip()

    return df


# ─────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def pdark(legend=True):
    """Return a dict of Plotly layout overrides for dark theme."""
    d = dict(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font_color    = "#E8D5C0",
        font_family   = "Special Elite",
    )
    if legend:
        d["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#E8D5C0"))
    return d


def pchart(fig, h=320):
    """Render a Plotly figure full-width with consistent height."""
    fig.update_layout(height=h)
    st.plotly_chart(fig, width="stretch")


def safe_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample up to n rows without crashing on small / empty DataFrames."""
    if df is None or df.empty:
        return df
    return df.sample(min(n, len(df)), random_state=seed)


def badge_html(sentiment: str) -> str:
    cls = {"Positive": "badge-pos", "Negative": "badge-neg",
           "Neutral": "badge-neu", "Unknown": "badge-neu"}.get(sentiment, "badge-neu")
    return f'<span class="badge {cls}">{sentiment}</span>'


def quote_card_html(text: str, sentiment: str, source: str = "") -> str:
    cls = {"Positive": "qc-pos", "Negative": "qc-neg",
           "Neutral": "qc-neu", "Unknown": "qc-neu"}.get(sentiment, "")
    src = (f'<span style="color:var(--muted);font-size:0.72rem;float:right;'
           f'font-family:\'Share Tech Mono\',monospace;">{source}</span>') if source else ""
    return (f'<div class="quote-card {cls}">{src}'
            f'{badge_html(sentiment)}'
            f'<span style="color:var(--text);margin-left:4px;">{text}</span></div>')


def empty_box(msg: str = "No data for current selection."):
    st.markdown(f'<div class="empty-box">🔦 {msg}</div>', unsafe_allow_html=True)


def section_label(txt: str):
    st.markdown(f'<p class="section-label">{txt}</p>', unsafe_allow_html=True)


def wordcloud_html(texts: list, max_words: int = 70) -> str:
    """Pure HTML/CSS word cloud — no external library needed."""
    words = []
    for t in texts:
        words += re.findall(r"\b[a-zA-Z]{4,}\b", t.lower())
    words = [w for w in words if w not in _SW]
    freq  = Counter(words).most_common(max_words)
    if not freq:
        return '<div class="wc-wrap"><span style="color:var(--muted)">Not enough text data.</span></div>'
    max_f  = freq[0][1]
    colours = ["#CC1512", "#E87C2B", "#F5C518", "#E8D5C0", "#9A7A5A", "#7A6A8A"]
    html   = '<div class="wc-wrap">'
    for word, count in freq:
        size = 11 + int((count / max_f) * 30)
        col  = colours[hash(word) % len(colours)]
        html += (f'<span style="font-size:{size}px;color:{col};'
                 f'font-family:\'Bebas Neue\',sans-serif;letter-spacing:1px;'
                 f'opacity:{0.55 + 0.45*(count/max_f):.2f};">{word}</span>')
    return html + "</div>"


def top_tfidf_keywords(df: pd.DataFrame, theme: str, n: int = 12) -> list:
    """Extract top n TF-IDF keywords for a given theme."""
    texts = df[df["theme"] == theme]["text"].dropna().tolist()
    if len(texts) < 3:
        return []
    try:
        vec  = TfidfVectorizer(stop_words="english", max_features=500,
                               ngram_range=(1, 2), min_df=2)
        mat  = vec.fit_transform(texts)
        scores = mat.sum(axis=0).A1
        vocab  = vec.get_feature_names_out()
        top    = sorted(zip(scores, vocab), reverse=True)
        # Filter URL fragments, reddit noise, and generic filler terms
        _noise = ("strangerthing","redd","self","http","imgur","discuss","reddit","old",
                  "com","www","things","stranger","season","episode","netflix",
                  "spoiler","title","thread","post","imgredd","img")
        clean  = [w for _, w in top
                  if not any(s in w for s in _noise)
                  and len(w) > 3
                  and not w.replace(" ","").isdigit()]
        return clean[:n]
    except Exception:
        return []


def freq_keywords(texts: list, n: int = 12) -> list:
    """Simple frequency-based keyword fallback."""
    words = []
    for t in texts:
        words += re.findall(r"\b[a-zA-Z]{4,}\b", t.lower())
    words = [w for w in words if w not in _SW]
    return [w for w, _ in Counter(words).most_common(n * 2)][:n]


# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
try:
    df = load_data("stranger_things_final.csv")
except FileNotFoundError:
    st.error("⚠️  **`stranger_things_final.csv` not found.**  "
             "Place the CSV in the same folder as `app.py` and restart.")
    st.stop()
except Exception as exc:
    st.error(f"⚠️  Error loading data: {exc}")
    st.stop()

# Derived constants
THEMES     = sorted(df["theme"].dropna().unique().tolist())
SENTIMENTS = ["Positive", "Neutral", "Negative", "Unknown"]
SOURCES    = sorted(df["source"].dropna().unique().tolist())

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔦 STRANGER THINGS")
    st.markdown(
        "<p style='font-family:Share Tech Mono;font-size:0.72rem;color:#7A6A5A;'>"
        "FAN THEORY INTELLIGENCE SYSTEM</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "NAVIGATE",
        options=[
            "🏠 Home & Overview",
            "🗂️ Theme Breakdown",
            "💬 Sentiment Analysis",
            "🔍 Theory Explorer",
            "✨ Fun Insights",
            "📋 Conclusion",
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-family:Share Tech Mono;font-size:0.72rem;color:#7A6A5A;'>"
        "GLOBAL FILTERS</p>",
        unsafe_allow_html=True,
    )

    # Sentiment filter
    sent_filter = st.multiselect(
        "Filter by Sentiment",
        options=SENTIMENTS,
        default=SENTIMENTS,
        label_visibility="visible",
    )
    # Source filter
    src_filter = st.multiselect(
        "Filter by Source",
        options=SOURCES,
        default=SOURCES,
        label_visibility="visible",
    )

    # Safeguard against empty selections
    if not sent_filter: sent_filter = SENTIMENTS
    if not src_filter:  src_filter  = SOURCES

    st.markdown("---")
    st.markdown(
        f"<p style='font-family:Share Tech Mono;font-size:0.7rem;color:#7A6A5A;'>"
        f"📊 {len(df):,} total discussions<br>"
        f"🧠 {len(THEMES)} theory clusters<br>"
        f"🌐 {len(SOURCES)} data source(s)</p>",
        unsafe_allow_html=True,
    )

# Apply global filters
fdf = df[
    df["sentiment"].isin(sent_filter) &
    df["source"].isin(src_filter)
].copy()


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home & Overview":

    # Hero banner
    st.markdown("""
    <div class="hero">
        <h1>STRANGER THINGS</h1>
        <p class="hero-sub">FAN THEORIES · SENTIMENT · NARRATIVE LOOPHOLES</p>
        <p class="hero-desc">
            719 fan discussions. 5 theory clusters. One unanswered question: <em>did the finale
            stick the landing?</em> This intelligence report breaks down what the internet really
            thought — comment by comment, theme by theme.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if fdf.empty:
        empty_box("No data matches the current sidebar filters.")
        st.stop()

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💬 Discussions",    f"{len(fdf):,}")
    k2.metric("🧠 Theme Clusters", f"{fdf['theme'].nunique()}")
    k3.metric("✅ Positive",        f"{(fdf['sentiment']=='Positive').sum():,}")
    k4.metric("😤 Negative",        f"{(fdf['sentiment']=='Negative').sum():,}")

    st.markdown("---")
    col_l, col_r = st.columns([1.3, 1])

    # ── Theme bar chart ──
    with col_l:
        st.markdown("## 🗂 Discussions per Theme")
        tc = fdf["theme"].value_counts().reset_index()
        tc.columns = ["Theme", "Count"]
        fig = px.bar(
            tc, x="Count", y="Theme", orientation="h",
            color="Count",
            color_continuous_scale=["#2A1A1A", "#CC1512", "#F5C518"],
            text="Count",
        )
        fig.update_traces(textfont_color="#fff", textposition="outside")
        fig.update_layout(
            **pdark(), coloraxis_showscale=False,
            xaxis_title="", yaxis_title="",
            margin=dict(l=0, r=30, t=10, b=0),
        )
        pchart(fig, max(300, len(tc) * 58))

    # ── Overall sentiment donut ──
    with col_r:
        st.markdown("## 😶 Overall Sentiment")
        sc = fdf["sentiment"].value_counts().reset_index()
        sc.columns = ["Sentiment", "Count"]
        fig2 = px.pie(
            sc, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=SENT_COLORS,
            hole=0.48,
        )
        fig2.update_traces(textfont_color="#fff", textfont_size=13)
        fig2.update_layout(**pdark(), margin=dict(l=0, r=0, t=10, b=0))
        pchart(fig2, 360)

    st.markdown("---")

    # ── Source chart ──
    st.markdown("## 🌐 Data Sources")
    src_c = fdf["source"].value_counts().reset_index()
    src_c.columns = ["Source", "Count"]
    fig3 = px.bar(
        src_c, x="Source", y="Count",
        color="Count",
        color_continuous_scale=["#2A1A1A", "#E87C2B", "#F5C518"],
        text="Count",
    )
    fig3.update_traces(textfont_color="#fff", textposition="outside")
    fig3.update_layout(
        **pdark(), coloraxis_showscale=False,
        xaxis_title="", yaxis_title="",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    pchart(fig3, 260)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — THEME BREAKDOWN
# ══════════════════════════════════════════════════════════════
elif page == "🗂️ Theme Breakdown":

    st.markdown("## 🗂️ Theme Breakdown")
    st.markdown(
        "<p style='color:var(--muted);font-family:Share Tech Mono;font-size:0.85rem;'>"
        "Which theories dominated fan discourse — and how positive was each one?</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if fdf.empty:
        empty_box()
        st.stop()

    col_bar, col_pie = st.columns(2)

    with col_bar:
        st.markdown("### 📊 Volume per Theme")
        tc = fdf["theme"].value_counts().reset_index()
        tc.columns = ["Theme", "Count"]
        fig = px.bar(
            tc, x="Theme", y="Count",
            color="Count",
            color_continuous_scale=["#2A1A1A", "#CC1512", "#F5C518"],
            text="Count",
        )
        fig.update_traces(textfont_color="#fff", textposition="outside")
        fig.update_layout(
            **pdark(), coloraxis_showscale=False,
            xaxis_title="", yaxis_title="Discussions",
            xaxis_tickangle=-20,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        pchart(fig, 360)

    with col_pie:
        st.markdown("### 🥧 Theme Share")
        fig2 = px.pie(
            tc, names="Theme", values="Count",
            color_discrete_sequence=["#CC1512","#E87C2B","#F5C518","#27AE60","#7A6A8A"],
            hole=0.38,
        )
        fig2.update_traces(
            textfont_color="#fff", textfont_size=12,
            textposition="outside",
        )
        fig2.update_layout(**pdark(), margin=dict(l=0, r=0, t=10, b=0))
        pchart(fig2, 360)

    st.markdown("---")
    st.markdown("## 🔎 Explore a Theme")

    sel_theme = st.selectbox(
        "Select a Theory Cluster",
        options=THEMES,
        format_func=lambda t: f"{THEME_ICONS.get(t, THEME_EMOJIS_FALLBACK)}  {t}",
    )
    tdf = fdf[fdf["theme"] == sel_theme]

    if tdf.empty:
        empty_box("No discussions in this theme for the current filters.")
        st.stop()

    # Theme story
    story = THEME_STORIES.get(sel_theme, f"Explore fan discussions about {sel_theme}.")
    st.markdown(
        f'<div class="insight-box">💡 <strong>ABOUT THIS THEME</strong><br><br>{story}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{THEME_ICONS.get(sel_theme,'🔮')} Total",  f"{len(tdf):,}")
    c2.metric("✅ Positive",  f"{(tdf['sentiment']=='Positive').sum():,}")
    c3.metric("😤 Negative",  f"{(tdf['sentiment']=='Negative').sum():,}")

    tab_sent, tab_comments = st.tabs(["📊 Sentiment Distribution", "💬 Sample Comments"])

    with tab_sent:
        ts = tdf["sentiment"].value_counts().reset_index()
        ts.columns = ["Sentiment", "Count"]
        fig3 = px.pie(
            ts, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=SENT_COLORS,
            hole=0.45,
        )
        fig3.update_traces(textfont_color="#fff", textfont_size=14)
        fig3.update_layout(**pdark(), margin=dict(l=0, r=0, t=10, b=0))
        pchart(fig3, 320)

    with tab_comments:
        n_show   = st.slider("Comments to show", 3, 20, 7, key="tb_n")
        sent_sel = st.radio(
            "Filter sentiment", ["All"] + [s for s in SENTIMENTS if s != "Unknown"],
            horizontal=True, key="tb_s"
        )
        q_df   = tdf if sent_sel == "All" else tdf[tdf["sentiment"] == sent_sel]
        q_long = q_df[q_df["text"].str.len() > 40]
        pool   = q_long if not q_long.empty else q_df
        sample = safe_sample(pool, n_show)

        if sample is None or sample.empty:
            empty_box("No comments found for this filter.")
        else:
            for _, row in sample.iterrows():
                st.markdown(
                    quote_card_html(row["text"], row["sentiment"], str(row.get("source",""))),
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "💬 Sentiment Analysis":

    st.markdown("## 💬 Sentiment Analysis")
    st.markdown(
        "<p style='color:var(--muted);font-family:Share Tech Mono;font-size:0.85rem;'>"
        "How did fans feel? This section dissects the emotional fingerprint of each theory cluster.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if fdf.empty:
        empty_box()
        st.stop()

    # ── Overall donut + key insight ──
    col_a, col_b = st.columns([1, 1.5])

    with col_a:
        st.markdown("### Overall Distribution")
        sc = fdf["sentiment"].value_counts().reset_index()
        sc.columns = ["Sentiment", "Count"]
        dominant = sc.iloc[0]["Sentiment"] if not sc.empty else "N/A"
        pct_dom  = int(sc.iloc[0]["Count"] / sc["Count"].sum() * 100) if not sc.empty else 0
        st.markdown(
            f'<div class="insight-box">'
            f'<strong>{dominant}</strong> sentiment leads at <strong>{pct_dom}%</strong> '
            f'of all {len(fdf):,} discussions. '
            f'Fans had feelings — and they were not shy about them.'
            f'</div>',
            unsafe_allow_html=True,
        )
        fig = px.pie(
            sc, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=SENT_COLORS,
            hole=0.5,
        )
        fig.update_traces(textfont_color="#fff", textfont_size=14)
        fig.update_layout(**pdark(), margin=dict(l=0, r=0, t=10, b=0))
        pchart(fig, 300)

    with col_b:
        st.markdown("### Sentiment by Theme (Stacked Bar)")
        cross = (
            fdf.groupby(["theme", "sentiment"])
               .size()
               .reset_index(name="Count")
        )
        if not cross.empty:
            fig2 = px.bar(
                cross, x="theme", y="Count", color="sentiment",
                color_discrete_map=SENT_COLORS,
                barmode="stack",
                text_auto=False,
            )
            fig2.update_layout(
                **pdark(),
                xaxis_title="", yaxis_title="Discussions",
                xaxis_tickangle=-20,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            pchart(fig2, 360)

    st.markdown("---")

    # ── Percentage stacked horizontal bar ──
    st.markdown("### 📈 Sentiment Proportion per Theme")
    pct = fdf.groupby(["theme", "sentiment"]).size().reset_index(name="Count")
    if not pct.empty:
        pct["Pct"] = (
            pct["Count"] /
            pct.groupby("theme")["Count"].transform("sum") * 100
        ).round(1)
        fig3 = px.bar(
            pct, x="Pct", y="theme", color="sentiment",
            color_discrete_map=SENT_COLORS,
            orientation="h",
            text=pct["Pct"].astype(str) + "%",
            barmode="stack",
        )
        fig3.update_traces(textfont_color="#fff", textposition="inside")
        fig3.update_layout(
            **pdark(),
            xaxis_title="% of discussions", yaxis_title="",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        pchart(fig3, max(300, pct["theme"].nunique() * 58))

    # ── Sentiment score (if populated) ──
    score_df = fdf.dropna(subset=["sentiment_score"])
    if not score_df.empty:
        st.markdown("---")
        st.markdown("### 🔢 Sentiment Score Distribution")
        fig4 = px.histogram(
            score_df, x="sentiment_score", color="sentiment",
            color_discrete_map=SENT_COLORS,
            nbins=40, barmode="overlay", opacity=0.78,
        )
        fig4.update_layout(
            **pdark(), xaxis_title="Score", yaxis_title="Count",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        pchart(fig4, 280)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — THEORY EXPLORER  (most important)
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Theory Explorer":

    st.markdown("## 🔍 Theory Explorer")
    st.markdown(
        "<p style='color:var(--muted);font-family:Share Tech Mono;font-size:0.85rem;'>"
        "Step inside each theory cluster. Read what fans actually said. 📺</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not THEMES:
        empty_box("No themes found in the dataset.")
        st.stop()

    sel = st.selectbox(
        "Choose a Theory Cluster to Explore",
        options=THEMES,
        format_func=lambda t: f"{THEME_ICONS.get(t, THEME_EMOJIS_FALLBACK)}  {t}",
        key="te_theme",
    )
    tdf  = fdf[fdf["theme"] == sel]
    icon = THEME_ICONS.get(sel, THEME_EMOJIS_FALLBACK)

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{icon} Discussions", f"{len(tdf):,}")
    k2.metric("✅ Positive",  f"{(tdf['sentiment']=='Positive').sum():,}")
    k3.metric("😤 Negative",  f"{(tdf['sentiment']=='Negative').sum():,}")
    k4.metric("😐 Neutral",   f"{(tdf['sentiment']=='Neutral').sum():,}")

    # Narrative paragraph
    story = THEME_STORIES.get(sel, f"Fan discussions in the {sel} cluster.")
    st.markdown(
        f'<div class="insight-box">'
        f'📖 <strong>THE STORY</strong><br><br>{story}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Top Keywords ──
    st.markdown("### 🔑 Top Keywords (TF-IDF)")
    keywords = top_tfidf_keywords(df, sel, n=14)  # use full df for TF-IDF context
    if not keywords:
        keywords = freq_keywords(tdf["text"].dropna().tolist(), n=14)

    if keywords:
        pills = "".join(f'<span class="keyword-pill">{kw}</span>' for kw in keywords)
        st.markdown(
            f'<div style="margin:10px 0 20px;">{pills}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Not enough data for keyword extraction.")

    st.markdown("---")

    # ── Real Fan Comments ──
    st.markdown("### 🗣️ Real Fan Comments")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        n_comments = st.slider("How many?", 5, 20, 8, key="te_n")
    with col_ctrl2:
        sent_filter_te = st.selectbox(
            "Sentiment",
            ["All", "Positive", "Neutral", "Negative"],
            key="te_s",
        )
    with col_ctrl3:
        min_len = st.slider("Min. comment length (chars)", 20, 120, 40, key="te_len")

    q_df = tdf if sent_filter_te == "All" else tdf[tdf["sentiment"] == sent_filter_te]
    q_df = q_df[q_df["text"].str.len() >= min_len]

    # Fallback if nothing passes the length filter
    if q_df.empty and not tdf.empty:
        q_df = tdf.nlargest(min(n_comments, len(tdf)), "text")

    if "te_seed" not in st.session_state:
        st.session_state.te_seed = 42

    sample = safe_sample(q_df, n_comments, seed=st.session_state.te_seed)

    if sample is None or sample.empty:
        empty_box("No comments match the current settings.")
    else:
        for _, row in sample.iterrows():
            st.markdown(
                quote_card_html(row["text"], row["sentiment"], str(row.get("source", ""))),
                unsafe_allow_html=True,
            )

    if st.button("🔀 Shuffle Comments", key="te_shuffle"):
        st.session_state.te_seed += 1
        st.rerun()

    st.markdown("---")

    # ── Sentiment mini chart for this theme ──
    st.markdown("### 📊 Sentiment in This Cluster")
    if not tdf.empty:
        ts  = tdf["sentiment"].value_counts().reset_index()
        ts.columns = ["Sentiment", "Count"]
        fig = px.bar(
            ts, x="Sentiment", y="Count",
            color="Sentiment", color_discrete_map=SENT_COLORS,
            text="Count",
        )
        fig.update_traces(textfont_color="#fff", textposition="outside", showlegend=False)
        fig.update_layout(
            **pdark(legend=False),
            xaxis_title="", yaxis_title="",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        pchart(fig, 260)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — FUN INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "✨ Fun Insights":

    st.markdown("## ✨ Fun Insights")
    st.markdown(
        "<p style='color:var(--muted);font-family:Share Tech Mono;font-size:0.85rem;'>"
        "Creative visualizations — because data should tell a story. 😈</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if fdf.empty:
        empty_box()
        st.stop()

    # ── 1. Word Clouds per theme ──
    st.markdown("### ☁️ Word Clouds by Theme")
    wc_theme = st.selectbox(
        "Pick a theme for the word cloud",
        options=THEMES,
        format_func=lambda t: f"{THEME_ICONS.get(t, THEME_EMOJIS_FALLBACK)}  {t}",
        key="wc_theme",
    )
    wc_texts = fdf[fdf["theme"] == wc_theme]["text"].dropna().tolist()
    st.markdown(wordcloud_html(wc_texts), unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Heatmap: theme × sentiment ──
    st.markdown("### 🔥 Heatmap: Themes vs Sentiment")
    heat = (
        fdf.groupby(["theme", "sentiment"])
           .size()
           .unstack(fill_value=0)
    )
    if not heat.empty:
        fig_heat = go.Figure(go.Heatmap(
            z=heat.values,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale=[[0, "#090909"], [0.4, "#CC1512"], [1, "#F5C518"]],
            text=heat.values,
            texttemplate="%{text}",
            textfont={"color": "#fff", "size": 13},
        ))
        fig_heat.update_layout(
            **pdark(legend=False),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Sentiment",
            yaxis_title="Theme",
        )
        pchart(fig_heat, 360)

    st.markdown("---")

    # ── 3. Most "emotional" themes ──
    st.markdown("### 😤 Most Emotional Themes")
    st.markdown(
        "<p style='color:var(--muted);font-family:Share Tech Mono;font-size:0.8rem;'>"
        "Measured by % of Non-Neutral comments per theme.</p>",
        unsafe_allow_html=True,
    )
    emo = fdf.copy()
    emo["is_emotional"] = emo["sentiment"].isin(["Positive", "Negative"]).astype(int)
    emo_by_theme = (
        emo.groupby("theme")["is_emotional"]
           .agg(["sum", "count"])
           .rename(columns={"sum": "emotional", "count": "total"})
           .reset_index()
    )
    emo_by_theme["pct_emotional"] = (
        emo_by_theme["emotional"] / emo_by_theme["total"] * 100
    ).round(1)
    emo_by_theme = emo_by_theme.sort_values("pct_emotional", ascending=True)

    fig_emo = px.bar(
        emo_by_theme, x="pct_emotional", y="theme",
        orientation="h",
        color="pct_emotional",
        color_continuous_scale=["#2A1A1A", "#CC1512", "#F5C518"],
        text=emo_by_theme["pct_emotional"].astype(str) + "%",
    )
    fig_emo.update_traces(textfont_color="#fff", textposition="outside")
    fig_emo.update_layout(
        **pdark(), coloraxis_showscale=False,
        xaxis_title="% Emotional Comments", yaxis_title="",
        margin=dict(l=0, r=30, t=10, b=0),
    )
    pchart(fig_emo, max(280, len(emo_by_theme) * 58))

    st.markdown("---")

    # ── 4. Positive vs Negative ratio bubble chart ──
    st.markdown("### ⚡ Positivity vs Negativity per Theme")
    pn = (
        fdf[fdf["sentiment"].isin(["Positive", "Negative"])]
            .groupby(["theme", "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
    )
    if "Positive" not in pn.columns: pn["Positive"] = 0
    if "Negative" not in pn.columns: pn["Negative"] = 0
    pn["total"]  = pn["Positive"] + pn["Negative"]
    pn["ratio"]  = (pn["Positive"] / pn["total"].replace(0, 1) * 100).round(1)

    fig_pn = px.scatter(
        pn, x="Positive", y="Negative",
        size="total", color="theme",
        text="theme",
        size_max=55,
        color_discrete_sequence=["#CC1512","#E87C2B","#F5C518","#27AE60","#7A6A8A"],
    )
    fig_pn.update_traces(
        textposition="top center",
        textfont_color="#E8D5C0",
        textfont_size=11,
        marker_opacity=0.85,
    )
    fig_pn.update_layout(
        **pdark(),
        xaxis_title="Positive Comments",
        yaxis_title="Negative Comments",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    pchart(fig_pn, 360)

    st.markdown("---")

    # ── 5. Comment length distribution per theme ──
    st.markdown("### 📏 Comment Length Distribution")
    fdf_len = fdf.copy()
    fdf_len["text_len"] = fdf_len["text"].str.len()
    fig_box = px.box(
        fdf_len, x="theme", y="text_len",
        color="theme",
        color_discrete_sequence=["#CC1512","#E87C2B","#F5C518","#27AE60","#7A6A8A"],
    )
    fig_box.update_layout(
        **pdark(),
        showlegend=False,
        xaxis_title="", yaxis_title="Characters",
        xaxis_tickangle=-15,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    pchart(fig_box, 340)


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — CONCLUSION
# ══════════════════════════════════════════════════════════════
elif page == "📋 Conclusion":

    st.markdown("## 📋 Conclusion")
    st.markdown("---")

    # Compute quick stats for dynamic insights
    most_discussed = (
        df["theme"].value_counts().idxmax()
        if not df.empty else "Unknown"
    )
    most_negative = (
        df[df["sentiment"] == "Negative"]["theme"].value_counts().idxmax()
        if (df["sentiment"] == "Negative").any() else "Unknown"
    )
    most_positive = (
        df[df["sentiment"] == "Positive"]["theme"].value_counts().idxmax()
        if (df["sentiment"] == "Positive").any() else "Unknown"
    )
    pct_neg = int(
        (df["sentiment"] == "Negative").sum() / max(1, len(df)) * 100
    )
    pct_pos = int(
        (df["sentiment"] == "Positive").sum() / max(1, len(df)) * 100
    )
    pct_neu = int(
        (df["sentiment"] == "Neutral").sum() / max(1, len(df)) * 100
    )

    st.markdown(f"""
    <div class="insight-box" style="font-size:0.95rem;line-height:2;">
    🎬 <strong>PROJECT SUMMARY</strong><br><br>

    Across <strong>{len(df):,} fan discussions</strong> spanning {len(THEMES)} distinct theory
    clusters, a clear picture emerges of how audiences processed the Stranger Things finale.<br><br>

    📌 The most-discussed theme was
    <strong>{most_discussed}</strong> — fans couldn't stop talking about it.<br>

    😤 The most negatively-charged theme was
    <strong>{most_negative}</strong>, reflecting unmet expectations.<br>

    ✅ The most positively-charged theme was
    <strong>{most_positive}</strong>, anchored in emotional connection.<br><br>

    Overall, <strong>{pct_neu}%</strong> of comments were Neutral,
    <strong>{pct_pos}%</strong> Positive, and
    <strong>{pct_neg}%</strong> Negative — suggesting that while
    the majority of discourse was measured, a significant faction of fans had
    <em>strong</em> feelings about the finale's resolution.<br><br>

    🧠 The clusters reveal a fandom that thinks deeply, theorizes creatively, and
    cares passionately. Stranger Things didn't just end a show — it sparked a conversation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Final theme breakdown recap
    st.markdown("### 🗂 Final Theme Overview")
    recap = df.groupby("theme").agg(
        Discussions=("text", "count"),
        Positive=("sentiment", lambda x: (x=="Positive").sum()),
        Negative=("sentiment", lambda x: (x=="Negative").sum()),
        Neutral=("sentiment",  lambda x: (x=="Neutral").sum()),
    ).reset_index().sort_values("Discussions", ascending=False)
    recap.index = range(1, len(recap)+1)

    st.dataframe(
        recap.style
            .background_gradient(subset=["Discussions"], cmap="Oranges")
            .background_gradient(subset=["Positive"],    cmap="Greens")
            .background_gradient(subset=["Negative"],    cmap="Reds"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-family:Share Tech Mono;font-size:0.75rem;color:#7A6A5A;text-align:center;'>"
        "Built with Streamlit · pandas · Plotly · scikit-learn · Pure CSS Word Cloud<br>"
        "Data sourced from Kaggle fan discussions · Stranger Things Fandom</p>",
        unsafe_allow_html=True,
    )
