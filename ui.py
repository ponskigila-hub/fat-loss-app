"""
ui.py — Premium UI components, CSS, and helper functions for FatLoss AI Dashboard
All styling, reusable HTML blocks, card components, and theme configuration live here.
"""

import streamlit as st


# =====================================
# THEME CONFIGURATION
# =====================================
COLORS = {
    # Greens
    "emerald_deep":   "#0A2E1F",
    "emerald":        "#1B4332",
    "emerald_mid":    "#2D6A4F",
    "emerald_light":  "#40916C",
    "mint":           "#52B788",
    "mint_pale":      "#95D5B2",
    "mint_ghost":     "#B7E4C7",
    "mint_mist":      "#D8F3DC",
    "surface":        "#F0FAF4",
    # Neutrals
    "white":          "#FFFFFF",
    "gray_50":        "#FAFAFA",
    "gray_100":       "#F4F4F5",
    "gray_200":       "#E4E4E7",
    "text_primary":   "#0F1F15",
    "text_secondary": "#3D6650",
    "text_muted":     "#6B9E7E",
    # Accents
    "orange":         "#F4A261",
    "coral":          "#E76F51",
    "amber":          "#F59E0B",
}

CHART_PALETTE = [
    "#2D6A4F", "#52B788", "#95D5B2",
    "#F4A261", "#E76F51", "#B7E4C7",
]


# =====================================
# INJECT GLOBAL CSS
# =====================================
def inject_css():
    """Inject all global CSS into the Streamlit app."""
    st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&family=Geist:wght@300;400;500;600;700&display=swap');

/* ══════════════════════════════════════════
   ROOT VARIABLES
══════════════════════════════════════════ */
:root {
    --emerald-deep:   #0A2E1F;
    --emerald:        #1B4332;
    --emerald-mid:    #2D6A4F;
    --emerald-light:  #40916C;
    --mint:           #52B788;
    --mint-pale:      #95D5B2;
    --mint-ghost:     #B7E4C7;
    --mint-mist:      #D8F3DC;
    --surface:        #F0FAF4;
    --white:          #FFFFFF;
    --text-primary:   #0F1F15;
    --text-secondary: #3D6650;
    --text-muted:     #6B9E7E;
    --orange:         #F4A261;
    --coral:          #E76F51;
    --border:         #C8E6D4;
    --shadow-sm:      0 1px 3px rgba(10,46,31,0.08), 0 1px 2px rgba(10,46,31,0.04);
    --shadow-md:      0 4px 16px rgba(10,46,31,0.10), 0 2px 6px rgba(10,46,31,0.06);
    --shadow-lg:      0 8px 32px rgba(10,46,31,0.14), 0 4px 12px rgba(10,46,31,0.08);
    --radius-sm:      8px;
    --radius-md:      14px;
    --radius-lg:      20px;
    --radius-xl:      28px;
}

/* ══════════════════════════════════════════
   GLOBAL BASE
══════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

.main .block-container {
    padding: 2rem 2.5rem 5rem !important;
    max-width: 1280px !important;
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(200deg,
        var(--emerald-deep) 0%,
        var(--emerald) 50%,
        #163d2a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

[data-testid="stSidebar"] * {
    color: #D4EFE0 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: rgba(212,239,224,0.75) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1rem !important;
    text-align: left !important;
    transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(82,183,136,0.15) !important;
    border-color: rgba(82,183,136,0.35) !important;
    color: #FFFFFF !important;
    transform: translateX(4px) !important;
}

/* ══════════════════════════════════════════
   TYPOGRAPHY
══════════════════════════════════════════ */
h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--emerald) !important;
    font-size: 2.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin-bottom: 0.2rem !important;
}

h2 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--emerald-mid) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    margin-top: 2rem !important;
}

h3 {
    font-family: 'Geist', sans-serif !important;
    color: var(--emerald-mid) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.005em !important;
}

/* ══════════════════════════════════════════
   STREAMLIT NATIVE COMPONENTS
══════════════════════════════════════════ */

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--mint), var(--emerald-light));
}

div[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--emerald) !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.02em !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* Main action buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--emerald-mid), var(--emerald-light)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 2px 8px rgba(45,106,79,0.25) !important;
    letter-spacing: 0.01em !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--emerald), var(--emerald-mid)) !important;
    box-shadow: 0 4px 16px rgba(45,106,79,0.35) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--mint-mist) !important;
    padding: 4px !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: var(--text-muted) !important;
    padding: 0.4rem 1.1rem !important;
    transition: all 0.15s ease !important;
}

.stTabs [aria-selected="true"] {
    background: var(--white) !important;
    color: var(--emerald-mid) !important;
    box-shadow: var(--shadow-sm) !important;
    font-weight: 600 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--emerald-mid) !important;
    border: 3px solid var(--white) !important;
    box-shadow: 0 2px 8px rgba(45,106,79,0.3) !important;
    width: 20px !important;
    height: 20px !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--mint), var(--emerald-light)) !important;
}

/* Select boxes & inputs */
.stSelectbox [data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    background: var(--white) !important;
}

.stNumberInput input, .stTextInput input {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    background: var(--white) !important;
}

.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--mint) !important;
    box-shadow: 0 0 0 3px rgba(82,183,136,0.15) !important;
}

/* Radio buttons */
.stRadio [data-baseweb="radio"] div {
    border-color: var(--emerald-mid) !important;
}

/* Multiselect */
.stMultiSelect [data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
}

/* Alerts */
.stSuccess {
    background: linear-gradient(135deg, #E6F7ED, #F0FAF4) !important;
    border: 1px solid var(--mint-ghost) !important;
    border-left: 4px solid var(--mint) !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
}

.stWarning {
    border-left: 4px solid var(--orange) !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
}

.stInfo {
    border-left: 4px solid var(--mint) !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--mint), var(--emerald-mid)) !important;
    border-radius: 100px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--mint-mist) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    color: var(--emerald-mid) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 2rem 0 !important;
    opacity: 0.6 !important;
}

/* ══════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
══════════════════════════════════════════ */

/* Page subtitle */
.page-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: -0.3rem;
    margin-bottom: 2rem;
    line-height: 1.6;
    max-width: 680px;
}

/* Section label / eyebrow */
.eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--mint);
    margin-bottom: 0.35rem;
}

/* Divider label */
.divider-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 2rem 0 1.25rem;
}

.divider-label::before,
.divider-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Hero Section ── */
.hero-section {
    background: linear-gradient(135deg,
        var(--emerald) 0%,
        var(--emerald-mid) 50%,
        var(--emerald-light) 100%);
    border-radius: var(--radius-xl);
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(82,183,136,0.25) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-section::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(181,228,199,0.12) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.9);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: white;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin-bottom: 0.75rem;
}

.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.72);
    line-height: 1.65;
    max-width: 520px;
    font-weight: 400;
    margin-bottom: 0;
}

.hero-badge {
    position: absolute;
    top: 2rem; right: 3rem;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.2rem;
    text-align: center;
    z-index: 2;
}

.hero-badge-value {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 700;
    color: white;
    line-height: 1;
}

.hero-badge-label {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.7);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}

/* ── Stat Card ── */
.stat-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}

.stat-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-3px);
    border-color: var(--mint-ghost);
}

.stat-card-icon {
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
    display: block;
}

.stat-card-value {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--emerald);
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.stat-card-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

.stat-card-accent {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--mint), var(--emerald-light));
    border-radius: 0 0 var(--radius-md) var(--radius-md);
}

/* ── Feature Card ── */
.feature-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.22s ease;
    height: 100%;
}

.feature-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--mint-ghost);
    transform: translateY(-2px);
}

.feature-icon {
    width: 44px;
    height: 44px;
    background: var(--mint-mist);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}

.feature-title {
    font-weight: 700;
    color: var(--emerald);
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}

.feature-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.55;
}

/* ── Info Banner ── */
.info-banner {
    background: linear-gradient(135deg, var(--mint-mist), #E8F5EE);
    border: 1px solid var(--mint-ghost);
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    font-size: 0.875rem;
    color: var(--emerald);
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}

.info-banner-icon {
    font-size: 1rem;
    margin-top: 1px;
    flex-shrink: 0;
}

/* ── Model Score Card ── */
.model-card {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
    height: 100%;
}

.model-card.best {
    border-color: var(--mint);
    background: linear-gradient(150deg, #FAFFFD, #F0FAF4);
    box-shadow: 0 4px 20px rgba(82,183,136,0.15);
}

.model-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.model-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.model-card-name {
    font-weight: 700;
    color: var(--emerald);
    font-size: 0.95rem;
}

.model-best-badge {
    background: linear-gradient(135deg, var(--mint), var(--emerald-light));
    color: white;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 100px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.model-metric-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.model-metric-chip {
    flex: 1;
    min-width: 55px;
    background: var(--mint-mist);
    border-radius: var(--radius-sm);
    padding: 8px 10px;
    text-align: center;
    border: 1px solid var(--border);
}

.model-metric-chip.r2 {
    background: #E6F7ED;
    border-color: var(--mint-ghost);
}

.model-metric-chip.mae {
    background: #FEF3E8;
    border-color: #F4C094;
}

.model-metric-chip.rmse {
    background: #FDF5E6;
    border-color: #E8C98A;
}

.model-metric-chip-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-muted);
    margin-bottom: 2px;
}

.model-metric-chip-value {
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--emerald);
}

.model-metric-chip.mae .model-metric-chip-value { color: #C4571A; }
.model-metric-chip.rmse .model-metric-chip-value { color: #A0522D; }

/* ── Prediction Result Hero ── */
.result-hero {
    background: linear-gradient(135deg, var(--emerald) 0%, var(--emerald-mid) 60%, #3D8C64 100%);
    border-radius: var(--radius-xl);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}

.result-hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(82,183,136,0.2) 0%, transparent 70%);
    border-radius: 50%;
}

.result-hero-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0.5rem;
}

.result-hero-calories {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: white;
    line-height: 1;
    letter-spacing: -0.03em;
}

.result-hero-unit {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
    font-weight: 400;
    margin-left: 0.4rem;
}

.result-hero-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.4rem;
}

.result-bmi-badge {
    position: absolute;
    top: 2.5rem; right: 3rem;
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: var(--radius-md);
    padding: 0.8rem 1.4rem;
    text-align: center;
    z-index: 2;
}

.result-bmi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: white;
    line-height: 1;
}

.result-bmi-cat {
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 3px;
}

/* ── BMI Status Badge (inline) ── */
.bmi-badge-inline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
}

/* ── Meal Card ── */
.meal-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}

.meal-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--mint), var(--emerald-light));
    border-radius: var(--radius-lg) 0 0 var(--radius-lg);
}

.meal-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-3px);
    border-color: var(--mint-ghost);
}

.meal-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-left: 0.25rem;
}

.meal-name {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--emerald);
    line-height: 1.3;
    letter-spacing: -0.01em;
}

.meal-kcal-badge {
    background: linear-gradient(135deg, #FEF3E8, #FDE8C8);
    color: #C4571A;
    border: 1px solid #F4C094;
    border-radius: var(--radius-sm);
    padding: 5px 12px;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
    flex-shrink: 0;
    margin-left: 1rem;
}

.macro-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--mint-mist);
    color: var(--emerald-mid);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 5px;
    border: 1px solid var(--border);
    transition: background 0.15s;
}

.macro-pill:hover {
    background: var(--mint-ghost);
}

.macro-pill.protein {
    background: #E6F7ED;
    color: #1B4332;
    border-color: var(--mint-ghost);
}

.macro-pill.fat {
    background: #FDF5E6;
    color: #8B4513;
    border-color: #E8C98A;
}

.macro-pill.sugar {
    background: #FFF8E1;
    color: #B8860B;
    border-color: #F0D882;
}

.macro-pill.fiber {
    background: #E8F5E9;
    color: #2D6A4F;
    border-color: #A5D6A7;
}

/* ── Preprocessing Summary Card ── */
.prep-summary {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
}

/* ── Config Panel ── */
.config-panel {
    background: linear-gradient(135deg, var(--mint-mist), #ECFAF1);
    border: 1px solid var(--mint-ghost);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}

/* ── Sidebar nav label ── */
.sidebar-nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(212,239,224,0.4);
    padding: 0 0.5rem;
    margin-bottom: 0.4rem;
    margin-top: 1rem;
}

/* ── Progress bar custom ── */
.progress-bar-wrapper {
    background: var(--mint-mist);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--mint), var(--emerald-light));
    transition: width 0.5s ease;
}

/* ── Insight Card ── */
.insight-card {
    background: linear-gradient(135deg, #E6F7ED, #F0FAF4);
    border: 1px solid var(--mint-ghost);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}

.insight-card.warning {
    background: linear-gradient(135deg, #FEF3E8, #FDF5E6);
    border-color: #F4C094;
}

.insight-text {
    font-size: 0.875rem;
    color: var(--emerald);
    line-height: 1.55;
    font-weight: 400;
}

/* ── Scaler Explanation Card ── */
.scaler-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    height: 100%;
    box-shadow: var(--shadow-sm);
}

.scaler-card-title {
    font-weight: 700;
    color: var(--emerald-mid);
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

.scaler-card-desc {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)


# =====================================
# SIDEBAR BRAND
# =====================================
def render_sidebar_brand():
    st.markdown("""
    <div style="
        padding: 1.5rem 1.25rem 1.25rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.5rem;
    ">
        <div style="
            display:flex; align-items:center; gap:10px; margin-bottom:6px;
        ">
            <div style="
                width:36px; height:36px;
                background: linear-gradient(135deg, #52B788, #2D6A4F);
                border-radius: 10px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.1rem;
                box-shadow: 0 3px 10px rgba(82,183,136,0.35);
                flex-shrink: 0;
            ">🥗</div>
            <div>
                <div style="
                    font-family: 'Fraunces', Georgia, serif;
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: white;
                    letter-spacing: -0.02em;
                    line-height: 1;
                ">FatLoss AI</div>
                <div style="
                    font-size: 0.68rem;
                    color: rgba(212,239,224,0.5);
                    font-weight: 500;
                    letter-spacing: 0.04em;
                    margin-top: 2px;
                ">Smart Diet Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_nav_label(label):
    st.markdown(
        f'<div class="sidebar-nav-label">{label}</div>',
        unsafe_allow_html=True
    )


def render_sidebar_footer():
    st.markdown("""
    <div style="
        margin-top: auto;
        padding: 1rem 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.07);
    ">
        <div style="font-size:0.68rem; color:rgba(212,239,224,0.3); text-align:center; line-height:1.6;">
            Powered by Scikit-learn & Streamlit<br>
            <span style="opacity:0.6;">v2.0 · AI Health Platform</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# PAGE HEADER
# =====================================
def page_header(title: str, subtitle: str = "", eyebrow: str = ""):
    """Render a consistent page header with optional eyebrow and subtitle."""
    if eyebrow:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


# =====================================
# HERO SECTION (HOME)
# =====================================
def render_home_hero(record_count: int, avg_calories: float):
    """Render the home page hero section."""
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-tag">✦ AI-Powered Health Platform</div>
        <div class="hero-title">Your Personal<br>Fat Loss Coach</div>
        <div class="hero-subtitle">
            Predict your maintenance calories using machine learning and receive
            a personalised meal plan built around your body metrics and fat-loss goals.
        </div>
        <div class="hero-badge">
            <div class="hero-badge-value">{record_count:,}</div>
            <div class="hero-badge-label">Health Records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# STAT CARDS (HOME)
# =====================================
def render_stat_card(icon: str, value: str, label: str):
    """Render a single animated stat card."""
    st.markdown(f"""
    <div class="stat-card">
        <span class="stat-card-icon">{icon}</span>
        <div class="stat-card-value">{value}</div>
        <div class="stat-card-label">{label}</div>
        <div class="stat-card-accent"></div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# FEATURE CARDS (HOME WORKFLOW)
# =====================================
def render_feature_cards():
    """Render the 4 workflow feature cards on the home page."""
    steps = [
        ("📊", "1. Explore Data",
         "Visualise distributions, correlations, and statistical patterns in the health dataset."),
        ("⚙️", "2. Preprocess",
         "Configure train/test split and apply StandardScaler or MinMaxScaler to features."),
        ("🤖", "3. Train & Evaluate",
         "Train Linear Regression, Random Forest, or SVR and compare performance metrics."),
        ("🍽", "4. Get Meal Plan",
         "Receive AI-curated fat-loss meal recommendations based on your calorie target."),
    ]

    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# =====================================
# INFO BANNER
# =====================================
def info_banner(text: str, icon: str = "ℹ️"):
    """Render a styled informational banner."""
    st.markdown(f"""
    <div class="info-banner">
        <span class="info-banner-icon">{icon}</span>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# DIVIDER WITH LABEL
# =====================================
def section_divider(label: str = ""):
    """Render a styled divider, optionally with a centred label."""
    if label:
        st.markdown(f'<div class="divider-label">{label}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<hr>", unsafe_allow_html=True)


# =====================================
# MODEL EVALUATION CARDS
# =====================================
def render_model_card(name: str, metrics: dict, is_best: bool = False):
    """Render a model evaluation card with metric chips."""
    badge = '<span class="model-best-badge">🏆 Best</span>' if is_best else ""
    card_class = "model-card best" if is_best else "model-card"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="model-card-header">
            <span class="model-card-name">{name}</span>
            {badge}
        </div>
        <div class="model-metric-row">
            <div class="model-metric-chip r2">
                <div class="model-metric-chip-label">R²</div>
                <div class="model-metric-chip-value">{metrics['R²']}</div>
            </div>
            <div class="model-metric-chip mae">
                <div class="model-metric-chip-label">MAE</div>
                <div class="model-metric-chip-value">{metrics['MAE']}</div>
            </div>
            <div class="model-metric-chip rmse">
                <div class="model-metric-chip-label">RMSE</div>
                <div class="model-metric-chip-value">{metrics['RMSE']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# PREDICTION RESULT HERO
# =====================================
def render_result_hero(maintenance_cal: float, fat_loss_cal: float,
                        deficit_pct: int, bmi: float, bmi_cat: str):
    """Render the full-width calorie result hero card."""
    st.markdown(f"""
    <div class="result-hero">
        <div style="display:flex; gap:3rem; flex-wrap:wrap; position:relative; z-index:2;">
            <div>
                <div class="result-hero-label">Fat Loss Target</div>
                <div>
                    <span class="result-hero-calories">{fat_loss_cal:,.0f}</span>
                    <span class="result-hero-unit">kcal / day</span>
                </div>
                <div class="result-hero-sub">
                    Down from {maintenance_cal:,.0f} kcal maintenance
                    — a {deficit_pct}% calorie deficit
                </div>
            </div>
        </div>
        <div class="result-bmi-badge">
            <div class="result-bmi-value">{bmi:.1f}</div>
            <div class="result-bmi-cat">BMI · {bmi_cat}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# MEAL CARD
# =====================================
def render_meal_card(row, rank: int):
    """Render a premium meal recommendation card."""
    st.markdown(f"""
    <div class="meal-card">
        <div class="meal-card-header">
            <div class="meal-name">
                <span style="color:var(--text-muted);font-size:0.8rem;
                             font-weight:600;font-family:'Geist',sans-serif;
                             margin-right:8px;">#{rank}</span>
                {row["food"]}
            </div>
            <div class="meal-kcal-badge">🔥 {row["Caloric Value"]} kcal</div>
        </div>
        <div style="padding-left:0.25rem;">
            <span class="macro-pill protein">💪 {row["Protein"]}g protein</span>
            <span class="macro-pill">🍞 {row["Carbohydrates"]}g carbs</span>
            <span class="macro-pill fat">🧈 {row["Fat"]}g fat</span>
            <span class="macro-pill sugar">🍬 {row["Sugars"]}g sugar</span>
            <span class="macro-pill fiber">🌿 {row["Dietary Fiber"]}g fiber</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# SCALER EXPLANATION CARDS
# =====================================
def render_scaler_cards():
    """Render explanation cards for each scaler option."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="scaler-card">
            <div class="scaler-card-title">⚖️ StandardScaler</div>
            <div class="scaler-card-desc">
                Transforms each feature to have <strong>zero mean and unit variance</strong>.
                Ideal when features have different units and the model assumes normally distributed data.
                Robust choice for Linear Regression and SVR.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="scaler-card">
            <div class="scaler-card-title">📏 MinMaxScaler</div>
            <div class="scaler-card-desc">
                Scales all features to a <strong>[0, 1] range</strong>.
                Preserves the shape of the original distribution and is
                sensitive to outliers. Works well with bounded neural network inputs.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =====================================
# HEALTH INSIGHT CARDS
# =====================================
def render_health_insights(bmi: float, fat_pct: float, final_cal: float):
    """Render AI-style health insight cards below the prediction."""
    insights = []

    if bmi < 18.5:
        insights.append(("warning", "⚠️ Your BMI indicates you may be underweight. A fat-loss deficit may not be appropriate — consult a nutritionist before proceeding."))
    elif bmi < 25:
        insights.append(("normal", "✅ Your BMI is in the healthy range. A moderate deficit can help you optimise body composition without losing muscle."))
    elif bmi < 30:
        insights.append(("normal", "📉 A consistent calorie deficit combined with resistance training is highly effective for reducing body fat at your current BMI."))
    else:
        insights.append(("normal", "🎯 A structured fat-loss programme at this BMI can significantly improve metabolic health markers within 8–12 weeks."))

    if fat_pct > 30:
        insights.append(("normal", "💪 Higher body fat percentage means your maintenance calories are relatively higher — your deficit window is comfortably wide."))

    if final_cal < 1200:
        insights.append(("warning", "⚠️ Your target calories are below 1,200 kcal. Consider reducing your deficit percentage to preserve lean muscle mass."))

    insights.append(("normal", f"🍽 Aim for {final_cal * 0.3 / 4:.0f}–{final_cal * 0.35 / 4:.0f}g of protein per meal across 4 meals to support muscle retention during fat loss."))

    for kind, text in insights:
        card_class = "insight-card warning" if kind == "warning" else "insight-card"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="insight-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================
# CUSTOM PROGRESS BAR
# =====================================
def custom_progress_bar(value: float, max_value: float, label: str = ""):
    """Render a styled HTML progress bar."""
    pct = min(100, (value / max_value) * 100) if max_value > 0 else 0
    st.markdown(f"""
    <div style="margin-bottom:0.6rem;">
        <div style="display:flex;justify-content:space-between;
                    font-size:0.78rem;color:var(--text-muted);
                    font-weight:500;margin-bottom:4px;">
            <span>{label}</span>
            <span>{value:.1f}g</span>
        </div>
        <div class="progress-bar-wrapper">
            <div class="progress-bar-fill" style="width:{pct:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)