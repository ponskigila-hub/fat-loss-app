import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Fat Loss Diet Recommendation",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --green-deepest: #0D2B1F;
    --green-deep:    #1B4332;
    --green-mid:     #2D6A4F;
    --green-light:   #52B788;
    --green-pale:    #B7E4C7;
    --green-ghost:   #D8F3DC;
    --accent:        #F4A261;
    --accent-warm:   #E76F51;
    --text-primary:  #1C2B22;
    --text-muted:    #557A62;
    --surface:       #FAFDF8;
    --card-bg:       #FFFFFF;
    --border:        #D0EBDA;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.main .block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--green-deepest) 0%, var(--green-deep) 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}

[data-testid="stSidebar"] * {
    color: #E8F5EC !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #E8F5EC !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(82,183,136,0.25) !important;
    border-color: rgba(82,183,136,0.5) !important;
    transform: translateX(3px) !important;
}

/* ── Headings ── */
h1 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--green-deep) !important;
    font-size: 2.1rem !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.25rem !important;
}

h2 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--green-mid) !important;
    font-size: 1.45rem !important;
    font-weight: 400 !important;
    margin-top: 1.8rem !important;
}

h3 {
    color: var(--green-mid) !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 2px 12px rgba(29,67,45,0.07) !important;
    transition: box-shadow 0.2s ease !important;
}

div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 20px rgba(29,67,45,0.12) !important;
}

div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--green-deep) !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--green-mid) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(45,106,79,0.3) !important;
}

.stButton > button:hover {
    background: var(--green-deep) !important;
    box-shadow: 0 4px 16px rgba(45,106,79,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--green-ghost);
    padding: 5px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--text-muted) !important;
    padding: 0.4rem 1rem !important;
}

.stTabs [aria-selected="true"] {
    background: var(--green-mid) !important;
    color: white !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1.5px solid var(--border) !important;
}

/* ── Inputs / Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--green-mid) !important;
}

.stRadio [data-baseweb="radio"] div {
    border-color: var(--green-mid) !important;
}

.stSelectbox [data-baseweb="select"] div {
    border-radius: 10px !important;
    border-color: var(--border) !important;
}

.stNumberInput input, .stTextInput input {
    border-radius: 10px !important;
    border-color: var(--border) !important;
}

/* ── Alerts ── */
.stSuccess {
    border-left: 4px solid var(--green-light) !important;
    border-radius: 0 10px 10px 0 !important;
}

.stWarning {
    border-left: 4px solid var(--accent) !important;
    border-radius: 0 10px 10px 0 !important;
}

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--green-light), var(--green-mid)) !important;
    border-radius: 100px !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Meal cards ── */
.meal-card {
    background: var(--card-bg);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(29,67,45,0.05);
    transition: box-shadow 0.2s, transform 0.2s;
}

.meal-card:hover {
    box-shadow: 0 6px 24px rgba(29,67,45,0.12);
    transform: translateY(-2px);
}

.meal-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: var(--green-deep);
    margin-bottom: 0.75rem;
}

.macro-pill {
    display: inline-block;
    background: var(--green-ghost);
    color: var(--green-mid);
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
    border: 1px solid var(--green-pale);
}

.macro-pill.kcal {
    background: #FEF3E8;
    color: #C4571A;
    border-color: #F4C094;
}

.macro-pill.protein {
    background: #EAF4EE;
    color: #1B4332;
    border-color: #95D5B2;
}

.macro-pill.fat {
    background: #FDF5E6;
    color: #A0522D;
    border-color: #E8C98A;
}

.macro-pill.fiber {
    background: #D8F3DC;
    color: #2D6A4F;
    border-color: #74C69D;
}

/* ── Page subtitle ── */
.page-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: -0.4rem;
    margin-bottom: 1.8rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--green-light);
    margin-bottom: 0.3rem;
}

/* ── Result box ── */
.result-highlight {
    background: linear-gradient(135deg, var(--green-ghost) 0%, #FFFFFF 100%);
    border: 1.5px solid var(--green-pale);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    padding: 1rem 0 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1.5rem;
}

.sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    font-weight: 400;
    color: white;
}

.sidebar-tagline {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.55);
    margin-top: 2px;
}

/* ── Badge ── */
.bmi-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# SESSION STATE
# =====================================
for key in [
    "page", "df", "food_df", "X_train", "X_test",
    "y_train", "y_test", "scaler", "scaler_name",
    "models", "eval_results"
]:
    if key not in st.session_state:
        st.session_state[key] = "Home" if key == "page" else None


# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_main_data():
    return pd.read_csv("Final_data.csv")


@st.cache_data
def load_food_data():
    return pd.read_csv("FOOD-DATA-GROUP5.csv")


df = load_main_data()
food_df = load_food_data()

st.session_state.df = df
st.session_state.food_df = food_df

features = ["Age", "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage"]
target = "Calories"


# =====================================
# HELPERS
# =====================================
def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def list_saved_pickles():
    return sorted([f for f in os.listdir(".") if f.endswith(".pkl")])


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#FFF3E0", "#E65100"
    elif bmi < 25:
        return "Normal", "#E8F5E9", "#2E7D32"
    elif bmi < 30:
        return "Overweight", "#FFF8E1", "#F57F17"
    else:
        return "Obese", "#FFEBEE", "#C62828"


def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#FAFDF8",
        "figure.facecolor": "#FAFDF8",
        "axes.edgecolor": "#D0EBDA",
        "axes.labelcolor": "#557A62",
        "xtick.color": "#557A62",
        "ytick.color": "#557A62",
        "axes.grid": True,
        "grid.color": "#E8F5E9",
        "grid.linewidth": 0.8,
    })


# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-title">🥗 FatLoss AI</div>
        <div class="sidebar-tagline">Smart Diet Recommendation</div>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "🏠  Home": "Home",
        "📊  Explore Data": "EDA",
        "⚙️  Preprocessing": "Preprocessing",
        "🤖  Train Model": "Model",
        "🍽  Get Recommendation": "Prediction"
    }

    for label, page in pages.items():
        if st.button(label, use_container_width=True, key=f"nav_{page}"):
            st.session_state.page = page
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:rgba(255,255,255,0.35);text-align:center;'>Powered by Scikit-learn & Streamlit</div>",
        unsafe_allow_html=True
    )


# =====================================
# HOME
# =====================================
if st.session_state.page == "Home":
    st.title("Fat Loss Diet Recommendation")
    st.markdown(
        "<div class='page-sub'>Predict your maintenance calories and get personalized meal recommendations tailored to your fat-loss goals.</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg. Maintenance Cal.", f"{df[target].mean():,.0f} kcal")
    c3.metric("Average BMI", f"{df['BMI'].mean():.1f}")
    c4.metric("Features Used", len(features))

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(8), width="stretch", height=280)

    with col_r:
        st.subheader("Quick Stats")
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ["#2D6A4F", "#52B788", "#B7E4C7", "#F4A261", "#E76F51"]
        df[features[:4]].mean().plot(
            kind="barh", ax=ax,
            color=colors[:4], edgecolor="white", linewidth=0.5
        )
        ax.set_xlabel("Mean Value", fontsize=9)
        ax.set_title("Feature Averages", fontsize=10, fontweight="bold", color="#1B4332")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(135deg,#D8F3DC,#B7E4C7);border-radius:14px;padding:1.2rem 1.5rem;border:1.5px solid #95D5B2;'>
    <span style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#2D6A4F;'>How It Works</span>
    <div style='display:flex;gap:2rem;margin-top:0.7rem;flex-wrap:wrap;'>
        <div style='flex:1;min-width:150px;'>
            <div style='font-size:1.3rem;margin-bottom:4px;'>📊</div>
            <div style='font-weight:600;color:#1B4332;font-size:0.9rem;'>1. Explore Data</div>
            <div style='font-size:0.8rem;color:#557A62;'>Visualise distributions, correlations, and patterns.</div>
        </div>
        <div style='flex:1;min-width:150px;'>
            <div style='font-size:1.3rem;margin-bottom:4px;'>⚙️</div>
            <div style='font-weight:600;color:#1B4332;font-size:0.9rem;'>2. Preprocess</div>
            <div style='font-size:0.8rem;color:#557A62;'>Scale and split data for training.</div>
        </div>
        <div style='flex:1;min-width:150px;'>
            <div style='font-size:1.3rem;margin-bottom:4px;'>🤖</div>
            <div style='font-weight:600;color:#1B4332;font-size:0.9rem;'>3. Train & Evaluate</div>
            <div style='font-size:0.8rem;color:#557A62;'>Compare Linear Regression, Random Forest, SVR.</div>
        </div>
        <div style='flex:1;min-width:150px;'>
            <div style='font-size:1.3rem;margin-bottom:4px;'>🍽</div>
            <div style='font-weight:600;color:#1B4332;font-size:0.9rem;'>4. Get Meals</div>
            <div style='font-size:0.8rem;color:#557A62;'>Receive personalised fat-loss meal recommendations.</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# EDA
# =====================================
elif st.session_state.page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("<div class='page-sub'>Understand the dataset before modelling — distributions, correlations, and feature relationships.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📈 Histograms", "🔥 Heatmap", "🔵 Scatter"])

    set_plot_style()

    with tab1:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), width="stretch")
        with col_r:
            st.subheader("Statistical Summary")
            st.dataframe(df[features + [target]].describe().round(2), width="stretch")

    with tab2:
        selected_features = st.multiselect(
            "Select features to plot",
            features,
            default=features[:3]
        )
        if selected_features:
            n = len(selected_features)
            cols_per_row = 2
            rows = (n + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(11, 3.5 * rows))
            axes = np.array(axes).flatten() if n > 1 else [axes]
            palette = ["#2D6A4F", "#52B788", "#95D5B2", "#F4A261", "#E76F51"]
            for idx, feat in enumerate(selected_features):
                sns.histplot(df[feat], kde=True, ax=axes[idx],
                             color=palette[idx % len(palette)], alpha=0.75,
                             edgecolor="white", linewidth=0.5)
                axes[idx].set_title(feat, fontweight="bold", fontsize=10)
                axes[idx].set_xlabel("")
            for j in range(idx + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)

    with tab3:
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(df[features + [target]].corr(), dtype=bool), k=1)
        sns.heatmap(
            df[features + [target]].corr(),
            annot=True, fmt=".2f",
            cmap=sns.diverging_palette(145, 20, s=70, l=40, as_cmap=True),
            ax=ax, linewidths=2, linecolor="white",
            annot_kws={"size": 9, "weight": "bold"},
            vmin=-1, vmax=1,
            square=True
        )
        ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold",
                     color="#1B4332", pad=12)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab4:
        selected_feature = st.selectbox("X-axis feature", features)
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.regplot(
            x=df[selected_feature].values,
            y=df[target].values,
            ax=ax,
            scatter_kws={"alpha": 0.45, "facecolor": "#52B788", "s": 22, "edgecolor": "white", "linewidths": 0.4},
            line_kws={"color": "#E76F51", "linewidth": 2},
        )
        ax.set_xlabel(selected_feature, fontsize=10)
        ax.set_ylabel("Calories", fontsize=10)
        ax.set_title(f"{selected_feature} vs Calories", fontweight="bold",
                     fontsize=11, color="#1B4332")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)


# =====================================
# PREPROCESSING
# =====================================
elif st.session_state.page == "Preprocessing":
    st.title("Data Preprocessing")
    st.markdown("<div class='page-sub'>Configure the train/test split and feature scaling before training.</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("#### ⚙️ Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20,
                                  help="Percentage of data reserved for testing.")
        with col2:
            random_state = st.number_input("Random State", 1, 999, 42,
                                           help="Seed for reproducibility.")
        with col3:
            scaler_option = st.radio(
                "Scaler",
                ["StandardScaler", "MinMaxScaler"],
                help="StandardScaler: zero mean, unit variance. MinMaxScaler: [0,1] range."
            )

    st.markdown(f"""
    <div style='background:#EAF4EE;border:1.5px solid #95D5B2;border-radius:12px;
                padding:0.9rem 1.2rem;margin:1rem 0;font-size:0.88rem;color:#1B4332;'>
    ℹ️ This will split <strong>{len(df):,} records</strong> into
    <strong>{int(len(df)*(1-test_size/100)):,} training</strong> and
    <strong>{int(len(df)*test_size/100):,} test</strong> samples, then apply
    <strong>{scaler_option}</strong>.
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  Run Preprocessing"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=random_state
        )

        scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state.X_train = X_train_scaled
        st.session_state.X_test = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.scaler_name = scaler_option

        st.success("✅ Preprocessing complete! Proceed to **Train Model**.")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Scaled Training Sample (first 5 rows)**")
            st.dataframe(
                pd.DataFrame(X_train_scaled[:5], columns=features).round(4),
                width="stretch"
            )
        with col_r:
            st.markdown("**Split Summary**")
            split_df = pd.DataFrame({
                "Set": ["Train", "Test"],
                "Records": [len(X_train), len(X_test)],
                "Share (%)": [100 - test_size, test_size]
            })
            st.dataframe(split_df, width="stretch", hide_index=True)


# =====================================
# MODEL
# =====================================
elif st.session_state.page == "Model":
    st.title("Model Training & Evaluation")
    st.markdown("<div class='page-sub'>Train one or more regression models and compare their performance metrics.</div>", unsafe_allow_html=True)

    if st.session_state.X_train is None:
        st.warning("⚠️ Please run **Preprocessing** before training.")
        st.stop()

    selected_models = st.multiselect(
        "Select models to train",
        ["Linear Regression", "Random Forest", "SVR"],
        default=["Linear Regression"],
        help="You can compare multiple models side-by-side."
    )

    if st.button("🚀  Train & Evaluate"):
        if not selected_models:
            st.warning("Select at least one model.")
        else:
            results = {}
            trained_models = {}
            progress = st.progress(0, text="Training in progress…")

            for i, model_name in enumerate(selected_models):
                progress.progress((i + 1) / len(selected_models),
                                  text=f"Training {model_name}…")

                model_map = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "SVR": SVR()
                }
                model = model_map[model_name]
                model.fit(st.session_state.X_train, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test)

                results[model_name] = {
                    "R²": round(r2_score(st.session_state.y_test, y_pred), 4),
                    "MAE": round(mean_absolute_error(st.session_state.y_test, y_pred), 2),
                    "MSE": round(mean_squared_error(st.session_state.y_test, y_pred), 2),
                    "RMSE": round(np.sqrt(mean_squared_error(st.session_state.y_test, y_pred)), 2)
                }
                trained_models[model_name] = model

            progress.empty()
            st.session_state.models = trained_models
            st.session_state.eval_results = results
            st.success("✅ Training complete!")

    if st.session_state.eval_results:
        st.markdown("---")
        st.subheader("Evaluation Results")

        results_df = pd.DataFrame(st.session_state.eval_results).T
        best_model = results_df["R²"].idxmax()

        # Metric tiles per model
        model_names = list(st.session_state.eval_results.keys())
        cols = st.columns(len(model_names))
        for col, mname in zip(cols, model_names):
            r = st.session_state.eval_results[mname]
            badge = " 🏆" if mname == best_model else ""
            col.markdown(f"""
            <div style='background:#FAFDF8;border:1.5px solid {"#52B788" if mname==best_model else "#D0EBDA"};
                        border-radius:14px;padding:1rem 1.2rem;'>
              <div style='font-weight:700;color:#1B4332;margin-bottom:8px;font-size:0.95rem;'>{mname}{badge}</div>
              <div style='display:flex;flex-wrap:wrap;gap:8px;'>
                <div style='background:#E8F5E9;border-radius:8px;padding:6px 12px;text-align:center;flex:1;min-width:60px;'>
                  <div style='font-size:0.7rem;color:#557A62;font-weight:600;text-transform:uppercase;'>R²</div>
                  <div style='font-size:1.15rem;font-weight:700;color:#2D6A4F;'>{r["R²"]}</div>
                </div>
                <div style='background:#FEF3E8;border-radius:8px;padding:6px 12px;text-align:center;flex:1;min-width:60px;'>
                  <div style='font-size:0.7rem;color:#8D6E63;font-weight:600;text-transform:uppercase;'>MAE</div>
                  <div style='font-size:1.15rem;font-weight:700;color:#C4571A;'>{r["MAE"]}</div>
                </div>
                <div style='background:#FDF5E6;border-radius:8px;padding:6px 12px;text-align:center;flex:1;min-width:60px;'>
                  <div style='font-size:0.7rem;color:#A0522D;font-weight:600;text-transform:uppercase;'>RMSE</div>
                  <div style='font-size:1.15rem;font-weight:700;color:#A0522D;'>{r["RMSE"]}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Bar chart comparison
        if len(model_names) > 1:
            set_plot_style()
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            metrics = ["R²", "MAE", "RMSE"]
            colors_m = ["#2D6A4F", "#52B788", "#B7E4C7", "#F4A261"]
            for ax, metric in zip(axes, metrics):
                vals = [st.session_state.eval_results[m][metric] for m in model_names]
                bars = ax.bar(model_names, vals,
                              color=[colors_m[i % len(colors_m)] for i in range(len(model_names))],
                              edgecolor="white", linewidth=0.5, zorder=3)
                ax.set_title(metric, fontweight="bold", fontsize=10, color="#1B4332")
                ax.set_xticklabels(model_names, fontsize=8, rotation=15, ha="right")
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            save_model = st.selectbox(
                "Select model to save",
                list(st.session_state.models.keys()),
                help="The saved model will be used in the Recommendation page."
            )
        with col_b:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("💾  Save Model & Scaler"):
                save_pickle(st.session_state.models[save_model], "MainModel.pkl")
                save_pickle(st.session_state.scaler, "scaler.pkl")
                st.success(f"✅ **{save_model}** and scaler saved successfully!")


# =====================================
# PREDICTION / RECOMMENDATION
# =====================================
elif st.session_state.page == "Prediction":
    st.title("Diet Recommendation")
    st.markdown("<div class='page-sub'>Enter your body metrics to receive a personalised fat-loss calorie target and meal suggestions.</div>", unsafe_allow_html=True)

    model_files = list_saved_pickles()

    if "MainModel.pkl" not in model_files:
        st.warning("⚠️ No saved model found. Please **Train & Save** a model first.")
        st.stop()

    with open("MainModel.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    st.markdown("#### 📋 Your Body Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", 15, 70, 25)
        weight = st.number_input("Weight (kg)", 30, 200, 75)
    with col2:
        height = st.number_input("Height (m)", 1.0, 2.5, 1.75, step=0.01)
        bmi = st.number_input("BMI", 10.0, 50.0, 24.0, step=0.1)
    with col3:
        fat_percentage = st.number_input("Body Fat (%)", 5.0, 50.0, 20.0, step=0.5)
        target_fat_loss = st.slider("Target Calorie Deficit (%)", 1, 25, 10,
                                    help="10-20% deficit is typically recommended for safe fat loss.")

    st.markdown("---")

    if st.button("🔍  Generate Recommendation"):

        input_data = np.array([[age, weight, height, bmi, fat_percentage]])
        input_scaled = scaler.transform(input_data)
        predicted_calories = model.predict(input_scaled)[0]

        calorie_deficit = predicted_calories * (target_fat_loss / 100)
        final_calories = predicted_calories - calorie_deficit

        cat, badge_bg, badge_color = bmi_category(bmi)

        # ── Result summary ──────────────────────────────────────────
        st.markdown("#### 📊 Your Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Maintenance Calories", f"{predicted_calories:,.0f} kcal")
        r2.metric("Fat Loss Calories", f"{final_calories:,.0f} kcal",
                  delta=f"-{calorie_deficit:,.0f} kcal", delta_color="inverse")
        r3.metric("Calorie Deficit", f"{target_fat_loss}%")
        r4.metric("BMI Category", cat)

        # Calorie bar visual
        set_plot_style()
        fig, ax = plt.subplots(figsize=(7, 2.2))
        bar_data = [predicted_calories, final_calories]
        bar_labels = ["Maintenance", "Fat Loss Target"]
        bar_colors = ["#52B788", "#F4A261"]
        bars = ax.barh(bar_labels, bar_data, color=bar_colors,
                       height=0.45, edgecolor="white", linewidth=0.5, zorder=3)
        for bar, val in zip(bars, bar_data):
            ax.text(val + max(bar_data) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f} kcal", va="center", fontsize=10, fontweight="bold",
                    color="#1B4332")
        ax.set_xlim(0, max(bar_data) * 1.25)
        ax.set_xlabel("Calories (kcal)", fontsize=9)
        ax.set_title("Calorie Comparison", fontsize=10, fontweight="bold", color="#1B4332")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # ── Meal recommendations ────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🍽 Recommended Meals")
        st.markdown(
            f"<div class='page-sub'>Foods with ≤ {final_calories/4:,.0f} kcal per serving, high protein, and low sugar — ranked by nutrition density.</div>",
            unsafe_allow_html=True
        )

        recommended_meals = food_df[
            (food_df["Caloric Value"] <= final_calories / 4) &
            (food_df["Protein"] >= 10) &
            (food_df["Sugars"] <= 10)
        ].sort_values(by=["Nutrition Density", "Protein"], ascending=False)

        if recommended_meals.empty:
            st.info("No meals match the current criteria. Try adjusting your inputs.")
        else:
            for _, row in recommended_meals.head(6).iterrows():
                st.markdown(f"""
                <div class="meal-card">
                    <div class="meal-name">🍽 {row["food"]}</div>
                    <div>
                        <span class="macro-pill kcal">🔥 {row["Caloric Value"]} kcal</span>
                        <span class="macro-pill protein">💪 {row["Protein"]}g protein</span>
                        <span class="macro-pill">🍞 {row["Carbohydrates"]}g carbs</span>
                        <span class="macro-pill fat">🧈 {row["Fat"]}g fat</span>
                        <span class="macro-pill">🍬 {row["Sugars"]}g sugar</span>
                        <span class="macro-pill fiber">🌿 {row["Dietary Fiber"]}g fiber</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Macros chart for top meal ───────────────────────────────
        if not recommended_meals.empty:
            top = recommended_meals.iloc[0]
            st.markdown("---")
            st.markdown(f"**Macro breakdown — {top['food']}**")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            macro_vals = [top["Protein"], top["Carbohydrates"], top["Fat"], top["Dietary Fiber"]]
            macro_labels = ["Protein", "Carbs", "Fat", "Fiber"]
            macro_colors = ["#2D6A4F", "#52B788", "#E76F51", "#95D5B2"]
            wedges, texts, autotexts = ax2.pie(
                macro_vals, labels=macro_labels, colors=macro_colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 9}
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax2.set_facecolor("#FAFDF8")
            fig2.patch.set_facecolor("#FAFDF8")
            plt.tight_layout()
            st.pyplot(fig2)
