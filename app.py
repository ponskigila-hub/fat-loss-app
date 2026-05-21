import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================
# PREMIUM UI IMPORTS
# =====================================
from ui import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_footer,
    render_sidebar_nav_label,
    page_header,
    render_home_hero,
    render_stat_card,
    render_feature_cards,
    info_banner,
    section_divider,
    render_model_card,
    render_result_hero,
    render_meal_card,
    render_scaler_cards,
    render_health_insights,
    CHART_PALETTE,
)

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="FatLoss AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD PREMIUM UI
# =====================================
inject_css()

# =====================================
# SESSION STATE
# =====================================
for key in [
    "page",
    "df",
    "food_df",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "scaler",
    "scaler_name",
    "models",
    "eval_results"
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

features = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "BMI",
    "Fat_Percentage"
]

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
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#F0FAF4",
        "figure.facecolor": "#F0FAF4",
        "axes.edgecolor": "#D8F3DC",
        "axes.labelcolor": "#3D6650",
        "xtick.color": "#3D6650",
        "ytick.color": "#3D6650",
        "axes.grid": True,
        "grid.color": "#E8F5E9",
        "grid.linewidth": 0.8,
    })


# =====================================
# SIDEBAR
# =====================================
with st.sidebar:

    render_sidebar_brand()

    render_sidebar_nav_label("Navigation")

    pages = {
        "🏠 Dashboard": "Home",
        "📊 Explore Data": "EDA",
        "⚙️ Preprocessing": "Preprocessing",
        "🤖 Train Model": "Model",
        "🍽 Recommendation": "Prediction"
    }

    for label, page in pages.items():
        if st.button(label, use_container_width=True, key=f"nav_{page}"):
            st.session_state.page = page
            st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    render_sidebar_footer()

# =====================================
# HOME PAGE
# =====================================
if st.session_state.page == "Home":

    render_home_hero(
        record_count=len(df),
        avg_calories=df[target].mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_stat_card(
            "📁",
            f"{len(df):,}",
            "TOTAL RECORDS"
        )

    with c2:
        render_stat_card(
            "🔥",
            f"{df[target].mean():,.0f}",
            "AVG CALORIES"
        )

    with c3:
        render_stat_card(
            "📏",
            f"{df['BMI'].mean():.1f}",
            "AVERAGE BMI"
        )

    with c4:
        render_stat_card(
            "🧠",
            str(len(features)),
            "ML FEATURES"
        )

    section_divider("Machine Learning Workflow")

    render_feature_cards()

    section_divider("Dataset Overview")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("Quick Statistics")

        set_plot_style()

        fig, ax = plt.subplots(figsize=(5, 4))

        df[features[:4]].mean().plot(
            kind="barh",
            ax=ax,
            color=CHART_PALETTE[:4],
            edgecolor="white",
            linewidth=1
        )

        ax.set_xlabel("Mean Value")
        ax.set_title("Feature Averages")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

# =====================================
# EDA PAGE
# =====================================
elif st.session_state.page == "EDA":

    page_header(
        "Exploratory Data Analysis",
        "Understand distributions, correlations, and feature relationships before modelling.",
        eyebrow="Analytics Dashboard"
    )

    set_plot_style()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dataset",
        "📈 Histograms",
        "🔥 Heatmap",
        "🔵 Scatter"
    ])

    # =====================================
    # TAB 1
    # =====================================
    with tab1:

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("Statistical Summary")
            st.dataframe(
                df[features + [target]].describe().round(2),
                use_container_width=True
            )

    # =====================================
    # TAB 2
    # =====================================
    with tab2:

        selected_features = st.multiselect(
            "Select Features",
            features,
            default=features[:3]
        )

        if selected_features:

            n = len(selected_features)
            cols_per_row = 2
            rows = (n + cols_per_row - 1) // cols_per_row

            fig, axes = plt.subplots(
                rows,
                cols_per_row,
                figsize=(11, 4 * rows)
            )

            axes = np.array(axes).flatten()

            for idx, feat in enumerate(selected_features):

                sns.histplot(
                    df[feat],
                    kde=True,
                    ax=axes[idx],
                    color=CHART_PALETTE[idx % len(CHART_PALETTE)],
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=1
                )

                axes[idx].set_title(
                    feat,
                    fontsize=11,
                    fontweight="bold"
                )

            for j in range(idx + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)

    # =====================================
    # TAB 3
    # =====================================
    with tab3:

        fig, ax = plt.subplots(figsize=(9, 7))

        sns.heatmap(
            df[features + [target]].corr(),
            annot=True,
            fmt=".2f",
            cmap="Greens",
            linewidths=1.5,
            linecolor="white",
            square=True,
            ax=ax
        )

        ax.set_title(
            "Feature Correlation Matrix",
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

    # =====================================
    # TAB 4
    # =====================================
    with tab4:

        selected_feature = st.selectbox(
            "X-axis Feature",
            features
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.regplot(
            x=df[selected_feature],
            y=df[target],
            ax=ax,
            scatter_kws={
                "alpha": 0.5,
                "s": 35,
                "color": CHART_PALETTE[1]
            },
            line_kws={
                "color": CHART_PALETTE[4],
                "linewidth": 2
            }
        )

        ax.set_title(
            f"{selected_feature} vs Calories",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

# =====================================
# PREPROCESSING PAGE
# =====================================
elif st.session_state.page == "Preprocessing":

    page_header(
        "Data Preprocessing",
        "Configure train/test split and scaling before model training.",
        eyebrow="Data Pipeline"
    )

    render_scaler_cards()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider(
            "Test Size (%)",
            10,
            40,
            20
        )

    with col2:
        random_state = st.number_input(
            "Random State",
            1,
            999,
            42
        )

    with col3:
        scaler_option = st.radio(
            "Scaler",
            ["StandardScaler", "MinMaxScaler"]
        )

    info_banner(
        f"""
        Dataset will be split into
        {int(len(df)*(1-test_size/100)):,} training rows and
        {int(len(df)*test_size/100):,} testing rows using
        {scaler_option}.
        """
    )

    if st.button("▶ Run Preprocessing"):

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=random_state
        )

        scaler = (
            StandardScaler()
            if scaler_option == "StandardScaler"
            else MinMaxScaler()
        )

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state.X_train = X_train_scaled
        st.session_state.X_test = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.scaler_name = scaler_option

        st.success("✅ Preprocessing Complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Scaled Training Sample")

            st.dataframe(
                pd.DataFrame(
                    X_train_scaled[:5],
                    columns=features
                ).round(4),
                use_container_width=True
            )

        with col2:
            st.subheader("Split Summary")

            split_df = pd.DataFrame({
                "Set": ["Train", "Test"],
                "Records": [len(X_train), len(X_test)],
                "Share (%)": [
                    100 - test_size,
                    test_size
                ]
            })

            st.dataframe(
                split_df,
                use_container_width=True,
                hide_index=True
            )

# =====================================
# MODEL PAGE
# =====================================
elif st.session_state.page == "Model":

    page_header(
        "Model Training & Evaluation",
        "Train and compare machine learning regression models.",
        eyebrow="Machine Learning"
    )

    if st.session_state.X_train is None:
        st.warning("⚠️ Please run preprocessing first.")
        st.stop()

    selected_models = st.multiselect(
        "Select Models",
        [
            "Linear Regression",
            "Random Forest",
            "SVR"
        ],
        default=["Linear Regression"]
    )

    if st.button("🚀 Train & Evaluate"):

        if not selected_models:
            st.warning("Select at least one model.")

        else:

            results = {}
            trained_models = {}

            progress = st.progress(
                0,
                text="Training models..."
            )

            for i, model_name in enumerate(selected_models):

                progress.progress(
                    (i + 1) / len(selected_models),
                    text=f"Training {model_name}..."
                )

                model_map = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(
                        random_state=42
                    ),
                    "SVR": SVR()
                }

                model = model_map[model_name]

                model.fit(
                    st.session_state.X_train,
                    st.session_state.y_train
                )

                y_pred = model.predict(
                    st.session_state.X_test
                )

                results[model_name] = {
                    "R²": round(
                        r2_score(
                            st.session_state.y_test,
                            y_pred
                        ),
                        4
                    ),
                    "MAE": round(
                        mean_absolute_error(
                            st.session_state.y_test,
                            y_pred
                        ),
                        2
                    ),
                    "RMSE": round(
                        np.sqrt(
                            mean_squared_error(
                                st.session_state.y_test,
                                y_pred
                            )
                        ),
                        2
                    )
                }

                trained_models[model_name] = model

            progress.empty()

            st.session_state.models = trained_models
            st.session_state.eval_results = results

            st.success("✅ Training Complete!")

    # =====================================
    # RESULTS
    # =====================================
    if st.session_state.eval_results:

        section_divider("Model Evaluation")

        results_df = pd.DataFrame(
            st.session_state.eval_results
        ).T

        best_model = results_df["R²"].idxmax()

        cols = st.columns(
            len(st.session_state.eval_results)
        )

        for col, mname in zip(
            cols,
            st.session_state.eval_results.keys()
        ):

            with col:

                render_model_card(
                    name=mname,
                    metrics=st.session_state.eval_results[mname],
                    is_best=(mname == best_model)
                )

        section_divider("Performance Comparison")

        if len(st.session_state.eval_results) > 1:

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(13, 4)
            )

            metrics = ["R²", "MAE", "RMSE"]

            for ax, metric in zip(axes, metrics):

                values = [
                    st.session_state.eval_results[m][metric]
                    for m in st.session_state.eval_results
                ]

                bars = ax.bar(
                    list(st.session_state.eval_results.keys()),
                    values,
                    color=CHART_PALETTE[:len(values)]
                )

                ax.set_title(metric)

                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha="center",
                        va="bottom"
                    )

            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)

        section_divider("Save Model")

        col1, col2 = st.columns([2, 1])

        with col1:

            save_model = st.selectbox(
                "Select Model to Save",
                list(st.session_state.models.keys())
            )

        with col2:

            st.markdown(
                "<div style='height:28px'></div>",
                unsafe_allow_html=True
            )

            if st.button("💾 Save Model"):

                save_pickle(
                    st.session_state.models[save_model],
                    "MainModel.pkl"
                )

                save_pickle(
                    st.session_state.scaler,
                    "scaler.pkl"
                )

                st.success(
                    f"✅ {save_model} saved successfully!"
                )

# =====================================
# PREDICTION PAGE
# =====================================
elif st.session_state.page == "Prediction":

    page_header(
        "AI Diet Recommendation",
        "Generate personalised calorie targets and meal recommendations.",
        eyebrow="AI Nutrition System"
    )

    model_files = list_saved_pickles()

    if "MainModel.pkl" not in model_files:

        st.warning(
            "⚠️ Please train and save a model first."
        )

        st.stop()

    with open("MainModel.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    section_divider("Body Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)",
            15,
            70,
            25
        )

        weight = st.number_input(
            "Weight (kg)",
            30,
            200,
            75
        )

    with col2:

        height = st.number_input(
            "Height (m)",
            1.0,
            2.5,
            1.75,
            step=0.01
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            50.0,
            24.0,
            step=0.1
        )

    with col3:

        fat_percentage = st.number_input(
            "Body Fat (%)",
            5.0,
            50.0,
            20.0,
            step=0.5
        )

        target_fat_loss = st.slider(
            "Target Calorie Deficit (%)",
            1,
            25,
            10
        )

    if st.button("🔍 Generate Recommendation"):

        input_data = np.array([[
            age,
            weight,
            height,
            bmi,
            fat_percentage
        ]])

        input_scaled = scaler.transform(input_data)

        predicted_calories = model.predict(
            input_scaled
        )[0]

        calorie_deficit = (
            predicted_calories
            * (target_fat_loss / 100)
        )

        final_calories = (
            predicted_calories
            - calorie_deficit
        )

        cat = bmi_category(bmi)

        # =====================================
        # HERO RESULT
        # =====================================
        render_result_hero(
            maintenance_cal=predicted_calories,
            fat_loss_cal=final_calories,
            deficit_pct=target_fat_loss,
            bmi=bmi,
            bmi_cat=cat
        )

        # =====================================
        # METRICS
        # =====================================
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Maintenance Calories",
            f"{predicted_calories:,.0f} kcal"
        )

        m2.metric(
            "Fat Loss Calories",
            f"{final_calories:,.0f} kcal"
        )

        m3.metric(
            "Calorie Deficit",
            f"{target_fat_loss}%"
        )

        # =====================================
        # CALORIE COMPARISON
        # =====================================
        section_divider("Calorie Comparison")

        fig, ax = plt.subplots(figsize=(7, 3))

        labels = [
            "Maintenance",
            "Fat Loss"
        ]

        values = [
            predicted_calories,
            final_calories
        ]

        bars = ax.barh(
            labels,
            values,
            color=[
                CHART_PALETTE[1],
                CHART_PALETTE[4]
            ]
        )

        for bar, val in zip(bars, values):

            ax.text(
                val + 20,
                bar.get_y() + bar.get_height()/2,
                f"{val:,.0f} kcal",
                va="center",
                fontweight="bold"
            )

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        # =====================================
        # RECOMMENDED MEALS
        # =====================================
        section_divider("Recommended Meals")

        recommended_meals = food_df[
            (food_df["Caloric Value"] <= final_calories / 4)
            & (food_df["Protein"] >= 10)
            & (food_df["Sugars"] <= 10)
        ].sort_values(
            by=["Nutrition Density", "Protein"],
            ascending=False
        )

        if recommended_meals.empty:

            st.info(
                "No meals match the selected criteria."
            )

        else:

            for idx, (_, row) in enumerate(
                recommended_meals.head(6).iterrows(),
                start=1
            ):

                render_meal_card(row, idx)

        # =====================================
        # HEALTH INSIGHTS
        # =====================================
        section_divider("AI Health Insights")

        render_health_insights(
            bmi=bmi,
            fat_pct=fat_percentage,
            final_cal=final_calories
        )

        # =====================================
        # MACRO PIE CHART
        # =====================================
        if not recommended_meals.empty:

            top = recommended_meals.iloc[0]

            section_divider("Top Meal Macro Breakdown")

            fig2, ax2 = plt.subplots(
                figsize=(4.5, 4.5)
            )

            macro_vals = [
                top["Protein"],
                top["Carbohydrates"],
                top["Fat"],
                top["Dietary Fiber"]
            ]

            macro_labels = [
                "Protein",
                "Carbs",
                "Fat",
                "Fiber"
            ]

            ax2.pie(
                macro_vals,
                labels=macro_labels,
                autopct="%1.1f%%",
                colors=CHART_PALETTE[:4],
                startangle=90,
                wedgeprops={
                    "edgecolor": "white",
                    "linewidth": 2
                }
            )

            plt.tight_layout()

            st.pyplot(fig2)
