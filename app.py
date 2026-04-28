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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Fat Loss Diet Recommendation",
    page_icon="🥗",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #1B4332;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

div[data-testid="stMetric"] {
    background: #1B4332;
    border: 2px solid #95D5B2;
    border-radius: 12px;
    padding: 15px;
}

.stButton>button {
    border-radius: 10px;
    font-weight: 600;
}

h1, h2, h3 {
    color: #2D6A4F;
}
</style>
""", unsafe_allow_html=True)

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
# HELPER
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


# =====================================
# SIDEBAR NAVIGATION
# =====================================
with st.sidebar:
    st.title("🥗 Fat Loss App")
    st.markdown("Machine Learning Diet Recommendation")

    pages = {
        "🏠 Home": "Home",
        "📊 EDA": "EDA",
        "⚙️ Preprocessing": "Preprocessing",
        "🤖 Model": "Model",
        "🍽 Recommendation": "Prediction"
    }

    for label, page in pages.items():
        if st.button(label, use_container_width=True):
            st.session_state.page = page
            st.rerun()


# =====================================
# HOME
# =====================================
if st.session_state.page == "Home":
    st.title("🥗 Fat Loss Diet Recommendation System")

    st.write(
        "Predict maintenance calories and generate fat-loss meal recommendations."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(df))
    c2.metric("Average Calories", round(df[target].mean(), 2))
    c3.metric("Average BMI", round(df["BMI"].mean(), 2))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)


# =====================================
# EDA
# =====================================
elif st.session_state.page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset",
        "Histogram",
        "Heatmap",
        "Scatterplot"
    ])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        selected_features = st.multiselect(
            "Select Features",
            features,
            default=features[:2]
        )

        for feature in selected_features:
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df[features + [target]].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)

    with tab4:
        selected_feature = st.selectbox(
            "Select Feature",
            features
        )

        fig, ax = plt.subplots()
        sns.regplot(
            x=df[selected_feature],
            y=df[target],
            ax=ax
        )
        st.pyplot(fig)


# =====================================
# PREPROCESSING
# =====================================
elif st.session_state.page == "Preprocessing":
    st.title("⚙️ Data Preprocessing")

    c1, c2 = st.columns(2)

    with c1:
        test_size = st.slider(
            "Test Size (%)",
            10,
            40,
            20
        )

    with c2:
        random_state = st.number_input(
            "Random State",
            1,
            999,
            42
        )

    scaler_option = st.radio(
        "Choose Scaler",
        [
            "StandardScaler",
            "MinMaxScaler"
        ]
    )

    if st.button("Run Preprocessing"):

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=random_state
        )

        if scaler_option == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state.X_train = X_train_scaled
        st.session_state.X_test = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.scaler_name = scaler_option

        st.success("Preprocessing Completed!")

        st.dataframe(
            pd.DataFrame(
                X_train_scaled,
                columns=features
            ).head()
        )


# =====================================
# MODEL
# =====================================
elif st.session_state.page == "Model":
    st.title("🤖 Model Training & Evaluation")

    if st.session_state.X_train is None:
        st.warning("Run preprocessing first.")
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

    if st.button("Train & Evaluate"):

        results = {}
        trained_models = {}

        progress = st.progress(0)

        for i, model_name in enumerate(selected_models):

            progress.progress((i + 1) / len(selected_models))

            if model_name == "Linear Regression":
                model = LinearRegression()

            elif model_name == "Random Forest":
                model = RandomForestRegressor()

            else:
                model = SVR()

            model.fit(
                st.session_state.X_train,
                st.session_state.y_train
            )

            y_pred = model.predict(
                st.session_state.X_test
            )

            results[model_name] = {
                "R2": r2_score(
                    st.session_state.y_test,
                    y_pred
                ),
                "MAE": mean_absolute_error(
                    st.session_state.y_test,
                    y_pred
                ),
                "MSE": mean_squared_error(
                    st.session_state.y_test,
                    y_pred
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        st.session_state.y_test,
                        y_pred
                    )
                )
            }

            trained_models[model_name] = model

        st.session_state.models = trained_models
        st.session_state.eval_results = results

    if st.session_state.eval_results:

        results_df = pd.DataFrame(
            st.session_state.eval_results
        ).T

        st.subheader("Evaluation Results")
        st.dataframe(results_df)

        save_model = st.selectbox(
            "Save Which Model?",
            list(st.session_state.models.keys())
        )

        if st.button("Save Model"):
            save_pickle(
                st.session_state.models[save_model],
                "MainModel.pkl"
            )

            save_pickle(
                st.session_state.scaler,
                "scaler.pkl"
            )

            st.success("Model and Scaler Saved!")


# =====================================
# PREDICTION
# =====================================
elif st.session_state.page == "Prediction":
    st.title("🍽 Diet Recommendation")

    model_files = list_saved_pickles()

    if "MainModel.pkl" not in model_files:
        st.warning("Train and save model first.")
        st.stop()

    with open("MainModel.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 15, 70, 25)
        weight = st.number_input("Weight (kg)", 30, 200, 75)
        height = st.number_input("Height (m)", 1.0, 2.5, 1.75)

    with col2:
        bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
        fat_percentage = st.number_input(
            "Fat Percentage",
            5.0,
            50.0,
            20.0
        )
        target_fat_loss = st.slider(
            "Target Fat Loss (%)",
            1,
            20,
            5
        )

    if st.button("Get Recommendation"):

        input_data = np.array([[
            age,
            weight,
            height,
            bmi,
            fat_percentage
        ]])

        input_scaled = scaler.transform(
            input_data
        )

        predicted_calories = model.predict(
            input_scaled
        )[0]

        calorie_deficit = predicted_calories * (
            target_fat_loss / 100
        )

        final_calories = (
            predicted_calories - calorie_deficit
        )

        category = bmi_category(bmi)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Maintenance Calories",
            round(predicted_calories, 2)
        )

        c2.metric(
            "Fat Loss Calories",
            round(final_calories, 2)
        )

        c3.metric(
            "BMI Category",
            category
        )

        chart_df = pd.DataFrame({
            "Type": [
                "Maintenance",
                "Fat Loss"
            ],
            "Calories": [
                predicted_calories,
                final_calories
            ]
        })


        st.subheader("Recommended Meals")

        recommended_meals = food_df[
            (food_df["Caloric Value"] <= final_calories / 4) &
            (food_df["Protein"] >= 10) &
            (food_df["Sugars"] <= 10)
        ].sort_values(
            by=["Nutrition Density", "Protein"],
            ascending=False
        )

        for _, row in recommended_meals.head(5).iterrows():
            st.markdown(f"""
            ### 🍽 {row["food"]}
            Calories: {row["Caloric Value"]} kcal  
            Protein: {row["Protein"]} g  
            Carbs: {row["Carbohydrates"]} g  
            Fat: {row["Fat"]} g  
            Sugar: {row["Sugars"]} g  
            Fiber: {row["Dietary Fiber"]} g  
            """)
