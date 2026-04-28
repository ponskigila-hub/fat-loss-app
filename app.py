import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Fat Loss Diet Recommendation",
    page_icon="🥗",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_main_data():
    return pd.read_csv("Final_data.csv")

@st.cache_data
def load_food_data():
    return pd.read_csv("Food_Nutrition_Dataset.csv")

df = load_main_data()
food_df = load_food_data()

# ==========================
# FEATURES
# ==========================
features = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "BMI",
    "Fat_Percentage"
]

target = "Calories"

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Choose Page",
    [
        "Home",
        "EDA",
        "Preprocessing",
        "Train Model",
        "Evaluation",
        "Diet Recommendation"
    ]
)

# ==========================
# HOME
# ==========================
if menu == "Home":
    st.title("🥗 Personalized Fat Loss Diet Recommendation")

    st.write("""
    Predict recommended daily calorie intake
    and get personalized food recommendations.
    """)

    st.subheader("Lifestyle Dataset Preview")
    st.dataframe(df.head())

# ==========================
# EDA
# ==========================
elif menu == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Dataset Head")
    st.dataframe(df.head())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Null Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    selected_feature = st.selectbox(
        "Select Feature",
        features
    )

    chart_type = st.selectbox(
        "Select Chart",
        ["Histogram", "Boxplot", "Scatterplot"]
    )

    if chart_type == "Histogram":
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        st.pyplot(fig)

    elif chart_type == "Boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_feature], ax=ax)
        st.pyplot(fig)

    elif chart_type == "Scatterplot":
        fig, ax = plt.subplots()
        sns.regplot(
            x=df[selected_feature],
            y=df[target],
            ax=ax
        )
        st.pyplot(fig)

# ==========================
# PREPROCESSING
# ==========================
elif menu == "Preprocessing":
    st.title("Data Preprocessing")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    st.success("Preprocessing Completed")

# ==========================
# TRAIN MODEL
# ==========================
elif menu == "Train Model":
    st.title("Train Your Model")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    if st.button("Train Model"):
        model = RandomForestRegressor()
        model.fit(X_train_scaled, y_train)

        with open("MainModel.pkl", "wb") as file:
            pickle.dump(model, file)

        with open("scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)

        st.success("Model Saved Successfully")

# ==========================
# EVALUATION
# ==========================
elif menu == "Evaluation":
    st.title("Model Evaluation")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open("MainModel.pkl", "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test_scaled)

    st.write("R2:", r2_score(y_test, y_pred))
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ==========================
# DIET RECOMMENDATION
# ==========================
elif menu == "Diet Recommendation":
    st.title("Diet Recommendation Demo")

    age = st.number_input("Age", 15, 70, 25)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.75)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
    fat_percentage = st.number_input("Fat Percentage", 5.0, 50.0, 20.0)

    if st.button("Get Recommendation"):
        with open("MainModel.pkl", "rb") as file:
            model = pickle.load(file)

        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)

        user_data = np.array([[
            age,
            weight,
            height,
            bmi,
            fat_percentage
        ]])

        user_scaled = scaler.transform(user_data)

        predicted_calories = model.predict(user_scaled)[0]

        st.success(
            f"Recommended Daily Intake: {round(predicted_calories,2)} kcal"
        )

        st.subheader("Recommended Foods")

        recommended_foods = food_df[
            (food_df["calories"] <= predicted_calories/4) &
            (food_df["protein"] >= 10)
        ].sort_values(
            by="protein",
            ascending=False
        )

        st.dataframe(
            recommended_foods[
                [
                    "food_name",
                    "category",
                    "calories",
                    "protein",
                    "carbs",
                    "fat"
                ]
            ].head(10)
        )
