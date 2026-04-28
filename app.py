import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
    page_title='Fat Loss Diet Recommendation',
    page_icon='🥗',
    layout='wide'
)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv('Final_data.csv')

df = load_data()

# =====================================
# FEATURES & TARGET
# =====================================
features = [
    'Age',
    'Weight (kg)',
    'Height (m)',
    'BMI',
    'Fat_Percentage',
    'Session_Duration (hours)',
    'Workout_Frequency (days/week)',
    'Calories_Burned'
]

target = 'Calories'

# =====================================
# SIDEBAR MENU
# =====================================
st.sidebar.title('Navigation')
menu = st.sidebar.selectbox(
    'Choose Page',
    [
        'Home',
        'Exploratory Data Analysis',
        'Data Preprocessing',
        'Train Your Model',
        'Model Evaluation',
        'Diet Recommendation Demo'
    ]
)

# =====================================
# HOME
# =====================================
if menu == 'Home':
    st.title('🥗 Personalized Fat Loss Diet Recommendation System')
    st.write('Predict your recommended calorie intake and get healthy meal recommendations.')

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Records', len(df))
    col2.metric('Average Calories', round(df[target].mean(), 2))
    col3.metric('Average Calories Burned', round(df['Calories_Burned'].mean(), 2))

    st.subheader('Dataset Preview')
    st.dataframe(df.head())

# =====================================
# EDA
# =====================================
elif menu == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')

    st.subheader('Dataset Head')
    st.dataframe(df.head())

    st.subheader('Data Types')
    st.write(df.dtypes)

    if st.checkbox('Show Null Values'):
        st.write(df.isnull().sum())

    st.subheader('Statistical Summary')
    st.write(df.describe())

    chart_option = st.selectbox(
        'Select Visualization',
        ['Histogram', 'Heatmap', 'Scatterplot', 'Boxplot']
    )

    selected_feature = st.selectbox('Select Feature', features)

    if chart_option == 'Histogram':
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        st.pyplot(fig)

    elif chart_option == 'Heatmap':
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif chart_option == 'Scatterplot':
        fig, ax = plt.subplots()
        sns.regplot(x=df[selected_feature], y=df[target], ax=ax)
        st.pyplot(fig)

    elif chart_option == 'Boxplot':
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_feature], ax=ax)
        st.pyplot(fig)

# =====================================
# PREPROCESSING
# =====================================
elif menu == 'Data Preprocessing':
    st.title('Data Preprocessing')

    test_size = st.slider('Test Size', 0.1, 0.5, 0.2)
    random_state = st.slider('Random State', 1, 100, 42)

    scaler_option = st.selectbox(
        'Select Scaler',
        ['StandardScaler', 'MinMaxScaler']
    )

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    if scaler_option == 'StandardScaler':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    st.success('Preprocessing completed')
    st.write('Train shape:', X_train_scaled.shape)
    st.write('Test shape:', X_test_scaled.shape)

# =====================================
# TRAIN MODEL
# =====================================
elif menu == 'Train Your Model':
    st.title('Train Your Model')

    model_option = st.selectbox(
        'Select Model',
        [
            'Linear Regression (Baseline)',
            'Random Forest Regressor (Proposed)',
            'SVR (Alternative)'
        ]
    )

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if st.button('Train Model'):
        if 'Linear Regression' in model_option:
            model = LinearRegression()
        elif 'Random Forest' in model_option:
            model = RandomForestRegressor()
        else:
            model = SVR()

        model.fit(X_train, y_train)

        with open('MainModel.pkl', 'wb') as file:
            pickle.dump(model, file)

        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        st.success('Model trained and saved successfully!')

# =====================================
# MODEL EVALUATION
# =====================================
elif menu == 'Model Evaluation':
    st.title('Model Evaluation')

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open('MainModel.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    st.subheader('Evaluation Metrics')
    st.write('R2 Score:', r2_score(y_test, y_pred))
    st.write('MAE:', mean_absolute_error(y_test, y_pred))
    st.write('MSE:', mean_squared_error(y_test, y_pred))
    st.write('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# =====================================
# DIET RECOMMENDATION DEMO
# =====================================
elif menu == 'Diet Recommendation Demo':
    st.title('Diet Recommendation Demo')

    age = st.number_input('Age', 15, 70, 25)
    weight = st.number_input('Weight (kg)', 30, 200, 75)
    height = st.number_input('Height (m)', 1.0, 2.5, 1.75)
    bmi = st.number_input('BMI', 10.0, 50.0, 24.0)
    fat_percentage = st.number_input('Fat Percentage', 5.0, 50.0, 20.0)
    session_duration = st.number_input('Session Duration (hours)', 0.5, 5.0, 1.5)
    workout_frequency = st.number_input('Workout Frequency (days/week)', 1, 7, 5)
    calories_burned = st.number_input('Calories Burned', 500, 5000, 2200)

    input_data = np.array([[
        age,
        weight,
        height,
        bmi,
        fat_percentage,
        session_duration,
        workout_frequency,
        calories_burned
    ]])

    if st.button('Get Recommendation'):
        with open('MainModel.pkl', 'rb') as file:
            model = pickle.load(file)

        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        input_scaled = scaler.transform(input_data)
        predicted_calories = model.predict(input_scaled)[0]

        st.success(f'Recommended Daily Calorie Intake: {round(predicted_calories, 2)} kcal')

        st.subheader('Recommended Diet Menu')

        recommended_meals = df[
            (df['Calories'] <= predicted_calories) &
            (df['Proteins'] >= 20)
        ]

        st.dataframe(
            recommended_meals[
                [
                    'meal_name',
                    'meal_type',
                    'Calories',
                    'Proteins',
                    'Carbs',
                    'Fats'
                ]
            ].head(10)
        )
