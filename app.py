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
# CUSTOM CSS (UI ENHANCEMENTS ONLY)
# =====================================
st.markdown("""
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.3);
    transform: translateX(5px);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #1B4332;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border: 1px solid #95D5B2;
}
div[data-testid="stMetric"] label {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] .stMetricValue {
    font-size: 2rem !important;
}

/* Main headers */
h1, h2, h3 {
    color: #2D6A4F;
    font-weight: 600;
}
h1 {
    border-left: 6px solid #2D6A4F;
    padding-left: 20px;
}

/* Buttons */
.stButton button {
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton button:hover {
    transform: scale(1.02);
}

/* Expander for meals */
.streamlit-expanderHeader {
    font-size: 1.2rem;
    font-weight: 600;
    background: #F0F9F0;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# SESSION STATE (UNCHANGED)
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
# LOAD DATA (UNCHANGED)
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
# HELPER FUNCTIONS (UNCHANGED)
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
# SIDEBAR NAVIGATION (UI ENHANCED BUTTONS)
# =====================================
with st.sidebar:
    st.title("🥗 Fat Loss App")
    st.markdown("### ML Diet Recommendation")
    st.markdown("---")
    
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
    
    st.markdown("---")
    st.caption("v1.0 | Powered by Scikit-learn")

# =====================================
# HOME (UI ENHANCED)
# =====================================
if st.session_state.page == "Home":
    st.title("🥗 Fat Loss Diet Recommendation System")
    st.markdown("Predict your maintenance calories and get science‑based fat‑loss meal suggestions.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Records", len(df))
    with col2:
        st.metric("🔥 Avg. Maintenance Calories", round(df[target].mean(), 2))
    with col3:
        st.metric("⚖️ Avg. BMI", round(df["BMI"].mean(), 2))
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# =====================================
# EDA (UNCHANGED LOGIC, UI TABS KEPT)
# =====================================
elif st.session_state.page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset", "Histogram", "Heatmap", "Scatterplot"
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
# PREPROCESSING (UNCHANGED LOGIC, UI IMPROVED LAYOUT)
# =====================================
elif st.session_state.page == "Preprocessing":
    st.title("⚙️ Data Preprocessing")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider(
            "Test Size (%)",
            10, 40, 20
        )
    with col2:
        random_state = st.number_input(
            "Random State",
            1, 999, 42
        )
    
    scaler_option = st.radio(
        "Choose Scaler",
        ["StandardScaler", "MinMaxScaler"],
        horizontal=True
    )
    
    if st.button("🚀 Run Preprocessing", type="primary", use_container_width=False):
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=random_state
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
        
        st.success("✅ Preprocessing completed successfully!")
        
        st.subheader("Scaled Training Data (first 5 rows)")
        st.dataframe(
            pd.DataFrame(X_train_scaled, columns=features).head(),
            use_container_width=True
        )

# =====================================
# MODEL (UNCHANGED LOGIC, UI ENHANCED WITH STATUS)
# =====================================
elif st.session_state.page == "Model":
    st.title("🤖 Model Training & Evaluation")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Please run preprocessing first.")
        st.stop()
    
    selected_models = st.multiselect(
        "Select Models",
        ["Linear Regression", "Random Forest", "SVR"],
        default=["Linear Regression"]
    )
    
    if st.button("🏋️ Train & Evaluate", type="primary"):
        results = {}
        trained_models = {}
        
        # UI: modern status container
        with st.status("Training models...", expanded=True) as status:
            for i, model_name in enumerate(selected_models):
                status.update(label=f"Training {model_name}...")
                
                if model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Random Forest":
                    model = RandomForestRegressor()
                else:
                    model = SVR()
                
                model.fit(st.session_state.X_train, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test)
                
                results[model_name] = {
                    "R2": r2_score(st.session_state.y_test, y_pred),
                    "MAE": mean_absolute_error(st.session_state.y_test, y_pred),
                    "MSE": mean_squared_error(st.session_state.y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
                }
                trained_models[model_name] = model
            status.update(label="Training complete!", state="complete")
        
        st.session_state.models = trained_models
        st.session_state.eval_results = results
    
    if st.session_state.eval_results:
        results_df = pd.DataFrame(st.session_state.eval_results).T
        st.subheader("📈 Evaluation Results")
        st.dataframe(results_df, use_container_width=True)
        
        save_model = st.selectbox(
            "💾 Save Which Model?",
            list(st.session_state.models.keys())
        )
        
        if st.button("Save Model"):
            save_pickle(st.session_state.models[save_model], "MainModel.pkl")
            save_pickle(st.session_state.scaler, "scaler.pkl")
            st.success("🎉 Model and scaler saved as `MainModel.pkl` and `scaler.pkl`")

# =====================================
# PREDICTION / RECOMMENDATION (UNCHANGED LOGIC, UI ENHANCED WITH EXPANDERS)
# =====================================
elif st.session_state.page == "Prediction":
    st.title("🍽 Diet Recommendation")
    
    model_files = list_saved_pickles()
    
    if "MainModel.pkl" not in model_files:
        st.warning("⚠️ Train and save a model first.")
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
        fat_percentage = st.number_input("Fat Percentage", 5.0, 50.0, 20.0)
        target_fat_loss = st.slider("Target Fat Loss (%)", 1, 20, 5)
    
    if st.button("✨ Get Recommendation", type="primary"):
        input_data = np.array([[
            age, weight, height, bmi, fat_percentage
        ]])
        input_scaled = scaler.transform(input_data)
        predicted_calories = model.predict(input_scaled)[0]
        
        calorie_deficit = predicted_calories * (target_fat_loss / 100)
        final_calories = predicted_calories - calorie_deficit
        category = bmi_category(bmi)
        
        # Metric cards with icons
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🔥 Maintenance Calories", f"{predicted_calories:.0f} kcal")
        col_b.metric("🎯 Fat Loss Calories", f"{final_calories:.0f} kcal", delta=f"-{calorie_deficit:.0f} kcal")
        col_c.metric("📊 BMI Category", category)
        
        # Simple bar chart (UI only)
        chart_df = pd.DataFrame({
            "Type": ["Maintenance", "Fat Loss"],
            "Calories": [predicted_calories, final_calories]
        })
        st.bar_chart(chart_df.set_index("Type"), use_container_width=True)
        
        # Recommended meals (original logic)
        recommended_meals = food_df[
            (food_df["Caloric Value"] <= final_calories / 4) &
            (food_df["Protein"] >= 10) &
            (food_df["Sugars"] <= 10)
        ].sort_values(
            by=["Nutrition Density", "Protein"],
            ascending=False
        )
        
        st.subheader("🍴 Recommended Meals")
        if recommended_meals.empty:
            st.info("No meals match your criteria. Try adjusting your fat loss target.")
        else:
            for idx, row in recommended_meals.head(5).iterrows():
                with st.expander(f"🍽️ {row['food']} – {row['Caloric Value']} kcal"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Protein:** {row['Protein']} g  \n**Carbs:** {row['Carbohydrates']} g")
                    c2.markdown(f"**Fat:** {row['Fat']} g  \n**Sugar:** {row['Sugars']} g  \n**Fiber:** {row['Dietary Fiber']} g")
