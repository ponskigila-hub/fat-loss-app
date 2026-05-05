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
# SESSION STATE INIT
# =====================================
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "df" not in st.session_state:
    st.session_state.df = None
if "food_df" not in st.session_state:
    st.session_state.food_df = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "scaler_name" not in st.session_state:
    st.session_state.scaler_name = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# =====================================
# DATA LOADING (with error handling)
# =====================================
@st.cache_data
def load_main_data():
    try:
        df = pd.read_csv("Final_data.csv")
        return df
    except FileNotFoundError:
        st.error("❌ 'Final_data.csv' not found. Please place it in the same directory.")
        return pd.DataFrame()

@st.cache_data
def load_food_data():
    try:
        food_df = pd.read_csv("FOOD-DATA-GROUP5.csv")
        return food_df
    except FileNotFoundError:
        st.error("❌ 'FOOD-DATA-GROUP5.csv' not found. Please place it in the same directory.")
        return pd.DataFrame()

# Load if not already in session
if st.session_state.df is None:
    st.session_state.df = load_main_data()
if st.session_state.food_df is None:
    st.session_state.food_df = load_food_data()

df = st.session_state.df
food_df = st.session_state.food_df

# Stop if data missing
if df.empty or food_df.empty:
    st.stop()

features = ["Age", "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage"]
target = "Calories"

# =====================================
# HELPER FUNCTIONS
# =====================================
def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def list_saved_models():
    return [f for f in os.listdir(".") if f.endswith(".pkl") and f != "scaler.pkl"]

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def recommend_daily_plan(food_df, calorie_target, meals=3):
    """Return a list of recommended foods for a day (up to `meals` items)."""
    # Filter healthy options
    healthy = food_df[
        (food_df["Caloric Value"] <= calorie_target * 0.6) &
        (food_df["Protein"] >= 8) &
        (food_df["Sugars"] <= 12)
    ].copy()
    if healthy.empty:
        return [], calorie_target
    # Prioritise nutrition density and protein
    healthy["score"] = healthy["Nutrition Density"] + healthy["Protein"] / 10
    healthy.sort_values("score", ascending=False, inplace=True)
    
    plan = []
    remaining = calorie_target
    for _ in range(meals):
        candidates = healthy[healthy["Caloric Value"] <= remaining * 0.9]
        if not candidates.empty:
            chosen = candidates.iloc[0]
            plan.append(chosen)
            remaining -= chosen["Caloric Value"]
        else:
            break
    return plan, remaining

def plot_bmi_gauge(bmi):
    fig, ax = plt.subplots(figsize=(6,1))
    colors = ["#6c9ebf", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    ranges = [0, 18.5, 25, 30, 40]
    for i in range(len(ranges)-1):
        ax.barh(0, ranges[i+1]-ranges[i], left=ranges[i], color=colors[i], height=0.5)
    ax.scatter([bmi], [0], color="red", s=120, zorder=10, edgecolors="black")
    ax.set_xlim(10, 45)
    ax.set_yticks([])
    ax.set_xticks([18.5, 25, 30])
    ax.set_xticklabels(["Underweight", "Normal", "Overweight"])
    ax.set_title("BMI Visualisation")
    return fig

# =====================================
# SIDEBAR NAVIGATION (Radio)
# =====================================
with st.sidebar:
    st.title("🥗 Fat Loss App")
    st.markdown("Machine Learning Diet Recommendation")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 EDA", "⚙️ Preprocessing", "🤖 Model", "🍽 Recommendation"],
        index=["Home", "EDA", "Preprocessing", "Model", "Prediction"].index(st.session_state.page)
    )
    st.session_state.page = page
    
    if st.button("🔄 Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# =====================================
# HOME
# =====================================
if st.session_state.page == "🏠 Home":
    st.title("🥗 Fat Loss Diet Recommendation System")
    st.write("Predict your maintenance calories and get a personalised fat-loss meal plan.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Calories", round(df[target].mean(), 2))
    col3.metric("Average BMI", round(df["BMI"].mean(), 2))
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# =====================================
# EDA
# =====================================
elif st.session_state.page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Histograms", "Heatmap", "Scatter"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
    
    with tab2:
        selected = st.multiselect("Select features", features, default=features[:2])
        for feat in selected:
            fig, ax = plt.subplots()
            sns.histplot(df[feat], kde=True, ax=ax)
            st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    with tab4:
        feat = st.selectbox("Feature vs Calories", features)
        fig, ax = plt.subplots()
        sns.regplot(x=df[feat], y=df[target], ax=ax)
        st.pyplot(fig)

# =====================================
# PREPROCESSING
# =====================================
elif st.session_state.page == "⚙️ Preprocessing":
    st.title("⚙️ Data Preprocessing")
    
    with st.popover("⚙️ Advanced split settings"):
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        random_state = st.number_input("Random State", 1, 999, 42)
    
    scaler_option = st.radio("Choose Scaler", ["StandardScaler", "MinMaxScaler"])
    
    if st.button("Run Preprocessing", type="primary"):
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
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
        
        st.success("Preprocessing completed!")
        st.write("Scaled training data (first 5 rows):")
        st.dataframe(pd.DataFrame(X_train_scaled, columns=features).head())
        st.write("Descriptive statistics of scaled features:")
        st.dataframe(pd.DataFrame(X_train_scaled, columns=features).describe())

# =====================================
# MODEL TRAINING
# =====================================
elif st.session_state.page == "🤖 Model":
    st.title("🤖 Model Training & Evaluation")
    
    if st.session_state.X_train is None:
        st.warning("Please run preprocessing first.")
        st.stop()
    
    selected_models = st.multiselect(
        "Select models to train",
        ["Linear Regression", "Random Forest", "SVR"],
        default=["Linear Regression"]
    )
    
    if st.button("Train & Evaluate", type="primary"):
        results = {}
        trained_models = {}
        
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
        st.subheader("Evaluation Results")
        st.dataframe(results_df)
        
        # Save model
        save_name = st.selectbox("Select model to save", list(st.session_state.models.keys()))
        model_to_save = st.session_state.models[save_name]
        if st.button("Save Model"):
            save_pickle(model_to_save, "MainModel.pkl")
            if st.session_state.scaler:
                save_pickle(st.session_state.scaler, "scaler.pkl")
            st.success("Model and scaler saved as 'MainModel.pkl' and 'scaler.pkl'")

# =====================================
# RECOMMENDATION (PREDICTION)
# =====================================
elif st.session_state.page == "🍽 Recommendation":
    st.title("🍽 Personalised Diet Recommendation")
    
    # Load model and scaler
    model_files = list_saved_models()
    if "MainModel.pkl" not in model_files:
        st.warning("No trained model found. Please train and save a model first.")
        st.stop()
    
    try:
        with open("MainModel.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        st.stop()
    
    # Input section
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 15, 70, 25)
        weight = st.number_input("Weight (kg)", 30, 200, 75)
        height = st.number_input("Height (m)", 1.0, 2.5, 1.75)
    with col2:
        bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
        fat_percentage = st.number_input("Fat Percentage (%)", 5.0, 50.0, 20.0)
        target_fat_loss = st.slider("Target Fat Loss (%)", 1, 20, 5)
    
    if st.button("Get Recommendation", type="primary"):
        # Predict
        input_data = np.array([[age, weight, height, bmi, fat_percentage]])
        input_scaled = scaler.transform(input_data)
        maint_cal = model.predict(input_scaled)[0]
        deficit = maint_cal * (target_fat_loss / 100)
        loss_cal = maint_cal - deficit
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Maintenance Calories", f"{maint_cal:.0f} kcal")
        c2.metric("Fat Loss Calories", f"{loss_cal:.0f} kcal", delta=f"-{deficit:.0f} kcal")
        cat = bmi_category(bmi)
        c3.metric("BMI Category", cat)
        
        # BMI gauge
        st.pyplot(plot_bmi_gauge(bmi))
        
        # Chart
        chart_df = pd.DataFrame({"Type": ["Maintenance", "Fat Loss"], "Calories": [maint_cal, loss_cal]})
        st.bar_chart(chart_df.set_index("Type"))
        
        # Meal recommendation
        st.subheader("📅 Your Personalised Daily Plan")
        if food_df is not None and not food_df.empty:
            plan, leftover = recommend_daily_plan(food_df, loss_cal, meals=4)
            if not plan:
                st.info("No exact food matches found. Try reducing filters or eat smaller portions.")
            else:
                for i, meal in enumerate(plan, 1):
                    with st.expander(f"🍽️ Meal {i}: {meal['food']} – {meal['Caloric Value']:.0f} kcal"):
                        col_a, col_b = st.columns(2)
                        col_a.metric("Protein", f"{meal['Protein']} g")
                        col_a.metric("Carbohydrates", f"{meal['Carbohydrates']} g")
                        col_b.metric("Fat", f"{meal['Fat']} g")
                        col_b.metric("Fiber", f"{meal['Dietary Fiber']} g")
                        st.caption(f"Sugar: {meal['Sugars']} g")
                if leftover > 50:
                    st.info(f"ℹ️ You have {leftover:.0f} kcal remaining – add a small snack or increase portion sizes.")
                else:
                    st.success("✅ Great balance! Your meal plan fits the calorie target perfectly.")
            
            # Macro distribution from selected meals
            if plan:
                total_protein = sum(m["Protein"] for m in plan)
                total_carbs = sum(m["Carbohydrates"] for m in plan)
                total_fat = sum(m["Fat"] for m in plan)
                fig2, ax2 = plt.subplots()
                ax2.pie([total_protein, total_carbs, total_fat], labels=["Protein", "Carbs", "Fat"],
                        autopct="%1.1f%%", colors=["#2E8B57", "#FFA07A", "#CD5C5C"])
                ax2.set_title("Macronutrient Breakdown")
                st.pyplot(fig2)
        else:
            st.error("Food database not available.")

# =====================================
# END
# =====================================
