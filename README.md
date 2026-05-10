# 🥗 Fat Loss Diet Recommendation System (Streamlit ML App)

A machine learning web application that predicts **maintenance calories** and provides **personalized fat-loss meal recommendations** based on user body metrics.

Built with **Streamlit**, **Scikit-learn**, **Pandas**, **Matplotlib**, and **Seaborn**, this project implements a full ML pipeline: data exploration, preprocessing, model training, evaluation, and recommendation system.

---

# 🚀 Features

## 🏠 Home Dashboard
- Dataset overview
- Key statistics (BMI, calories, features)
- Feature distribution visualization
- Workflow explanation

---

## 📊 Exploratory Data Analysis (EDA)
- Dataset preview
- Statistical summary
- Histogram distributions
- Correlation heatmap
- Feature vs Calories regression plot

---

## ⚙️ Data Preprocessing
- Train-test split configuration
- Scaler selection:
  - StandardScaler
  - MinMaxScaler
- Scaled dataset preview
- Split summary visualization

---

## 🤖 Machine Learning Models
Supported models:
- Linear Regression
- Random Forest Regressor
- Support Vector Regression (SVR)

Evaluation metrics:
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Additional:
- Model comparison dashboard
- Best model selection
- Save model using pickle (`.pkl`)

---

## 🍽 Diet Recommendation System
User inputs:
- Age
- Weight
- Height
- BMI
- Body Fat Percentage

Outputs:
- Maintenance calorie prediction
- Fat-loss calorie target
- Personalized meal recommendations

Filtering rules:
- Calories ≤ predicted target / 4
- Protein ≥ 10g
- Sugar ≤ 10g

Includes:
- Macro nutrient visualization (pie chart)
- Meal ranking based on nutrition density

---

# 🧠 Machine Learning Pipeline

## Dataset
- `Final_data.csv`

## Features
- Age
- Weight (kg)
- Height (m)
- BMI
- Fat_Percentage

## Target
- Calories

## Steps
1. Load dataset
2. Feature selection
3. Train-test split
4. Scaling (StandardScaler / MinMaxScaler)
5. Model training
6. Evaluation
7. Model saving (`MainModel.pkl`)

---

# 📁 Project Structure
project/
│── app.py
│── Final_data.csv
│── FOOD-DATA-GROUP5.csv
│── requirements.txt
│── MainModel.pkl
│── scaler.pkl
│── README.md


---

📊 Evaluation Metrics
R² Score → Model fit quality
MAE → Average absolute error
MSE → Squared error sensitivity
RMSE → Error magnitude in original unit
🍽 Recommendation Logic

Food filtering rules:

Caloric Value ≤ (Predicted Calories / 4)
Protein ≥ 10g
Sugar ≤ 10g

Ranking priority:

Nutrition Density
Protein content
⚠️ Limitations
1. Dataset Limitations
Limited features (no activity level, gender, or lifestyle data)
No real BMR/TDEE calculation
2. Model Limitations
Only basic ML models used
No hyperparameter tuning
No ensemble optimization
3. Recommendation System
Rule-based only (not AI recommender system)
No personalization (diet type, allergies, preferences)
4. Medical Validity
Predictions are statistical only
Not a substitute for nutritionist or medical advice
5. Deployment
No API backend (only Streamlit UI)
Model stored using pickle (no version control)

🚀 Future Improvements
🧬 Feature Engineering
Add activity level (sedentary, active, athlete)
Add gender-based metabolic adjustment
Add TDEE/BMR calculation
🤖 Model Improvements
Use XGBoost / LightGBM
Add Neural Network regression
Hyperparameter tuning (GridSearchCV / Optuna)
Cross-validation improvements
🍽 Recommendation System Upgrade
Content-based filtering
Hybrid recommender system
User preference learning
Diet modes:
Vegan
Vegetarian
Allergy-safe
Muscle gain / fat loss modes
📊 Explainability
SHAP / LIME feature importance
Residual error analysis
Confidence interval prediction
🌐 Deployment
Streamlit Cloud / HuggingFace Spaces
FastAPI backend integration
Docker containerization
MLflow model tracking
📱 UI Enhancements
User login system
Save recommendation history
Progress tracking dashboard
Mobile-responsive UI improvements

👨‍💻 Author
Machine Learning & Data Science project built with Streamlit.

📜 License
This project is for educational purposes only.
