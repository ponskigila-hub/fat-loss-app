# 🥗 Fat Loss Diet Recommendation System (Streamlit ML App)

A machine learning web application that predicts **maintenance calories** and provides **personalized fat-loss meal recommendations** based on user body metrics.

Built with **Streamlit**, **Scikit-learn**, **Pandas**, **Matplotlib**, and **Seaborn**, this project implements a full ML pipeline: data exploration, preprocessing, model training, evaluation, and recommendation system.

---

## 🌐 Deployment
**Live Demo:** [https://fat-loss-app-n.streamlit.app/](https://fat-loss-app-n.streamlit.app/)

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results & Metrics](#results--metrics)
- [How to Run](#how-to-run)
- [Features](#features)
- [Recommendation Logic](#recommendation-logic)
- [Project Structure](#project-structure)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## 🎯 Problem Statement
Achieving a caloric deficit for weight loss often requires tedious manual calculations and generic meal planning that ignores individual body compositions. This project solves that problem by leveraging Machine Learning to predict personalized maintenance calories based on specific body metrics (Age, Weight, Height, BMI, Body Fat Percentage). Furthermore, it acts as a dietary assistant by instantly filtering and recommending high-protein, low-sugar meals tailored to the user's predicted caloric deficit target.

---

## 📂 Datasets
This application relies on two primary datasets:
1. **`Final_data.csv` (User Metrics):** Used for training the regression models. 
   - *Features:* `Age`, `Weight (kg)`, `Height (m)`, `BMI`, `Fat_Percentage`
   - *Target:* `Calories` (Maintenance caloric need)
2. **`FOOD-DATA-GROUP5.csv` (Nutritional Database):** Used for the rule-based recommendation engine. Contains meal names, caloric values, macronutrients (Protein, Fat, Carbs), and sugar content.

---

## 🧠 Methodology

### 1. Data Preprocessing & EDA
- **Exploratory Data Analysis:** Visualizing feature distributions, correlation heatmaps, and feature-to-target regression plots.
- **Data Scaling:** Offering user-selectable scaling methods (`StandardScaler` or `MinMaxScaler`) to normalize body metrics before feeding them into the models.
- **Data Splitting:** Configurable Train-Test split ratios.

### 2. Machine Learning Pipeline
Training regression models to predict maintenance calories:
- **Linear Regression**
- **Random Forest Regressor**
- **Support Vector Regression (SVR)**

### 3. Recommendation System
A rule-based filtering algorithm that takes the predicted maintenance calories, calculates a fat-loss deficit, and queries the food database for meals matching strict nutritional constraints.

---

## 📊 Results & Metrics
Models are evaluated in real-time based on the user's preprocessing configuration. The evaluation metrics used are:
- **R² Score:** Evaluates how well the model fits the variance of the calorie data.
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual calories.
- **MSE (Mean Squared Error):** Penalizes larger prediction errors.
- **RMSE (Root Mean Squared Error):** Represents the error magnitude in original unit (Calories).

The best-performing model is saved as `MainModel.pkl` along with its `scaler.pkl` for immediate inference in the recommendation engine.

---

## 💻 How to Run

**Prerequisites:** Python 3.8+

**1. Clone the repository and navigate to the directory**
```bash
git clone <your-repo-link>
cd <project-folder>
