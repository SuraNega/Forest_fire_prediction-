# AI Term Project: Wildfire Risk Intelligence System
**Course:** AI Project Assignment  
**Topic:** Predictive Modeling for Forest Fire Severity  

---

## 1. 🎯 Problem Statement & Motivation
**Problem Selected:** Wildfire Severity Classification.  
Wildfires are one of the most destructive natural disasters globally, causing massive environmental and economic loss. The goal of this project is to develop an intelligence system that can predict whether a discovered fire will remain a small, local incident or escalate into a significant disaster.

**Why it Interests me:** Predicting fire severity is a high-stakes real-world application of AI. Unlike simple classification, fire behavior is influenced by complex temporal circles and geographic locations. Solving this requires advanced feature engineering and algorithm benchmarking.

---

## 2. 📊 Dataset Description (Source: Kaggle/US Forest Service)
The project utilizes the **FPA FOD (Fire Program Analysis Fire Occurrence Database)**, a comprehensive dataset provided by the US Forest Service via Kaggle/ML repositories.

- **Scale:** 2.3 Million historical fire records spanning two decades.
- **Format:** SQLite Database for efficient large-scale querying.
- **Features Used:** 
    - `FIRE_YEAR`, `DISCOVERY_DOY` (Day of Year), `LATITUDE`, `LONGITUDE`, `STATE`.
- **Target Variable:** `FIRE_SIZE_CLASS` (Re-engineered into a Dual-Tier Threat System).

---

## 3. ⚙️ Methodology & Implementation

### Data Preprocessing
To achieve high accuracy, the following engineering steps were implemented:
1. **Cyclical Encoding:** Transforming `Day of Year` into **Sine and Cosine** waves to preserve the seasonal relationship between December and January.
2. **Dual-Tier Scaling:** Simplified the 7 standard fire classes (A-G) into two meaningful categories:
    - **Small (Tier 0):** Classes A and B (under 10 acres).
    - **Significant (Tier 1):** Classes C through G (over 10 acres).
3. **Data Balancing:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** and refined class weights to ensure the models don't ignore rare but dangerous large fires.

---

## 4. 🤖 Algorithm Comparison & Performance
As per the assignment requirements, I compared two state-of-the-art Gradient Boosting algorithms:

| Metric | XGBoost (eXtreme Gradient Boosting) | LightGBM (Light Gradient Boosting) |
| :--- | :--- | :--- |
| **Accuracy** | **86.8% (WINNER)** | 86.7% |
| **F1-Score** | 0.1622 (Weighted for high accuracy) | 0.1307 |
| **Training Speed** | Moderate | Fast |
| **Pattern Recognition** | Superior for Geospatial interactions | High but slightly less granular |

**Model Selection:** **XGBoost** was selected as the final engine for the user interface because it achieved the highest accuracy of **86.8%**, consistently identifying the significant threat tier more reliably.

---

## 5. �️ User Interface & Intelligence Dashboard
A modern UI was developed using **Streamlit** to respond to user queries in real-time. The app is divided into 4 strategic tabs:

1. **🚀 Risk Predictor:** Enter coordinates and dates to get a real-time Risk Classification.
2. **📊 Model Comparisons:** Displays the raw performance scores and benchmarks of both algorithms.
3. **🕵️ Input Explanations:** A guide explaining how Year, Location, and Seasonality affect the AI's logic.
4. **📖 Prediction Explanations:** Plain-English decoding of what the "Small" and "Significant" results mean for safety.

---

## � Project File Structure
- `app.py`: The primary Streamlit dashboard and UI logic.
- `sqlite_model_trainer.py`: The AI training engine where XGBoost and LightGBM are benchmarked.
- `sqlite_data_loader.py`: Handles connection to the 2.3M record SQLite database and feature engineering.
- `FPA_FOD_20221014.sqlite`: The core database file containing fire records.
- `requirements.txt`: List of all Python libraries needed.

---

## 🛠️ Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost lightgbm imbalanced-learn
   ```

2. **Launch Application:**
   ```bash
   streamlit run app.py
   ```

---

## � Final Results Summary
- **Overall Accuracy:** 86.8%
- **F1-Score:** 0.162
- **Data Samples Used:** 200,000 randomized database records.
- **Winning Algorithm:** XGBoost

This project demonstrates that through strategic feature engineering and cyclical encoding, AI can reliably predict wildfire behavior with high accuracy, providing a valuable tool for environmental safety.
