# 📊 Algorithm Comparison & Selection

This project compares three distinct machine learning models trained on the Forest Fires dataset to ensure the most reliable predictions.

---

## 🔝 1. XGBoost (🏆 The Overall Winner)

**How it works:**
XGBoost (Extreme Gradient Boosting) is an advanced ensemble method. It builds trees sequentially, where each new tree specifically focuses on correcting the errors made by the previous trees.

**Why it is best for this project:**
- **Highest Accuracy:** It achieved the highest testing accuracy and F1-score in our final tests.
- **Precision & Recall:** It provides the best balance between catching real fires and avoiding false alarms.
- **Handling Complexity:** Forest fire patterns are mathematically complex; XGBoost is designed to find these thin patterns better than simpler models.

---

## 🌳 2. Random Forest Classifier

**How it works:**
Random Forest is a "Bagging" method. It creates 300 different decision trees at the same time and lets them "vote" on whether a fire will occur.

**Strengths:**
- **Robustness:** It is very resistant to "noise" in the weather data.
- **Outlier Handling:** It performs well even when weather readings are extreme.
- **Stability:** It provides reliable results across different types of forest conditions.

---

## 📉 3. Logistic Regression (The Baseline)

**How it works:**
Logistic Regression is a linear probabilistic model. It tries to draw a straight decision boundary to separate "Safe" conditions from "Fire" conditions.

**Role in the project:**
- **Baseline:** We use it as a reference point to prove that the more complex models (XGBoost and Random Forest) provide a significant performance boost.
- **Interpretation:** It shows the basic linear relationship between features like temperature and humidity with fire risk.

---

## 🏁 Final Comparison Summary

| Metric | Logistic Regression | Random Forest | **XGBoost (Winner)** |
| :--- | :--- | :--- | :--- |
| **Testing Accuracy** | ~53.8% | ~64.4% | **~67.3%** |
| **F1-Score** | ~56.3% | ~65.4% | **~67.9%** |
| **Fire Detection Rate** | Low | Medium-High | **Highest** |
| **Stability** | Good | Excellent | **Excellent** |

---

## 🎨 Training Enhancements (Under the Hood)

To reach these accuracy levels, the current system uses:
1. **SMOTE:** Artificially balances the training data so the models see enough "Fire" examples.
2. **Feature Engineering:** Creates specific indices (like Heat-Wind interaction) to help the algorithms "see" the risk more clearly.
3. **Hyperparameter Tuning:** Optimized settings for depth, learning rate, and tree counts.
