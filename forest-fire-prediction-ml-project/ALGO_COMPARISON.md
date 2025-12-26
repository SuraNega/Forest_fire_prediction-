# 📊 Algorithm Comparison & Selection

This project compares two fundamental machine learning approaches for predicting forest fires.

## 1. Random Forest Classifier (🏆 Winner)

**How it works:**
Random Forest is an ensemble learning method that builds a "forest" of many decision trees. 
- It uses **bagging** (Bootstrap Aggregating) to train each tree on a different random subset of the data.
- For each split in a tree, it only considers a random subset of features.
- The final prediction is made by taking a **majority vote** from all survival trees.

**Pros for Forest Fires:**
- **Capture Non-Linearities:** Fire risk is non-linear (e.g., risk increases exponentially with temperature). Random Forest handles these curves better than linear models.
- **Feature Interactions:** It automatically understands that high wind *combined* with low humidity is much more dangerous than either alone.
- **Robustness:** Less sensitive to outliers in weather data (like extreme heat waves).

## 2. Logistic Regression

**How it works:**
A linear model used for classification. It calculates a weighted sum of the input features and applies the **Sigmoid function** to output a probability between 0 and 1.

**Pros:**
- **Interpretation:** Directly shows which features increase or decrease risk via coefficients.
- **Speed:** Extremely fast to train and predict.

**Cons for Forest Fires:**
- **Linearity Constraint:** Assumes a straight-line relationship between features and risk, which is often too simple for complex ecological events like fires.

---

## 🏁 Why Random Forest Won?

In our testing, the **Random Forest** outperformed Logistic Regression because:

1.  **Complex Decision Boundaries:** Forest fires are triggered by a "perfect storm" of conditions. Random Forest can create complex, multi-dimensional decision rules.
2.  **Handling Sparse Events:** Fires are relatively rare in the dataset. Random Forest's ability to balance classes and its ensemble nature made it more sensitive to fire patterns.
3.  **Accuracy & F1-Score:** It achieved a higher **F1-Score**, which is the harmonic mean of Precision and Recall. This means it was better at catching actual fires while minimizing false alarms.

---

## 📈 Detailed Performance Analysis

### 🎯 Metric 1: Accuracy (~77%)
Accuracy measures the **overall correctness** of the model. 
*   **77%** means that for every 100 days we test, the model correctly predicts the outcome (Fire or No Fire) for **77 of them**.
*   While 77% is good, in fire prediction, we care more about *not missing a fire* (Recall) than just being right on safe days.

### 🧩 Metric 2: Confusion Matrix Explained
The Confusion Matrix (shown in the charts below) breaks down the answers into 4 categories. Here is how to read it for our **Fire Prediction** problem:

| | Predicted: **NO FIRE** (Safe) | Predicted: **FIRE** (Danger) |
| :--- | :--- | :--- |
| **Actual: NO FIRE** | **True Negative (TN)** <br>✅ *Correctly said Safe.* <br>The model relaxed when it should have. | **False Positive (FP)** <br>⚠️ *False Alarm.* <br>Model predicted fire, but nothing happened. Better safe than sorry! |
| **Actual: FIRE** | **False Negative (FN)** <br>❌ *Missed Fire.* <br>Model said Safe, but a fire started. <br>**This is the most dangerous error.** | **True Positive (TP)** <br>✅ *Correctly Predicted Fire.* <br>The alert worked! Services were warned. |

**For our Random Forest Model:**
*   It has higher **True Positives** (TP) than Logistic Regression.
*   It minimizes **False Negatives** (FN) better, meaning it misses fewer real fires.

---

## 📊 Summary Table

| Metric | Random Forest | Logistic Regression |
| :--- | :--- | :--- |
| **Accuracy** | **~77%** | ~73% |
| **Precision** | **Higher** (Fewer false alarms) | Lower |
| **Recall** | **Higher** (Caught more fires) | Lower |
| **F1-Score** | **Winner (~67%)** | Runner-up (~62%) |
