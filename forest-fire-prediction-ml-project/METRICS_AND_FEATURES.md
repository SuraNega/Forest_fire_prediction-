# 📉 Metrics, Data & Features Explained

## 1. Where do the numbers come from?

The scores (67% Vs 77%) are calculated by the `model_trainer.py` script using the **Test Set** (the 20% of data hidden from the model during training).

### **Random Forest (Winner) Scores:**
*   **Accuracy:** **~77%** (This means 77 out of 100 times, it correctly says "Fire" or "No Fire")
*   **F1-Score:** **~67%** (This is the "quality" of fire detection. It's lower than accuracy because predicting "rare" fires is hard!)

### **Logistic Regression (2nd Place) Scores:**
*   **Accuracy:** ~73%
*   **F1-Score:** ~62%

> **Why did we pick the winner based on 67% (F1) and not 77% (Accuracy)?**
> Because in fire safety, **missing a fire is dangerous**. High Accuracy is easy (just guess "No Fire" every time). High F1-Score proves the model is actually good at catching the real fires.

---

## 2. Does it use `forestfires.csv`?

**YES.** The code reads the `forestfires.csv` file directly.
*   **Input:** It reads rows of weather data from the CSV.
*   **Target:** It looks at the **'area'** column. If `area > 0`, it labels it **"Fire" (1)**. If `area == 0`, it labels as **"No Fire" (0)**.

---

## 3. Simple Explanation of the 12 Features

The model uses **12 distinct pieces of information** to make its decision. Here they are simply:

### **📍 Location & Time (4 Inputs)**
1.  **X Coordinate:** Where in the park (East/West).
2.  **Y Coordinate:** Where in the park (North/South).
3.  **Month:** (e.g., August is hotter/drier than January).
4.  **Day:** (e.g., Weekends usually have more human visitors -> more accidental fires).

### **🔥 The "Fire Indices" (4 Inputs)**
*Think of these as "Dryness Scores" calculated by forest rangers:*
5.  **FFMC:** How dry are the **leaves/twigs** on top? (Dry = easy to spark).
6.  **DMC:** How dry is the **rotten soil** just under the surface?
7.  **DC:** How dry is the **deep soil** down below? (Indicates long droughts).
8.  **ISI:** Based on wind + top dryness. Shows **how fast** a fire would run if it started.

### **☁️ The Weather (4 Inputs)**
9.  **Temperature:** Heat (Hotter = Easier to burn).
10. **Humidity:** Moisture in air (Drier air = Fire sucks moisture from plants).
11. **Wind Speed:** Fans the flames and spreads sparks.
12. **Rain:** Wetting the ground (Resets the danger).
