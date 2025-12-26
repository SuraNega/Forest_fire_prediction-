# 🚀 Quick Start Guide - Forest Fire Prediction

## 3-Step Setup

### Step 1: Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit joblib
```

### Step 2: Train Models (ALREADY DONE! ✅)
```bash
python model_trainer.py
```

### Step 3: Launch Web App
```bash
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## Quick Test Cases

### 🔥 High Fire Risk
- **Location:** X=7, Y=5
- **Time:** August, Friday  
- **Weather:** 28°C, 25% humidity, 6 km/h wind, 0mm rain
- **FWI:** FFMC=92, DMC=150, DC=700, ISI=12
- **Expected:** HIGH RISK

### ✅ Low Fire Risk
- **Location:** X=3, Y=4
- **Time:** February, Monday
- **Weather:** 10°C, 70% humidity, 2 km/h wind, 2mm rain
- **FWI:** FFMC=70, DMC=50, DC=200, ISI=3
- **Expected:** LOW RISK

---

## Files Generated ✅
- `best_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - Feature names
- `label_encoders.pkl` - Month/Day encoders
- `model_comparison.png` - Performance visualization

---

## Troubleshooting

**"Model files not found"**  
→ Run `python model_trainer.py` first

**"forestfires.csv not found"**  
→ Ensure CSV is in parent directory

**Port already in use**  
→ Run `streamlit run app.py --server.port 8502`

---

**Ready to predict forest fires! 🔥**
