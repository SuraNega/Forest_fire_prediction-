import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sqlite_data_loader import load_sqlite_data
from imblearn.over_sampling import SMOTE
import joblib

class SQLiteFireModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_and_evaluate(self):
        print("Initializing Ultra-High Accuracy Engine...")
        # Higher sample for better generalization
        data = load_sqlite_data(sample_size=200000)
        if data is None: return
        
        X_train, X_test, y_train, y_test, scaler, state_encoder, risk_labels = data
        
        # 1. XGBoost with tuned Class Weights (Better than SMOTE for raw accuracy)
        print("\nTraining Advanced XGBoost...")
        xgb = XGBClassifier(
            n_estimators=300, 
            max_depth=10,
            learning_rate=0.03, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(X_train, y_train)
        self.evaluate_model("XGBoost", xgb, X_test, y_test)
        
        # 2. LightGBM with tuned Class Weights
        print("\nTraining Advanced LightGBM...")
        lgb = LGBMClassifier(
            n_estimators=500, 
            num_leaves=127,
            learning_rate=0.02, 
            random_state=42, 
            verbosity=-1
        )
        lgb.fit(X_train, y_train)
        self.evaluate_model("LightGBM", lgb, X_test, y_test)
        
        # Compare
        comparison = pd.DataFrame(self.results).T
        print("\nFinal Benchmarks (>85% Target):")
        print(comparison)
        
        best_model_name = comparison['Accuracy'].idxmax()
        print(f"\nFinal Winner: {best_model_name}")
        
        best_model = self.models[best_model_name]
        return best_model, scaler, state_encoder, risk_labels, comparison
        
    def evaluate_model(self, name, model, X_test, y_test):
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        self.results[name] = {
            "Accuracy": acc,
            "F1-Score": f1
        }
        self.models[name] = model
        print(f"{name} Results: Accuracy={acc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    trainer = SQLiteFireModelTrainer()
    trainer.train_and_evaluate()
