"""
Enhanced Model Training Module for Forest Fire Prediction
Implements advanced techniques to maximize accuracy:
- SMOTE for class imbalance
- GridSearchCV for hyperparameter tuning
- XGBoost algorithm
- Feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from data_loader import load_forest_fire_data


class EnhancedForestFirePredictor:
    """
    Enhanced class to train and compare forest fire prediction models
    with advanced techniques for maximum accuracy
    """
    
    def __init__(self, use_smote=True, use_grid_search=False):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.label_encoders = None
        self.use_smote = use_smote
        self.use_grid_search = use_grid_search
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("🌲 Loading Forest Fires dataset...")
        print("="*70)
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names, self.scaler, self.label_encoders = load_forest_fire_data()
        
        # Apply SMOTE if enabled
        if self.use_smote:
            print("\n🔄 Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42, k_neighbors=3)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"✅ Training set after SMOTE: {self.X_train.shape[0]} samples")
            print(f"   Class distribution: {np.bincount(self.y_train)}")
    
    def create_engineered_features(self, X):
        """Create new features from existing ones"""
        X_new = X.copy()
        
        # Assuming feature order: X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain
        # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        
        # Heat-Wind interaction (dangerous combination)
        X_new = np.column_stack([X_new, X[:, 8] * X[:, 10]])  # temp * wind
        
        # Dryness index (low humidity + high temp)
        X_new = np.column_stack([X_new, X[:, 8] / (X[:, 9] + 1)])  # temp / (RH + 1)
        
        # Fire Weather Index composite
        X_new = np.column_stack([X_new, (X[:, 4] + X[:, 5] + X[:, 6]) / 3])  # avg of FFMC, DMC, DC
        
        # Moisture deficit
        X_new = np.column_stack([X_new, 100 - X[:, 9]])  # 100 - RH
        
        return X_new
        
    def train_random_forest(self):
        """Train Random Forest Classifier with optional GridSearch"""
        print("\n" + "="*70)
        print("🌳 Training Random Forest Classifier...")
        print("="*70)
        
        if self.use_grid_search:
            print("🔍 Running GridSearchCV for hyperparameter optimization...")
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf_base = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            grid_search = GridSearchCV(
                rf_base, param_grid, cv=5, 
                scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            rf_model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
        else:
            # Optimized parameters from experience
            rf_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = rf_model.predict(self.X_train)
        y_pred_test = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Store model
        self.models['Random Forest'] = rf_model
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            self.y_train, y_pred_train,
            self.y_test, y_pred_test,
            y_pred_proba
        )
        
        # Feature importance
        metrics['feature_importance'] = rf_model.feature_importances_
        
        self.results['Random Forest'] = metrics
        self._print_metrics('Random Forest', metrics)
        
        return rf_model, metrics
    
    def train_xgboost(self):
        """Train XGBoost Classifier"""
        print("\n" + "="*70)
        print("🚀 Training XGBoost Classifier...")
        print("="*70)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train model
        xgb_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = xgb_model.predict(self.X_train)
        y_pred_test = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Store model
        self.models['XGBoost'] = xgb_model
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            self.y_train, y_pred_train,
            self.y_test, y_pred_test,
            y_pred_proba
        )
        
        # Feature importance
        metrics['feature_importance'] = xgb_model.feature_importances_
        
        self.results['XGBoost'] = metrics
        self._print_metrics('XGBoost', metrics)
        
        return xgb_model, metrics
    
    def train_logistic_regression(self):
        """Train Logistic Regression Classifier"""
        print("\n" + "="*70)
        print("📊 Training Logistic Regression Classifier...")
        print("="*70)
        
        lr_model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'
        )
        
        # Train model
        lr_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = lr_model.predict(self.X_train)
        y_pred_test = lr_model.predict(self.X_test)
        y_pred_proba = lr_model.predict_proba(self.X_test)[:, 1]
        
        # Store model
        self.models['Logistic Regression'] = lr_model
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            self.y_train, y_pred_train,
            self.y_test, y_pred_test,
            y_pred_proba
        )
        
        self.results['Logistic Regression'] = metrics
        self._print_metrics('Logistic Regression', metrics)
        
        return lr_model, metrics
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test, y_pred_proba):
        """Calculate performance metrics"""
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba
        }
        return metrics
    
    def _print_metrics(self, model_name, metrics):
        """Print model metrics"""
        print(f"\n{model_name} Performance Metrics:")
        print("-" * 60)
        print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Testing Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1-Score:          {metrics['f1_score']:.4f}")
        print(f"ROC-AUC Score:     {metrics['roc_auc']:.4f}")
        
    def compare_models(self):
        """Compare performance of all models"""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70)
        
        comparison_data = {
            'Metric': ['Training Accuracy', 'Testing Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        }
        
        for model_name in self.results.keys():
            comparison_data[model_name] = [
                self.results[model_name]['train_accuracy'],
                self.results[model_name]['test_accuracy'],
                self.results[model_name]['precision'],
                self.results[model_name]['recall'],
                self.results[model_name]['f1_score'],
                self.results[model_name]['roc_auc']
            ]
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))
        
        # Determine best model based on F1-score
        best_f1 = 0
        best_model_name = None
        for model_name, metrics in self.results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model_name = model_name
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {best_f1:.4f}")
        print(f"   Test Accuracy: {self.results[best_model_name]['test_accuracy']:.4f}")
        
        return comparison_df, best_model_name
    
    def plot_results(self):
        """Generate visualization plots"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Forest Fire Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrices (first two models)
        for idx, (model_name, metrics) in enumerate(list(self.results.items())[:2]):
            ax = axes[0, idx]
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_xticklabels(['No Fire', 'Fire'])
            ax.set_yticklabels(['No Fire', 'Fire'])
        
        # 2. ROC Curves
        ax = axes[1, 0]
        for model_name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
            ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # 3. Metrics Comparison
        ax = axes[1, 1]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        x = np.arange(len(metrics_names))
        width = 0.25
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for idx, (model_name, color) in enumerate(zip(self.results.keys(), colors)):
            model_metrics = [
                self.results[model_name]['test_accuracy'],
                self.results[model_name]['precision'],
                self.results[model_name]['recall'],
                self.results[model_name]['f1_score'],
                self.results[model_name]['roc_auc']
            ]
            offset = (idx - 1) * width
            bars = ax.bar(x + offset, model_metrics, width, label=model_name, color=color)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=7)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig('model_comparison_enhanced.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'model_comparison_enhanced.png'")


def main():
    """Main training pipeline with enhancements"""
    print("\n" + "="*70)
    print("🔥 ENHANCED FOREST FIRE PREDICTION - ML MODEL TRAINING")
    print("="*70)
    print("Applying advanced techniques:")
    print("  ✓ SMOTE for class imbalance")
    print("  ✓ XGBoost algorithm")
    print("  ✓ Optimized hyperparameters")
    print("="*70)
    
    # Initialize predictor with SMOTE enabled
    predictor = EnhancedForestFirePredictor(use_smote=True, use_grid_search=False)
    
    # Load data
    predictor.load_data()
    
    # Train models
    predictor.train_random_forest()
    predictor.train_xgboost()
    predictor.train_logistic_regression()
    
    # Compare models
    comparison_df, best_model_name = predictor.compare_models()
    
    # Plot results
    predictor.plot_results()
    
    print("\n" + "="*70)
    print("✅ ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📈 Accuracy Improvement Techniques Applied:")
    print(f"   • SMOTE for balanced training data")
    print(f"   • XGBoost ensemble algorithm")
    print(f"   • Optimized Random Forest parameters")
    print(f"   • Class weight balancing")
    

if __name__ == "__main__":
    main()
