"""
Model Training Module for Forest Fire Prediction
Trains and compares Random Forest and Logistic Regression models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_loader import load_forest_fire_data


class ForestFirePredictor:
    """
    Class to train and compare forest fire prediction models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.label_encoders = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("🌲 Loading Forest Fires dataset...")
        print("="*70)
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names, self.scaler, self.label_encoders = load_forest_fire_data()
        
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("\n" + "="*70)
        print("🌳 Training Random Forest Classifier...")
        print("="*70)
        
        # Initialize model with optimized hyperparameters
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Train model
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
        
        # Print feature importance
        self._print_feature_importance('Random Forest')
        
        return rf_model, metrics
    
    def train_logistic_regression(self):
        """Train Logistic Regression Classifier"""
        print("\n" + "="*70)
        print("📊 Training Logistic Regression Classifier...")
        print("="*70)
        
        # Initialize model
        lr_model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
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
        
    def _print_feature_importance(self, model_name):
        """Print feature importance for Random Forest"""
        if model_name == 'Random Forest' and 'feature_importance' in self.results[model_name]:
            importance = self.results[model_name]['feature_importance']
            feature_imp_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print(f"\n🔍 Top 5 Most Important Features:")
            print("-" * 60)
            for idx, row in feature_imp_df.head(5).iterrows():
                print(f"{row['Feature']:20s}: {row['Importance']:.4f}")
    
    def compare_models(self):
        """Compare performance of both models"""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70)
        
        comparison_df = pd.DataFrame({
            'Metric': ['Training Accuracy', 'Testing Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Random Forest': [
                self.results['Random Forest']['train_accuracy'],
                self.results['Random Forest']['test_accuracy'],
                self.results['Random Forest']['precision'],
                self.results['Random Forest']['recall'],
                self.results['Random Forest']['f1_score'],
                self.results['Random Forest']['roc_auc']
            ],
            'Logistic Regression': [
                self.results['Logistic Regression']['train_accuracy'],
                self.results['Logistic Regression']['test_accuracy'],
                self.results['Logistic Regression']['precision'],
                self.results['Logistic Regression']['recall'],
                self.results['Logistic Regression']['f1_score'],
                self.results['Logistic Regression']['roc_auc']
            ]
        })
        
        print("\n", comparison_df.to_string(index=False))
        
        # Determine best model based on F1-score (better for imbalanced data)
        rf_score = self.results['Random Forest']['f1_score']
        lr_score = self.results['Logistic Regression']['f1_score']
        
        best_model_name = 'Random Forest' if rf_score > lr_score else 'Logistic Regression'
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {max(rf_score, lr_score):.4f}")
        print(f"   Test Accuracy: {self.results[best_model_name]['test_accuracy']:.4f}")
        
        return comparison_df, best_model_name
    
    def plot_results(self):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Forest Fire Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrices
        for idx, (model_name, metrics) in enumerate(self.results.items()):
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
        rf_metrics = [
            self.results['Random Forest']['test_accuracy'],
            self.results['Random Forest']['precision'],
            self.results['Random Forest']['recall'],
            self.results['Random Forest']['f1_score'],
            self.results['Random Forest']['roc_auc']
        ]
        lr_metrics = [
            self.results['Logistic Regression']['test_accuracy'],
            self.results['Logistic Regression']['precision'],
            self.results['Logistic Regression']['recall'],
            self.results['Logistic Regression']['f1_score'],
            self.results['Logistic Regression']['roc_auc']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        bars1 = ax.bar(x - width/2, rf_metrics, width, label='Random Forest', color='#2ecc71')
        bars2 = ax.bar(x + width/2, lr_metrics, width, label='Logistic Regression', color='#3498db')
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'model_comparison.png'")
        
    def save_best_model(self, best_model_name):
        """Save the best performing model and preprocessing objects"""
        best_model = self.models[best_model_name]
        
        # Save model
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        
        print(f"\n💾 Best model ({best_model_name}) saved successfully!")
        print("   Files created:")
        print("   - best_model.pkl (trained model)")
        print("   - scaler.pkl (feature scaler)")
        print("   - feature_names.pkl (feature names)")
        print("   - label_encoders.pkl (categorical encoders)")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🔥 FOREST FIRE PREDICTION - ML MODEL TRAINING")
    print("="*70)
    
    # Initialize predictor
    predictor = ForestFirePredictor()
    
    # Load data
    predictor.load_data()
    
    # Train models
    predictor.train_random_forest()
    predictor.train_logistic_regression()
    
    # Compare models
    comparison_df, best_model_name = predictor.compare_models()
    
    # Plot results
    predictor.plot_results()
    
    # Save best model
    predictor.save_best_model(best_model_name)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n🚀 Next step: Run 'streamlit run app.py' to launch the web interface")
    

if __name__ == "__main__":
    main()
