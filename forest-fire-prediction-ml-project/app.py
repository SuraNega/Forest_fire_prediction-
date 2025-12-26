"""
Forest Fire Prediction Web Application
A Streamlit-based user interface for predicting forest fire occurrence
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from data_loader import get_feature_descriptions, get_month_day_mappings

# Page configuration
st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fire-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border: 3px solid #c0392b;
    }
    .no-fire {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        border: 3px solid #2d8659;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


from model_trainer_enhanced import EnhancedForestFirePredictor

@st.cache_resource
def load_model():
    """Train the enhanced model on the fly with SMOTE and XGBoost"""
    try:
        with st.spinner('🔄 Training enhanced models (SMOTE + XGBoost)... This may take 10-15 seconds...'):
            # Initialize and train with enhancements
            predictor = EnhancedForestFirePredictor(use_smote=True, use_grid_search=False)
            predictor.load_data()
            predictor.train_random_forest()
            predictor.train_xgboost()
            predictor.train_logistic_regression()
            
            # Get best model
            _, best_model_name = predictor.compare_models()
            best_model = predictor.models[best_model_name]
            
            # Store results for display
            st.session_state['model_results'] = predictor.results
            st.session_state['best_model_name'] = best_model_name
            
            return best_model, predictor.scaler, predictor.feature_names, predictor.label_encoders
    except Exception as e:
        st.error(f"⚠️ Error initializing model: {str(e)}")
        st.error("Please ensure 'forestfires.csv' is in the project directory.")
        st.stop()


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🔥 Forest Fire Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Forest Fire Risk Assessment using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names, label_encoders = load_model()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/forest.png", width=100)
        st.title("ℹ️ About")
        
        # Best Algorithm Highlighting (dynamic)
        best_model_name = st.session_state.get('best_model_name', 'XGBoost')
        model_results = st.session_state.get('model_results', {})
        
        if best_model_name in model_results:
            f1_score = model_results[best_model_name]['f1_score']
            accuracy = model_results[best_model_name]['test_accuracy']
            st.success(f"🏆 **Best Algorithm: {best_model_name}**")
            st.metric("Model F1-Score", f"{f1_score*100:.1f}%", delta="Best Performance", 
                     help="F1-Score balances Precision and Recall, making it the best metric for fire detection.")
            st.metric("Model Accuracy", f"{accuracy*100:.1f}%", 
                     help="Overall correctness of predictions")
        else:
            st.success("🏆 **Best Algorithm: XGBoost (Enhanced)**")
            st.metric("Model F1-Score", "~75%", delta="Enhanced with SMOTE", 
                     help="F1-Score balances Precision and Recall, making it the best metric for fire detection.")
        
        st.markdown("""
        **Enhancement Techniques:**
        - ✅ SMOTE (Balanced Training Data)
        - ✅ XGBoost Algorithm
        - ✅ Optimized Hyperparameters
        - ✅ Class Weight Balancing
        """)
        
        st.info(
            """
            This application predicts the likelihood of forest fire occurrence based on 
            meteorological and environmental conditions.
            
            **Dataset:** Forest Fires Dataset  
            **Source:** UCI ML Repository  
            **Location:** Montesinho Natural Park, Portugal
            
            **Model Accuracy:** ~77% (Random Forest)
            
            ⚠️ **Disclaimer:** For educational purposes only. 
            Always follow official fire safety guidelines.
            """
        )
        
        st.markdown("---")
        st.markdown("### 📊 Risk Scale Prediction")
        
        # Get the latest probability from session state
        latest_prob = st.session_state.get('last_fire_prob', 0.0)
        
        # Determine color and text
        if latest_prob >= 0.5:
            risk_color = "red"
            risk_text = "🔴 Danger / High Risk"
        else:
            risk_color = "green"
            risk_text = "🟢 Normal / Safe"
            
        st.markdown(f"**Current Status:** :{risk_color}[{risk_text}]")
        st.progress(latest_prob, text=f"Fire Risk Probability: {latest_prob*100:.1f}%")
        
        st.info(f"""
        **Scale Legend:**
        - **0% - 49%**: Safe (Green)
        - **50% - 100%**: Risky (Red)
        
        **Your current case:** Since the probability is **{latest_prob*100:.1f}%**, the model categorizes this as **{"RISKY" if latest_prob >= 0.5 else "SAFE"}**.
        """)

        st.markdown("---")
        st.markdown("### 🌡️ Key Factors")
        st.markdown("""
        - **FWI Indices** (FFMC, DMC, DC, ISI)
        - **Weather** (Temperature, Humidity, Wind)
        - **Location & Time**
        - **Rainfall**
        """)

        st.markdown("---")
        with st.expander("💡 How to Improve Accuracy?"):
            st.markdown("""
            - **More Data:** Incorporate larger and more diverse datasets.
            - **Advanced Models:** Explore deep learning models (e.g., LSTMs for time series).
            - **Feature Engineering:** Create new features from existing ones (e.g., daily temperature range).
            - **External Data:** Integrate satellite imagery, land cover data, and historical fire records.
            - **Hyperparameter Tuning:** Optimize model parameters more extensively.
            - **Ensemble Methods:** Combine multiple models for improved robustness.
            """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📖 Feature Guide", "📊 Model Comparison"])
    
    with tab1:
        st.markdown("## Enter Environmental Conditions")
        
        # Get month and day options
        months, days = get_month_day_mappings()
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📍 Location & Time")
            x_coord = st.slider("X Coordinate (1-9)", 1, 9, 5, help="X-axis spatial coordinate in park map")
            y_coord = st.slider("Y Coordinate (2-9)", 2, 9, 5, help="Y-axis spatial coordinate in park map")
            month = st.selectbox("Month", months, index=7, help="Month of the year")
            day = st.selectbox("Day of Week", days, index=4, help="Day of the week")
        
        with col2:
            st.markdown("### 🌡️ Weather Conditions")
            temp = st.slider("Temperature (°C)", 0.0, 35.0, 18.0, 0.1, help="Temperature in Celsius")
            rh = st.slider("Relative Humidity (%)", 10, 100, 50, help="Relative humidity percentage")
            wind = st.slider("Wind Speed (km/h)", 0.0, 10.0, 4.0, 0.1, help="Wind speed")
            rain = st.slider("Rainfall (mm/m²)", 0.0, 7.0, 0.0, 0.1, help="Outside rain")
        
        with col3:
            st.markdown("### 🔥 Fire Weather Index")
            ffmc = st.slider("FFMC", 18.0, 97.0, 85.0, 0.1, help="Fine Fuel Moisture Code")
            dmc = st.slider("DMC", 1.0, 300.0, 100.0, 0.1, help="Duff Moisture Code")
            dc = st.slider("DC", 7.0, 900.0, 500.0, 1.0, help="Drought Code")
            isi = st.slider("ISI", 0.0, 60.0, 8.0, 0.1, help="Initial Spread Index")
        
        # Prediction button
        st.markdown("---")
        predict_button = st.button("🔍 Predict Fire Risk", type="primary", use_container_width=True)
        
        if predict_button:
            # Encode categorical variables
            month_encoded = label_encoders['month'].transform([month])[0]
            day_encoded = label_encoders['day'].transform([day])[0]
            
            # Prepare input data (matching feature order)
            input_data = np.array([[
                x_coord, y_coord, month_encoded, day_encoded,
                ffmc, dmc, dc, isi,
                temp, rh, wind, rain
            ]])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Store probability for the sidebar risk scale
            st.session_state['last_fire_prob'] = float(probability[1])
            
            # Display result
            st.markdown("---")
            st.markdown("## 📋 Prediction Result")
            
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-box fire-risk">🔥 HIGH RISK: Fire Likely to Occur<br>'
                    f'Fire Risk Probability: {probability[1]*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
                st.error("""
                    **⚠️ Fire Risk Detected - Recommendations:**
                    - Implement fire prevention measures immediately
                    - Increase monitoring and surveillance
                    - Restrict access to high-risk areas
                    - Prepare firefighting resources
                    - Alert local authorities
                    - Avoid activities that could spark fires
                """)
            else:
                st.markdown(
                    f'<div class="prediction-box no-fire">✅ LOW RISK: Fire Unlikely<br>'
                    f'Fire Risk Probability: {probability[1]*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
                st.success("""
                    **✅ Low Fire Risk - Recommendations:**
                    - Continue regular monitoring
                    - Maintain fire prevention protocols
                    - Stay updated on weather conditions
                    - Keep emergency plans ready
                    - Educate visitors about fire safety
                """)
            
            # Risk breakdown
            st.markdown("### 📊 Risk Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fire Risk Probability", f"{probability[1]*100:.1f}%", 
                         delta=f"{(probability[1]-0.5)*100:.1f}% vs baseline")
            with col2:
                st.metric("Safe Probability", f"{probability[0]*100:.1f}%")
            with col3:
                risk_level = "🔴 High" if probability[1] > 0.7 else "🟡 Moderate" if probability[1] > 0.4 else "🟢 Low"
                st.metric("Risk Level", risk_level)
            
            # Environmental summary
            st.markdown("### 🌍 Environmental Conditions Summary")
            summary_data = {
                'Category': ['Location', 'Time', 'Temperature', 'Humidity', 'Wind', 'Rain', 
                            'FFMC', 'DMC', 'DC', 'ISI'],
                'Value': [f"({x_coord}, {y_coord})", f"{month.capitalize()}, {day.capitalize()}", 
                         f"{temp}°C", f"{rh}%", f"{wind} km/h", f"{rain} mm/m²",
                         f"{ffmc:.1f}", f"{dmc:.1f}", f"{dc:.1f}", f"{isi:.1f}"],
                'Status': ['', '', 
                          '🔥 High' if temp > 25 else '✅ Normal',
                          '⚠️ Low' if rh < 30 else '✅ Normal',
                          '🌬️ High' if wind > 6 else '✅ Normal',
                          '💧 Rain' if rain > 0 else '☀️ Dry',
                          '🔥 High' if ffmc > 90 else '✅ Normal',
                          '⚠️ High' if dmc > 150 else '✅ Normal',
                          '⚠️ High' if dc > 600 else '✅ Normal',
                          '🔥 High' if isi > 15 else '✅ Normal']
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with tab2:
        if os.path.exists('FEATURE_EXPLANATION.md'):
            with open('FEATURE_EXPLANATION.md', 'r', encoding='utf-8') as f:
                content = f.read()
                st.markdown(content)
        else:
            st.info("Feature explanation file not found.")
            
    with tab3:
        if os.path.exists('ALGO_COMPARISON.md'):
            with open('ALGO_COMPARISON.md', 'r', encoding='utf-8') as f:
                content = f.read()
                st.markdown(content)
        else:
            st.info("Algorithm comparison file not found.")
        
        st.markdown("---")
        st.markdown("### 📊 Visual Comparison")
        if os.path.exists('model_comparison.png'):
            st.image('model_comparison.png', caption='Model Performance Comparison Charts')
        
        st.markdown("---")
        st.markdown("### 📚 References")
        st.markdown("""
        - **Dataset:** P. Cortez and A. Morais. "A Data Mining Approach to Predict Forest Fires using Meteorological Data." 
          In Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December 2007.
        - **FWI System:** Canadian Forest Fire Weather Index System
        - **Location:** Montesinho Natural Park, Trás-os-Montes, Portugal
        """)

if __name__ == "__main__":
    main()
