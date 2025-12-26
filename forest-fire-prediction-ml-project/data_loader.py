"""
Data Loader Module for Forest Fire Prediction
Loads and prepares the Forest Fires dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_forest_fire_data(csv_path='forestfires.csv'):
    """
    Load the Forest Fires dataset
    
    Dataset from: UCI Machine Learning Repository
    Source: Cortez and Morais, 2007
    
    Features:
    - X, Y: Spatial coordinates in Montesinho park map (1-9)
    - month: Month of the year (jan to dec)
    - day: Day of the week (mon to sun)
    - FFMC: Fine Fuel Moisture Code (18.7 to 96.20)
    - DMC: Duff Moisture Code (1.1 to 291.3)
    - DC: Drought Code (7.9 to 860.6)
    - ISI: Initial Spread Index (0.0 to 56.10)
    - temp: Temperature in Celsius (2.2 to 33.30)
    - RH: Relative Humidity in % (15.0 to 100)
    - wind: Wind speed in km/h (0.40 to 9.40)
    - rain: Outside rain in mm/m2 (0.0 to 6.4)
    - area: Burned area of the forest (in ha) - TARGET
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders
    """
    
    # Load dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset loaded successfully from {csv_path}")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("Please ensure 'forestfires.csv' is in the project directory")
        raise
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Display basic statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"\nTarget variable (area) statistics:")
    print(df['area'].describe())
    
    # Create binary classification target
    # If area > 0, classify as fire (1), else no fire (0)
    df['fire_occurred'] = (df['area'] > 0).astype(int)
    
    print(f"\n🔥 Fire occurrence distribution:")
    print(df['fire_occurred'].value_counts())
    print(f"Fire occurred: {df['fire_occurred'].sum()} ({df['fire_occurred'].mean()*100:.1f}%)")
    print(f"No fire: {(df['fire_occurred']==0).sum()} ({(1-df['fire_occurred'].mean())*100:.1f}%)")
    
    # Encode categorical variables
    label_encoders = {}
    
    # Encode month
    month_encoder = LabelEncoder()
    df['month_encoded'] = month_encoder.fit_transform(df['month'])
    label_encoders['month'] = month_encoder
    
    # Encode day
    day_encoder = LabelEncoder()
    df['day_encoded'] = day_encoder.fit_transform(df['day'])
    label_encoders['day'] = day_encoder
    
    # Select features for model
    feature_columns = ['X', 'Y', 'month_encoded', 'day_encoded', 
                      'FFMC', 'DMC', 'DC', 'ISI', 
                      'temp', 'RH', 'wind', 'rain']
    
    X = df[feature_columns].values
    y = df['fire_occurred'].values
    
    # Feature names for reference
    feature_names = feature_columns
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Data preprocessing complete!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, label_encoders


def get_feature_descriptions():
    """
    Returns detailed descriptions of each feature
    """
    descriptions = {
        'X': 'X-axis spatial coordinate within the Montesinho park map (1 to 9)',
        'Y': 'Y-axis spatial coordinate within the Montesinho park map (2 to 9)',
        'month': 'Month of the year (January to December)',
        'day': 'Day of the week (Monday to Sunday)',
        'FFMC': 'Fine Fuel Moisture Code from FWI system (18.7 to 96.2) - Indicates moisture content of fine fuels',
        'DMC': 'Duff Moisture Code from FWI system (1.1 to 291.3) - Indicates moisture content of decomposed organic material',
        'DC': 'Drought Code from FWI system (7.9 to 860.6) - Indicates seasonal drought effects',
        'ISI': 'Initial Spread Index from FWI system (0 to 56.1) - Indicates expected rate of fire spread',
        'temp': 'Temperature in Celsius degrees (2.2 to 33.3)',
        'RH': 'Relative Humidity in percentage (15 to 100)',
        'wind': 'Wind speed in km/h (0.4 to 9.4)',
        'rain': 'Outside rain in mm/m² (0.0 to 6.4)'
    }
    return descriptions


def get_month_day_mappings():
    """
    Returns the mappings for months and days
    """
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    
    return months, days


if __name__ == "__main__":
    # Test the data loader
    X_train, X_test, y_train, y_test, features, scaler, encoders = load_forest_fire_data()
    print("\n✅ Data loader test successful!")
