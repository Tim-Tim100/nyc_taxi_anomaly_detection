import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_rolling_features(df, window_size=None):
    # Creates rolling mean and rolling std to capture local trend and volatility
    if window_size is None:
        window_size = config.ROLLING_WINDOW
    
    df_features = df.copy()
    
    df_features['rolling_mean'] = df_features['value'].rolling(window=window_size, center=True).mean()
    df_features['rolling_std'] = df_features['value'].rolling(window=window_size, center=True).std()
    
    print(f"Rolling features created with window size {window_size}:")
    print(f"  - rolling_mean: captures local trend")
    print(f"  - rolling_std: captures local volatility")
    
    return df_features

def create_difference_features(df):
    # Computes first and second order differences to capture rate of change
    df_features = df.copy()
    
    df_features['diff_1'] = df_features['value'].diff().fillna(0)
    df_features['diff_2'] = df_features['value'].diff().diff().fillna(0)
    
    print(f"Difference features created:")
    print(f"  - diff_1: first order difference (rate of change)")
    print(f"  - diff_2: second order difference (acceleration)")
    
    return df_features

def create_lag_features(df, lags=None):
    # Creates lagged features to capture temporal dependencies
    if lags is None:
        lags = config.LAG_VALUES
    
    df_features = df.copy()
    
    for lag in lags:
        df_features[f'lag_{lag}'] = df_features['value'].shift(lag)
    
    print(f"Lag features created:")
    print(f"  - lag_{lags}: previous values to capture patterns")
    print(f"  Note: lag_24 captures hourly seasonal pattern (if data is hourly)")
    
    return df_features

def create_statistical_features(df, window_size=None):
    # Creates percentile-based features to capture distribution shape
    if window_size is None:
        window_size = config.STATISTICAL_WINDOW
    
    df_features = df.copy()
    
    df_features['percentile_25'] = df_features['value'].rolling(window=window_size).quantile(0.25)
    df_features['percentile_75'] = df_features['value'].rolling(window=window_size).quantile(0.75)
    df_features['iqr'] = df_features['percentile_75'] - df_features['percentile_25']
    
    print(f"Statistical features created with window size {window_size}:")
    print(f"  - percentile_25: 25th percentile")
    print(f"  - percentile_75: 75th percentile")
    print(f"  - iqr: interquartile range (spread measure)")
    
    return df_features

def drop_nan_rows(df):
    # Removes NaN values created by rolling/lag operations
    initial_rows = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    final_rows = len(df_clean)
    removed = initial_rows - final_rows
    
    print(f"NaN rows removed (from rolling/lag operations):")
    print(f"  Rows removed: {removed}")
    print(f"  Rows remaining: {final_rows}")
    
    return df_clean

def create_features(df, window_size=None, lag_values=None, statistical_window=None):
    # Main feature engineering pipeline that creates all features in sequence
    if window_size is None:
        window_size = config.ROLLING_WINDOW
    if lag_values is None:
        lag_values = config.LAG_VALUES
    if statistical_window is None:
        statistical_window = config.STATISTICAL_WINDOW
    
    print("\n=== Starting Feature Engineering Pipeline ===\n")
    
    print("Step 1: Create rolling features")
    df = create_rolling_features(df, window_size=window_size)
    
    print("\nStep 2: Create difference features")
    df = create_difference_features(df)
    
    print("\nStep 3: Create lag features")
    df = create_lag_features(df, lags=lag_values)
    
    print("\nStep 4: Create statistical features")
    df = create_statistical_features(df, window_size=statistical_window)
    
    print("\nStep 5: Remove NaN values created by feature engineering")
    df = drop_nan_rows(df)
    
    print("\n=== Feature Engineering Complete ===")
    print(f"Final dataset shape: {df.shape}")
    
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'value']]
    print(f"Features created: {feature_columns}")
    
    return df

def save_features_data(df, output_path=None):
    # Saves engineered features to CSV file
    if output_path is None:
        output_path = config.FEATURES_DATA_PATH
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\nFeatures data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving features data: {e}")
        return False

if __name__ == "__main__":
    # Load processed data
    print(f"Loading processed data from {config.PROCESSED_DATA_PATH}...")
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        
        # Create features
        df_features = create_features(df)
        
        # Save features data
        save_features_data(df_features)
        
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {config.PROCESSED_DATA_PATH}")
        print("Please run preprocessing.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")