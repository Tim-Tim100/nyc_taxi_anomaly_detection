import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def handle_missing_values(df, method='forward_fill'):
    # Fills missing values using specified method to handle data gaps
    df_clean = df.copy()
    
    initial_missing = df_clean['value'].isnull().sum()
    
    if method == 'forward_fill':
        df_clean['value'] = df_clean['value'].fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        df_clean['value'] = df_clean['value'].interpolate(method='linear')
    elif method == 'drop':
        df_clean = df_clean.dropna(subset=['value'])
    else:
        print(f"Unknown method: {method}")
        return df
    
    final_missing = df_clean['value'].isnull().sum()
    
    print(f"Missing values handled with '{method}' method:")
    print(f"  Before: {initial_missing} missing values")
    print(f"  After: {final_missing} missing values")
    print(f"  Rows removed: {len(df) - len(df_clean)}")
    
    return df_clean

def remove_outliers_iqr(df, column='value', multiplier=1.5):
    # Removes extreme outliers using Interquartile Range (IQR) method
    df_clean = df.copy()
    
    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_count = len(df_clean)
    df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    final_count = len(df_clean)
    
    removed = initial_count - final_count
    
    print(f"Outlier removal using IQR method:")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Outliers removed: {removed}")
    print(f"  Rows remaining: {final_count}")
    
    return df_clean

def convert_timestamp_to_datetime(df):
    # Converts timestamp column to proper datetime format
    df_clean = df.copy()
    
    try:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        print(f"Timestamp converted to datetime successfully")
        print(f"  Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
        return df_clean
    except Exception as e:
        print(f"Error converting timestamp: {e}")
        return df

def sort_by_timestamp(df):
    # Sorts dataframe by timestamp to ensure chronological order
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Data sorted by timestamp")
    return df_sorted

def remove_duplicates(df):
    # Removes duplicate rows based on timestamp
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    final_count = len(df_clean)
    
    removed = initial_count - final_count
    print(f"Duplicate removal:")
    print(f"  Duplicates removed: {removed}")
    print(f"  Rows remaining: {final_count}")
    
    return df_clean

def clean_data(df, handle_missing_method=None, remove_outliers=None):
    # Main preprocessing pipeline that applies all cleaning steps in order
    if handle_missing_method is None:
        handle_missing_method = config.HANDLE_MISSING_METHOD
    if remove_outliers is None:
        remove_outliers = config.REMOVE_OUTLIERS
    
    print("\n=== Starting Data Cleaning Pipeline ===\n")
    
    print("Step 1: Convert timestamp to datetime")
    df = convert_timestamp_to_datetime(df)
    
    print("\nStep 2: Remove duplicates")
    df = remove_duplicates(df)
    
    print("\nStep 3: Handle missing values")
    df = handle_missing_values(df, method=handle_missing_method)
    
    if remove_outliers:
        print("\nStep 4: Remove statistical outliers")
        df = remove_outliers_iqr(df, multiplier=config.IQR_MULTIPLIER)
    
    print("\nStep 5: Sort by timestamp")
    df = sort_by_timestamp(df)
    
    print("\n=== Data Cleaning Complete ===")
    print(f"Final dataset shape: {df.shape}")
    
    return df

def save_processed_data(df, output_path=None):
    # Saves cleaned dataframe to CSV file
    if output_path is None:
        output_path = config.PROCESSED_DATA_PATH
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False

if __name__ == "__main__":
    # Load raw data
    print(f"Loading raw data from {config.RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
        
        # Clean data
        df_clean = clean_data(df)
        
        # Save processed data
        save_processed_data(df_clean)
        
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {config.RAW_DATA_PATH}")
        print("Please run data_loader.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")