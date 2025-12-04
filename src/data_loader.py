import pandas as pd
import urllib.request
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Dataset URL from Numenta Anomaly Benchmark (NAB) - NYC Taxi dataset
NAB_DATASET_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"

def download_dataset(url=NAB_DATASET_URL, output_path=None):
    # Downloads dataset from GitHub if not already present locally
    if output_path is None:
        output_path = config.RAW_DATA_PATH
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return True
    
    try:
        print(f"Downloading dataset from GitHub...")
        print(f"URL: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"Download successful!")
        print(f"Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_data(file_path=None):
    # Loads CSV file into pandas DataFrame and performs basic validation
    if file_path is None:
        file_path = config.RAW_DATA_PATH
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully!")
        print(f"File path: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print(f"Please download dataset first using download_dataset()")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def validate_data(df):
    # Checks if dataframe has required columns and valid structure
    required_columns = ['timestamp', 'value']
    
    if df is None:
        print("Error: DataFrame is None")
        return False
    
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    print(f"Data validation passed!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Data types:\n{df.dtypes}")
    return True

def get_basic_stats(df):
    # Prints statistical summary of the dataset
    if df is None:
        print("Error: Cannot compute stats on None DataFrame")
        return
    
    print("\n=== Data Statistics ===")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nValue column statistics:")
    print(f"  Min: {df['value'].min():.2f}")
    print(f"  Max: {df['value'].max():.2f}")
    print(f"  Mean: {df['value'].mean():.2f}")
    print(f"  Std Dev: {df['value'].std():.2f}")
    print(f"  Missing values: {df['value'].isnull().sum()}")

def prepare_raw_data(url=NAB_DATASET_URL, output_path=None):
    # Main function that combines all steps: download, load, validate, stats
    if output_path is None:
        output_path = config.RAW_DATA_PATH
    
    print("=== Starting Data Preparation Pipeline ===\n")
    
    print("Step 1: Download dataset")
    download_dataset(url, output_path)
    
    print("\nStep 2: Load data from CSV")
    df = load_data(output_path)
    
    print("\nStep 3: Validate data structure")
    if validate_data(df):
        print("\nStep 4: Display basic statistics")
        get_basic_stats(df)
        print("\n=== Data Preparation Complete ===")
        return df
    else:
        print("Data validation failed!")
        return None

