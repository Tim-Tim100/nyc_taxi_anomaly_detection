import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def set_plotting_style():
    # Sets consistent plotting style for all visualizations
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 6)
    plt.rcParams['font.size'] = 10
    print("Plotting style set")

def plot_time_series(df, title="Time Series Data", save_path=None):
    # Plots raw time series to visualize patterns and identify obvious anomalies
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df['value'], linewidth=0.8, alpha=0.7, label='Value')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(config.VISUALIZATION_PATH, 'time_series.png')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # plt.show()

def plot_anomalies(df, anomaly_column, title="Anomaly Detection Results", save_path=None):
    # Plots time series with anomalies highlighted in red
    plt.figure(figsize=(15, 5))
    
    normal_mask = df[anomaly_column] == 0
    anomaly_mask = df[anomaly_column] == 1
    
    plt.plot(df.index[normal_mask], df.loc[normal_mask, 'value'], 
             linewidth=0.8, alpha=0.7, label='Normal', color='blue')
    plt.scatter(df.index[anomaly_mask], df.loc[anomaly_mask, 'value'], 
                color='red', s=50, label='Anomaly', zorder=5)
    
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(config.VISUALIZATION_PATH, 'anomalies_detected.png')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # plt.show()

def plot_feature_distributions(df, features, save_path=None):
    # Creates histogram to show feature distributions and identify skewness
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feature in enumerate(features):
        axes[idx].hist(df[feature], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[idx].set_title(f"Distribution of {feature}")
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(config.VISUALIZATION_PATH, 'feature_distributions.png')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # plt.show()

def save_model(model, filepath=None, model_name="model"):
    # Saves trained model as pickle file for later reuse
    if filepath is None:
        filepath = os.path.join(config.MODELS_PATH, f'{model_name}.pkl')
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(filepath):
    # Loads previously trained model from pickle file
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_results(df, filepath=None):
    # Saves dataframe with anomaly results as CSV for reporting
    if filepath is None:
        filepath = config.ANOMALIES_OUTPUT
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def print_anomaly_summary(df, anomaly_column):
    # Prints statistical summary of detected anomalies
    anomaly_count = (df[anomaly_column] == 1).sum()
    anomaly_percentage = (anomaly_count / len(df)) * 100
    
    print("\n=== Anomaly Detection Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
    print(f"Normal records: {len(df) - anomaly_count}")
    
    if anomaly_count > 0:
        anomaly_indices = df[df[anomaly_column] == 1].index.tolist()
        print(f"First 10 anomaly indices: {anomaly_indices[:10]}")

def save_metrics_summary(metrics_dict, filepath=None):
    # Saves metrics summary to text file
    if filepath is None:
        filepath = config.METRICS_OUTPUT
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value}\n")
        print(f"Metrics saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False

if __name__ == "__main__":
    # Load anomaly results
    print(f"Loading anomaly results from {config.ANOMALIES_OUTPUT}...")
    try:
        df_results = pd.read_csv(config.ANOMALIES_OUTPUT)
        
        # Set plotting style
        set_plotting_style()
        
        # Plot time series
        plot_time_series(df_results, title="NYC Taxi Data - Time Series")
        
        # Plot anomalies
        plot_anomalies(df_results, anomaly_column='anomaly', title="NYC Taxi Data - Anomalies Detected")
        
        # Print summary
        print_anomaly_summary(df_results, anomaly_column='anomaly')
        
    except FileNotFoundError:
        print(f"Error: Anomaly results file not found at {config.ANOMALIES_OUTPUT}")
        print("Please run anomaly_detectors.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")