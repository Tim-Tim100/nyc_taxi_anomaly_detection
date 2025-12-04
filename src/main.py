import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_loader, preprocessing, feature_engineering, anomaly_detectors, utils
import pandas as pd
import numpy as np

def main():
    # Main execution pipeline: download -> clean -> features -> train -> predict -> visualize
    
    print("\n" + "="*80)
    print("NYC TAXI ANOMALY DETECTION - MAIN PIPELINE")
    print("="*80 + "\n")
    
    print("PHASE 1: DATA LOADING")
    print("-" * 80)
    df_raw = data_loader.prepare_raw_data(
        url=config.NAB_DATASET_URL,
        output_path=config.RAW_DATA_PATH
    )
    
    if df_raw is None:
        print("Error: Failed to load data. Exiting.")
        return
    
    print("\nPHASE 2: DATA CLEANING AND PREPROCESSING")
    print("-" * 80)
    df_cleaned = preprocessing.clean_data(
        df_raw,
        handle_missing_method=config.HANDLE_MISSING_METHOD,
        remove_outliers=config.REMOVE_OUTLIERS
    )
    preprocessing.save_processed_data(df_cleaned, config.PROCESSED_DATA_PATH)
    
    print("\nPHASE 3: FEATURE ENGINEERING")
    print("-" * 80)
    df_features = feature_engineering.create_features(
        df_cleaned,
        window_size=config.ROLLING_WINDOW,
        lag_values=config.LAG_VALUES,
        statistical_window=config.STATISTICAL_WINDOW
    )
    feature_engineering.save_features_data(df_features, config.FEATURES_DATA_PATH)
    
    print("\nPHASE 4: PREPARE DATA FOR MODEL TRAINING")
    print("-" * 80)
    feature_columns = [col for col in df_features.columns if col not in ['timestamp', 'value']]
    X = df_features[feature_columns].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features used: {feature_columns}")
    
    print("\nPHASE 5: TRAIN ANOMALY DETECTION MODELS")
    print("-" * 80)
    
    if_detector = anomaly_detectors.IsolationForestDetector(
        contamination=config.ISOLATION_FOREST_CONTAMINATION
    )
    if_detector.train(X)
    utils.save_model(if_detector, config.IF_MODEL_PATH, "isolation_forest_model")
    utils.save_model(if_detector.scaler, config.SCALER_PATH, "scaler")
    
    ee_detector = anomaly_detectors.EllipticEnvelopeDetector(
        contamination=config.ELLIPTIC_ENVELOPE_CONTAMINATION
    )
    ee_detector.train(X)
    utils.save_model(ee_detector, config.EE_MODEL_PATH, "elliptic_envelope_model")
    
    print("\nPHASE 6: GENERATE PREDICTIONS")
    print("-" * 80)
    
    if_predictions = if_detector.predict(X)
    print(f"Isolation Forest anomalies: {if_predictions.sum()}")
    
    ee_predictions = ee_detector.predict(X)
    print(f"Elliptic Envelope anomalies: {ee_predictions.sum()}")
    
    ensemble_predictions = anomaly_detectors.ensemble_predictions([if_predictions, ee_predictions])
    
    print("\nPHASE 7: CREATE RESULTS DATAFRAME")
    print("-" * 80)
    
    results_df = df_features.copy()
    results_df['IF_anomaly'] = if_predictions
    results_df['EE_anomaly'] = ee_predictions
    results_df['Ensemble_anomaly'] = ensemble_predictions
    
    utils.save_results(results_df, config.ANOMALIES_OUTPUT)
    
    print("\nPHASE 8: VISUALIZATIONS AND SUMMARY")
    print("-" * 80)
    
    utils.set_plotting_style()
    utils.plot_time_series(results_df, title="Raw NYC Taxi Time Series")
    utils.plot_anomalies(results_df, 'Ensemble_anomaly', title="Anomalies Detected by Ensemble Model")
    
    feature_sample = feature_columns[:4] if len(feature_columns) >= 4 else feature_columns
    utils.plot_feature_distributions(results_df, feature_sample)
    
    utils.print_anomaly_summary(results_df, 'Ensemble_anomaly')
    
    metrics = {
        'Total Records': len(results_df),
        'Isolation Forest Anomalies': int(if_predictions.sum()),
        'Elliptic Envelope Anomalies': int(ee_predictions.sum()),
        'Ensemble Anomalies': int(ensemble_predictions.sum()),
        'Anomaly Percentage': f"{(ensemble_predictions.sum() / len(results_df)) * 100:.2f}%"
    }
    utils.save_metrics_summary(metrics, config.METRICS_OUTPUT)
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {config.RESULTS_PATH}")
    print(f"Models saved to: {config.MODELS_PATH}")
    print(f"Data saved to: {config.PROCESSED_DATA_PATH}")
    print("\n")

if __name__ == "__main__":
    main()