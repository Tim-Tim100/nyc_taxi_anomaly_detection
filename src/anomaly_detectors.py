import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class AnomalyDetector:
    # Base class for all anomaly detection algorithms
    
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.features_scaled = None
    
    def scale_features(self, features):
        # Standardizes features to mean=0, std=1 for better algorithm performance
        self.features_scaled = self.scaler.fit_transform(features)
        print(f"Features scaled to mean=0, std=1")
        return self.features_scaled
    
    def predict(self, features):
        # Placeholder for prediction logic (overridden in subclasses)
        raise NotImplementedError("Subclasses must implement predict()")

class IsolationForestDetector(AnomalyDetector):
    # Isolation Forest: Isolates anomalies by randomly selecting features and values
    
    def __init__(self, contamination=None, random_state=42):
        if contamination is None:
            contamination = config.ISOLATION_FOREST_CONTAMINATION
        
        super().__init__(contamination)
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.algorithm_name = "Isolation Forest"
    
    def train(self, features):
        # Trains the Isolation Forest model on feature data
        scaled_features = self.scale_features(features)
        
        print(f"\nTraining {self.algorithm_name}...")
        print(f"  Contamination (expected anomaly rate): {self.contamination}")
        print(f"  Training samples: {len(features)}")
        
        self.model.fit(scaled_features)
        
        print(f"  Training complete!")
        print(f"  Model fitted successfully")
        
        return self.model
    
    def predict(self, features):
        # Returns predictions: 1 for anomaly, 0 for normal
        if self.model is None:
            print("Error: Model not trained yet. Call train() first.")
            return None
        
        scaled_features = self.scaler.transform(features)
        predictions = self.model.predict(scaled_features)
        predictions = np.where(predictions == -1, 1, 0)
        
        return predictions
    
    def get_anomaly_scores(self, features):
        # Returns anomaly scores (higher = more anomalous)
        if self.model is None:
            print("Error: Model not trained yet. Call train() first.")
            return None
        
        scaled_features = self.scaler.transform(features)
        scores = -self.model.score_samples(scaled_features)
        
        return scores

class EllipticEnvelopeDetector(AnomalyDetector):
    # Elliptic Envelope: Fits an ellipse around normal data, flags points outside
    
    def __init__(self, contamination=None, random_state=42):
        if contamination is None:
            contamination = config.ELLIPTIC_ENVELOPE_CONTAMINATION
        
        super().__init__(contamination)
        self.model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state
        )
        self.algorithm_name = "Elliptic Envelope"
    
    def train(self, features):
        # Trains the Elliptic Envelope model on feature data
        scaled_features = self.scale_features(features)
        
        print(f"\nTraining {self.algorithm_name}...")
        print(f"  Contamination (expected anomaly rate): {self.contamination}")
        print(f"  Training samples: {len(features)}")
        
        self.model.fit(scaled_features)
        
        print(f"  Training complete!")
        print(f"  Model fitted successfully")
        
        return self.model
    
    def predict(self, features):
        # Returns predictions: 1 for anomaly, 0 for normal
        if self.model is None:
            print("Error: Model not trained yet. Call train() first.")
            return None
        
        scaled_features = self.scaler.transform(features)
        predictions = self.model.predict(scaled_features)
        predictions = np.where(predictions == -1, 1, 0)
        
        return predictions
    
    def get_anomaly_scores(self, features):
        # Returns anomaly scores (lower = more anomalous)
        if self.model is None:
            print("Error: Model not trained yet. Call train() first.")
            return None
        
        scaled_features = self.scaler.transform(features)
        scores = -self.model.mahalanobis(scaled_features)
        
        return scores

def ensemble_predictions(predictions_list):
    # Combines multiple detector predictions using voting (majority wins)
    ensemble = np.column_stack(predictions_list)
    voting_result = (ensemble.sum(axis=1) > len(predictions_list) / 2).astype(int)
    
    anomaly_count = voting_result.sum()
    
    print(f"\nEnsemble voting:")
    print(f"  Models combined: {len(predictions_list)}")
    print(f"  Anomalies from voting: {anomaly_count}")
    
    return voting_result

if __name__ == "__main__":
    # Load features data
    print(f"Loading features data from {config.FEATURES_DATA_PATH}...")
    try:
        df_features = pd.read_csv(config.FEATURES_DATA_PATH)
        
        # Select feature columns (exclude non-feature columns if any remain)
        feature_cols = [col for col in df_features.columns if col not in ['timestamp', 'value']]
        X = df_features[feature_cols]
        
        print(f"Features selected for training: {feature_cols}")
        
        # Initialize detectors
        iso_forest = IsolationForestDetector()
        elliptic_env = EllipticEnvelopeDetector()
        
        # Train models
        iso_forest.train(X)
        elliptic_env.train(X)
        
        # Get predictions
        pred_if = iso_forest.predict(X)
        pred_ee = elliptic_env.predict(X)
        
        # Get anomaly scores
        score_if = iso_forest.get_anomaly_scores(X)
        score_ee = elliptic_env.get_anomaly_scores(X)
        
        # Ensemble predictions
        final_predictions = ensemble_predictions([pred_if, pred_ee])
        
        # Add predictions and scores to dataframe
        df_results = df_features.copy()
        df_results['IF_anomaly'] = pred_if
        df_results['EE_anomaly'] = pred_ee
        df_results['Ensemble_anomaly'] = final_predictions
        df_results['IF_score'] = score_if
        df_results['EE_score'] = score_ee
        
        # Save results
        try:
            df_results.to_csv(config.ANOMALIES_OUTPUT, index=False)
            print(f"\nAnomaly detection results saved to: {config.ANOMALIES_OUTPUT}")
            
            # Print summary
            n_anomalies = df_results['Ensemble_anomaly'].sum()
            print(f"Total anomalies detected: {n_anomalies}")
            print(f"Anomaly rate: {n_anomalies / len(df_results):.2%}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            
    except FileNotFoundError:
        print(f"Error: Features data file not found at {config.FEATURES_DATA_PATH}")
        print("Please run feature_engineering.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")