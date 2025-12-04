import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ===== DATA PATHS =====
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'nyc_taxi.csv')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nyc_taxi_processed.csv')
FEATURES_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nyc_taxi_features.csv')

# ===== MODEL PATHS =====
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
SCALER_PATH = os.path.join(MODELS_PATH, 'scaler.pkl')
IF_MODEL_PATH = os.path.join(MODELS_PATH, 'isolation_forest_model.pkl')
EE_MODEL_PATH = os.path.join(MODELS_PATH, 'elliptic_envelope_model.pkl')

# ===== RESULTS PATHS =====
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
VISUALIZATION_PATH = os.path.join(RESULTS_PATH, 'visualizations')
ANOMALIES_OUTPUT = os.path.join(RESULTS_PATH, 'anomalies_detected.csv')
METRICS_OUTPUT = os.path.join(RESULTS_PATH, 'metrics_summary.txt')

# ===== DATASET URL =====
NAB_DATASET_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"

# ===== ALGORITHM PARAMETERS =====
ISOLATION_FOREST_CONTAMINATION = 0.05
ELLIPTIC_ENVELOPE_CONTAMINATION = 0.05
ENSEMBLE_THRESHOLD = 0.5

# ===== FEATURE ENGINEERING PARAMETERS =====
ROLLING_WINDOW = 5
LAG_VALUES = [1, 2, 3, 24]
STATISTICAL_WINDOW = 24

# ===== PREPROCESSING PARAMETERS =====
HANDLE_MISSING_METHOD = 'forward_fill'
REMOVE_OUTLIERS = False
IQR_MULTIPLIER = 1.5

# ===== CREATE DIRECTORIES IF THEY DON'T EXIST =====
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

print("Configuration loaded successfully!")
print(f"Raw data path: {RAW_DATA_PATH}")
print(f"Processed data path: {PROCESSED_DATA_PATH}")
print(f"Models path: {MODELS_PATH}")
print(f"Results path: {RESULTS_PATH}")