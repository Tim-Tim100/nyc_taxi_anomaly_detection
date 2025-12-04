# NYC Taxi Anomaly Detection 🚕

> **Automatically detect unusual patterns in time series data using unsupervised machine learning**

---

## Overview

This project implements a production-grade anomaly detection system that automatically identifies unusual patterns in NYC taxi demand data. Using unsupervised learning, the system learns what "normal" looks like and flags deviations without requiring manual labeling.

**Real-world problem solved**: Companies can't manually review thousands of data points. This system automates anomaly detection 24/7.

---

## Quick Start

### Option 1: Run Everything (2-3 minutes)
```bash
cd nyc_taxi_anomaly_detection
pip install -r requirements.txt
python main.py
```

### Option 2: Learn Step-by-Step (13 minutes)
```bash
jupyter notebook
# Run notebooks in order: 01_data_exploration → 05_evaluation_results
```

---

## What This Project Does

### Input
- Real NYC taxi demand data (1,000 hourly records)
- Raw, messy real-world data

### Process
1. **Clean data** - Handle missing values, duplicates, format timestamps
2. **Engineer features** - Create 15+ features (rolling stats, differences, lags, percentiles)
3. **Train models** - Isolation Forest + Elliptic Envelope with ensemble voting
4. **Detect anomalies** - Flag unusual patterns with confidence scores
5. **Visualize results** - Create plots highlighting detected anomalies

### Output
- `anomalies_detected.csv` - Predictions for all data points
- `visualizations/` - Plots showing detected anomalies
- `models/` - Trained models and scalers for production use
- `metrics_summary.txt` - Performance statistics

---

## Project Structure

```
nyc_taxi_anomaly_detection/
├── main.py                     # Run the complete pipeline
├── config.py                   # Configuration and parameters
├── requirements.txt            # Dependencies
├── inference.py                # Use trained models on new data
│
├── src/                        # Production code
│   ├── data_loader.py         # Download and load data
│   ├── preprocessing.py       # Clean messy data
│   ├── feature_engineering.py # Create features
│   ├── anomaly_detectors.py   # ML algorithms
│   └── utils.py               # Visualization and helpers
│
├── notebooks/                  # Step-by-step learning
│   ├── 01_data_exploration.ipynb
│  
│  
│  
│  
│
├── data/                       # Data storage
│   ├── raw/                   # Downloaded data
│   └── processed/             # Cleaned data
│
├── models/                     # Trained models
│   ├── scaler.pkl
│   ├── isolation_forest_model.pkl
│   └── elliptic_envelope_model.pkl
│
└── results/                    # Output files
    ├── anomalies_detected.csv
    ├── metrics_summary.txt
    └── visualizations/
```

---

## Technical Details

### Two ML Algorithms

**1. Isolation Forest**
- Isolates anomalies through random partitioning
- Fast and effective for high-dimensional data
- Catches isolated unusual points

**2. Elliptic Envelope**
- Statistical approach using covariance matrix
- Fits ellipse around normal data distribution
- Catches statistical outliers

**Why both?** Ensemble voting from two complementary approaches provides more reliable predictions.

### Features Created (15+)

- **Rolling Mean** - Captures trend
- **Rolling Std Dev** - Captures volatility
- **Differences (1st & 2nd order)** - Captures rate of change
- **Lag Features** - Captures temporal patterns
- **Statistical Features** - Captures distribution shape

### Key Production Practice

**Scaler Persistence**: The StandardScaler learned during training is saved and used consistently for all predictions, ensuring reliable results on new data.

---

## Results

The system detects three types of anomalies:

1. **Point Anomalies** - Sudden spikes or drops
2. **Contextual Anomalies** - Unusual for specific context (time of day, day of week)
3. **Pattern Anomalies** - Changes in expected patterns

### Example Output

CSV contains:
- `timestamp` - When data point occurs
- `value` - Actual taxi demand
- `IF_anomaly` - Isolation Forest prediction
- `EE_anomaly` - Elliptic Envelope prediction
- `Ensemble_anomaly` - Both models agree (highest confidence)
- Feature columns - All engineered features

---



### Setup
```bash
git clone https://github.com/YOUR_USERNAME/nyc_taxi_anomaly_detection.git
cd nyc_taxi_anomaly_detection
pip install -r requirements.txt
python main.py
```

---

## Data Source

**Dataset**: Numenta Anomaly Benchmark (NAB) - Real NYC Taxi Demand

- 1,000 hourly records
- Contains known anomalies for validation
- Real-world messy data
- Publicly available benchmark dataset

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.11+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning algorithms |
| Matplotlib | Visualization |
| Jupyter | Interactive learning |

---

## Key Features

✅ **Complete Pipeline** - Data loading to deployment
✅ **Real Messy Data** - Handles missing values, duplicates, formatting issues
✅ **Production Ready** - Models and scalers saved for inference
✅ **Ensemble Approach** - Two algorithms with voting
✅ **Well Documented** - Code comments and learning notebooks
✅ **Reproducible** - Same results every run

---

## Use Cases

This approach applies to any time series anomaly detection:

- **Finance** - Unusual trading patterns, fraud detection
- **Healthcare** - Abnormal patient vitals
- **IT/Ops** - Server outages, performance degradation
- **Manufacturing** - Equipment failures
- **Security** - Intrusion detection, bot activity
- **Utilities** - Unusual consumption patterns

---

## How to Use on Your Data

1. Replace data source in `data_loader.py`
2. Adjust feature engineering if needed (in `config.py`)
3. Run `python main.py`
4. Review results in `results/anomalies_detected.csv`

---


```

---

## References

- **Isolation Forest**: Liu et al. (2008) "Isolation-Based Anomaly Detection"
- **Elliptic Envelope**: Robust Covariance Estimation
- **Numenta NAB**: Benchmark dataset for evaluating anomaly detection algorithms

---

## License

MIT License - Free to use, modify, and distribute

---

## Acknowledgments

- **Data**: Numenta Anomaly Benchmark
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib community
- **Inspiration**: Real-world production anomaly detection systems
