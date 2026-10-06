# Chapter 23 — Anomaly Detection (Isolation Forest)

## Overview

Anomaly Detection is a Machine Learning technique used to identify observations that behave differently from the normal pattern.

### Real-world applications

- Fraud detection
- Network intrusion detection
- Manufacturing defect detection
- System monitoring
- Sensor failure detection
- Medical abnormality detection

## Techniques Covered

- Z-Score
- IQR
- Isolation Forest
- One-Class SVM
- DBSCAN

## Isolation Forest

Isolation Forest identifies anomalies by randomly splitting the data.

The main idea is:

> **Anomalies are easier to isolate than normal observations.**

Unusual observations are generally separated from the rest of the data using fewer random splits, giving them shorter isolation paths.

## Practical Implementation

For this chapter, a synthetic dataset was created using Scikit-Learn:

- 300 normal observations
- 10 artificial anomalies
- 2 features

The Isolation Forest model was configured with:

```python
IsolationForest(
    contamination=0.05,
    random_state=42
)
```

### Prediction values

- `1` → Normal
- `-1` → Anomaly

### Anomaly Scores

The `decision_function()` was used to examine anomaly scores.

- Higher score → more normal
- Lower score → more anomalous

## Practical Workflow

```text
Generate normal data
        ↓
Add artificial anomalies
        ↓
Visualize dataset
        ↓
Train Isolation Forest
        ↓
Predict normal/anomaly
        ↓
Calculate anomaly scores
        ↓
Evaluate results
        ↓
Experiment with contamination
        ↓
Final visualization
```

## Key Takeaway

An anomaly is not simply a far-away point. It is an observation that behaves differently from the overall pattern in the data.

## Tools

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Google Colab
