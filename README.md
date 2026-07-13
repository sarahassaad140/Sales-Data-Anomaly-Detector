# Sales Data Anomaly Detector

A lightweight and interpretable anomaly detection system for identifying unusual patterns in daily sales data.

The application analyzes historical transaction data, learns expected seasonal behavior, and automatically flags days where sales significantly deviate from normal patterns.

The project combines statistical anomaly detection with an interactive Streamlit dashboard, providing an auditable solution for monitoring sales fluctuations, operational issues, and unexpected business events.

---

## Project Overview

Businesses often need to monitor sales performance and quickly identify unusual changes such as:

- Sudden drops in revenue
- Unexpected sales spikes
- Data recording errors
- Missing transactions
- Unusual customer behavior
- Effects of promotions or external events

This project provides an automated solution that detects these anomalies while remaining:

 Interpretable  
 Lightweight  
 Reproducible  
 Easy to audit  

Instead of relying on complex black-box machine learning models, the system uses statistical methods that clearly explain why an observation was flagged.

---

# Features

## Statistical Anomaly Detection

The system:

- Groups sales data by calendar month
- Computes monthly expected sales behavior
- Calculates deviations using statistical scoring
- Flags observations outside the expected range

Anomalies are detected when:

\[
|z-score| > threshold
\]

where the default threshold is ±2 standard deviations.

---

## Seasonal Awareness

Sales patterns often vary depending on the month.

Instead of comparing every day against a single global average, the system builds separate baselines for each month.

Example:

January sales are compared with other January sales.

This reduces false alerts caused by natural seasonal variations.

---

## Data Sufficiency Protection

Small datasets can create unreliable statistical estimates.

To avoid misleading results, the detector applies a minimum data requirement:

- Months with insufficient historical data are ignored
- No anomaly alerts are generated from unreliable baselines

This improves precision and reduces false positives.

---

## Interactive Dashboard

A Streamlit interface allows users to:

- Upload sales CSV files
- Adjust anomaly detection thresholds
- Visualize sales trends
- Inspect detected anomalies
- Export reports

---

#  Example Output

The application generates:

### 1. Sales Time-Series Visualization

Shows:

- Actual daily sales
- Expected monthly baseline
- Detected anomalies

### 2. Anomaly Report

Contains:

| Date | Sales | Expected Sales | Z-score | Deviation |
|---|---|---|---|---|

This allows users to understand why each anomaly was detected.

---

#  Technologies Used

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

## Application Framework

- Streamlit

## Statistical Methods

- Mean and standard deviation
- Z-score anomaly detection
- Seasonal baselines
- Data sufficiency filtering

---

# Project Architecture

Input CSV
    |
    ↓
Data Preprocessing
    |
    ↓
Monthly Seasonal Baseline Calculation
    |
    ↓
Z-score Anomaly Detection
    |
    ↓
Filtering Using Data Sufficiency Rules
    |
    ↓
Visualization & Report Generation
    |
    ↓
Streamlit Dashboard

---

#  Dataset Format

The application expects a CSV file containing:

| Column | Description |
|---|---|
| Date | Transaction date |
| Sales | Daily sales value |

Example:

```csv
Date,Sales
2024-01-01,1250
2024-01-02,1320
2024-01-03,900
