# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:13:08 2024

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


INPUT_CSV    = "sales_anomaly_dataset.csv"   
DATE_COL     = "Date"
VALUE_COL    = "Sales_Amount"
Z_THRESHOLD  = 2.0                            
OUTPUT_DIR   = "outputs"                      
MIN_BUCKET   = 10                            
EPS          = 1e-9                          


def load_data(path, date_col, value_col):
    df = pd.read_csv(path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{date_col}' and '{value_col}'")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col).reset_index(drop=True)
    return df

def compute_monthly_baseline(df, date_col, value_col):
    """
    Compute month-level baseline stats and mark whether each month has enough data.
    If a month is sparse, we leave month_mean/std as NaN so detect_anomalies can
    (a) avoid flagging, and (b) optionally use a global fallback for plotting.
    """
    df = df.copy()
    df["month"] = df[date_col].dt.month

    stats = (
        df.groupby("month")[value_col]
          .agg(month_mean="mean",
               month_std=lambda s: s.std(ddof=1),
               month_n="size")
          .reset_index()
    )

   
    stats["enough_data"] = stats["month_n"] >= MIN_BUCKET

   
    stats["month_mean"] = stats["month_mean"].where(stats["enough_data"], np.nan)
    stats["month_std"]  = stats["month_std"].where(stats["enough_data"],  np.nan)

    df = df.merge(stats, on="month", how="left")

    return df

def detect_anomalies(df, value_col, z_threshold):
    """
    Compute z-scores and flags.
    - Only flag anomalies for rows where enough_data == True.
    - For plotting/visibility, we fill missing expected with a global median,
      but we NEVER flag on rows without enough_data.
    """
    df = df.copy()

  
    std_safe = np.where(df["enough_data"].fillna(False),
                        df["month_std"].replace(0.0, EPS),
                        np.nan)


    overall_median = df[value_col].median()
    expected = df["month_mean"].copy()
    expected = expected.fillna(overall_median)

   
    df["z_score"] = (df[value_col] - expected) / std_safe

    
    df["is_anomaly"] = df["enough_data"].fillna(False) & (df["z_score"].abs() > z_threshold)

   
    df["expected"]   = expected
    df["abs_diff"]   = df[value_col] - df["expected"]
    df["confidence"] = np.where(df["enough_data"].fillna(False), "high", "low")

    return df

def save_anomaly_report(df, date_col, value_col, out_dir):
    os.makedirs(out_dir, exist_ok=True)
   
    cols = [date_col, value_col, "expected", "month_std", "z_score", "abs_diff", "confidence", "month_n"]
 
    cols = [c for c in cols if c in df.columns]

    report = (
        df.loc[df["is_anomaly"], cols]
          .sort_values("z_score", key=lambda s: s.abs(), ascending=False)
          .reset_index(drop=True)
    )
    path = os.path.join(out_dir, "anomalies.csv")
    report.to_csv(path, index=False)
    return path

def plot_with_anomalies(df, date_col, value_col, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[date_col], df[value_col], linewidth=1.0, label="Sales")
    ax.plot(df[date_col], df["expected"], linewidth=1.0, alpha=0.6, label="Baseline (expected)")

    anom = df[df["is_anomaly"]]
    if not anom.empty:
        ax.scatter(anom[date_col], anom[value_col], s=36, label="Anomaly")

    ax.set_title(f"Sales Over Time with Anomalies (|z| > {Z_THRESHOLD:.1f})")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "sales_anomalies.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

def main():
    print("Loading data…")
    df = load_data(INPUT_CSV, DATE_COL, VALUE_COL)

    print("Computing monthly baseline (with enough-data gate)…")
    df = compute_monthly_baseline(df, DATE_COL, VALUE_COL)

    print("Detecting anomalies…")
    df = detect_anomalies(df, VALUE_COL, Z_THRESHOLD)

    total_anom = int(df["is_anomaly"].sum())
    high_conf  = int(df.loc[df["is_anomaly"] & (df["confidence"]=="high")].shape[0])
    print(f"Anomalies found: {total_anom} (high-confidence: {high_conf})")

    csv_path = save_anomaly_report(df, DATE_COL, VALUE_COL, OUTPUT_DIR)
    print(f"Saved anomaly report → {csv_path}")

    png_path = plot_with_anomalies(df, DATE_COL, VALUE_COL, OUTPUT_DIR)
    print(f"Saved plot → {png_path}")

if __name__ == "__main__":
    main()
