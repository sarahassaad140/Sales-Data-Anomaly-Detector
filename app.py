# app.py
# Streamlit UI for the Sales Anomaly Detector
# - Monthly / Rolling baselines (mean/std or robust median/MAD)
# - Enough-data gating
# - Persistence rule
# - Ignore (known) dates
# - Alert budget (auto-τ)
# - Episode detection
# - Cost-aware ranking
# - Calendar Heatmap (Month x Day)
# - Chart & CSV downloads

import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Sales Anomaly Detector", layout="wide", initial_sidebar_state="expanded")


EPS = 1e-9 



def _mad_std(x: np.ndarray) -> float:
   
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad  # scale factor


def read_uploaded_table(file, nrows=None, delimiter_choice="auto") -> pd.DataFrame:
  
    name = getattr(file, "name", "")
    ext = os.path.splitext(name)[1].lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file, nrows=nrows)

    # CSV-like
    if delimiter_choice == "auto":
        return pd.read_csv(file, sep=None, engine="python", nrows=nrows)
    elif delimiter_choice.startswith(","):
        return pd.read_csv(file, sep=",", nrows=nrows)
    elif delimiter_choice.startswith(";"):
        return pd.read_csv(file, sep=";", nrows=nrows)
    elif "tab" in delimiter_choice or delimiter_choice.startswith("\\t"):
        return pd.read_csv(file, sep="\t", nrows=nrows)
    else:
        return pd.read_csv(file, sep=None, engine="python", nrows=nrows)


def load_data_from_upload(file, date_col, value_col, delimiter_choice="auto") -> pd.DataFrame:
    """Full load with parsing + cleaning + sorting."""
    df = read_uploaded_table(file, nrows=None, delimiter_choice=delimiter_choice)

    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{date_col}' and '{value_col}' columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = (
        df.dropna(subset=[date_col, value_col])
          .sort_values(date_col)
          .reset_index(drop=True)
    )
    return df


def compute_monthly_baseline(df, date_col, value_col, min_bucket=10, mode="mean") -> pd.DataFrame:
    """
    Monthly seasonal baseline.
    mode: "mean" (mean/std) or "robust" (median/MAD*1.4826)
    """
    df = df.copy()
    df["month"] = df[date_col].dt.month

    if mode == "robust":
        stats = (
            df.groupby("month")[value_col]
              .agg(month_mean=lambda s: np.median(s),
                   month_std=lambda s: _mad_std(np.asarray(s, dtype=float)),
                   month_n="size")
              .reset_index()
        )
    else:
        stats = (
            df.groupby("month")[value_col]
              .agg(month_mean="mean",
                   month_std=lambda s: s.std(ddof=1),
                   month_n="size")
              .reset_index()
        )

    stats["enough_data"] = stats["month_n"] >= min_bucket
    # keep stats only where reliable; else NaN so we won't flag on them
    stats["month_mean"] = stats["month_mean"].where(stats["enough_data"], np.nan)
    stats["month_std"]  = stats["month_std"].where(stats["enough_data"],  np.nan)

    return df.merge(stats, on="month", how="left")


def compute_rolling_baseline(
    df, date_col, value_col, window_days=365, robust=False, min_hist=10
) -> pd.DataFrame:
  
    df = df.sort_values(date_col).copy()
    dt = df[date_col]

    expected = np.full(len(df), np.nan, dtype=float)
    spread   = np.full(len(df), np.nan, dtype=float)
    counts   = np.zeros(len(df), dtype=int)


    j = 0
    for i in range(len(df)):
        cutoff = dt.iloc[i] - pd.Timedelta(days=window_days)
        # advance j while history is too old
        while j < i and dt.iloc[j] <= cutoff:
            j += 1
        if i - j >= min_hist:
            hist = df.iloc[j:i][value_col].astype(float)
            counts[i] = len(hist)
            if robust:
                mu = np.median(hist)
                sd = _mad_std(hist.values)
            else:
                mu = hist.mean()
                sd = hist.std(ddof=1)
            expected[i] = mu
            spread[i]   = sd

    df["month_mean"]  = expected     # reuse names to keep detect() unchanged
    df["month_std"]   = spread
    df["month_n"]     = counts
    df["enough_data"] = (~pd.isna(df["month_mean"])) & (~pd.isna(df["month_std"]))
    return df


def detect_anomalies(df, date_col, value_col, z_threshold) -> pd.DataFrame:
  
    df = df.copy()

    std_safe = np.where(
        df["enough_data"].fillna(False),
        df["month_std"].replace(0.0, EPS),
        np.nan,
    )

    overall_median = df[value_col].median()
    expected = df["month_mean"].copy().fillna(overall_median)

    df["z_score"] = (df[value_col] - expected) / std_safe
    df["is_anomaly"] = df["enough_data"].fillna(False) & (df["z_score"].abs() > z_threshold)

    df["expected"]   = expected
    df["abs_diff"]   = df[value_col] - df["expected"]
    df["confidence"] = np.where(df["enough_data"].fillna(False), "high", "low")

    return df


def apply_persistence(df, date_col, z_threshold, w=3, k=2) -> pd.DataFrame:
  

    df = df.sort_values(date_col).copy()
    base_flag = (df["enough_data"].fillna(False)) & (df["z_score"].abs() > z_threshold)
    # rolling sum over boolean -> counts in window
    roll = base_flag.rolling(window=w, min_periods=1).sum()
    df["is_anomaly_persist"] = base_flag & (roll >= k)
    return df


def auto_tau_for_budget(df, date_col, budget_per_month):
   
    elig = df[df["enough_data"].fillna(False) & df["z_score"].abs().notna()].copy()
    if elig.empty:
        return None, 0, 0, None

    ym = elig[date_col].dt.to_period("M")
    months = int(ym.nunique())
    if months <= 0:
        return None, 0, 0, None

    target_total = budget_per_month * months
    zabs = np.sort(elig["z_score"].abs().values)
    n = len(zabs)
    if n == 0 or target_total <= 0:
        return None, months, n, None

    q = float(np.clip(1.0 - (target_total / max(n, 1)), 0.0, 1.0))
    tau = float(np.quantile(zabs, q))
    return tau, months, n, target_total


def compute_episodes(df, date_col, flag_col) -> pd.DataFrame:
    
    if df.empty or df[flag_col].sum() == 0:
        return pd.DataFrame(columns=["start","end","days","count","total_abs_dev","net_dev","peak_|z|"])

    d = df.sort_values(date_col).loc[:, [date_col, flag_col, "abs_diff", "z_score"]].copy()
    d["date_norm"] = d[date_col].dt.normalize()

    # Keep only flagged rows for episode stats
    flagged = d[d[flag_col]].copy()
    flagged["gap"] = (flagged["date_norm"].diff().dt.days != 1).fillna(True).cumsum()

    ep = (
        flagged.groupby("gap")
        .agg(
            start=(date_col, "min"),
            end=(date_col, "max"),
            days=("date_norm", lambda s: int((s.max() - s.min()).days) + 1),
            count=(flag_col, "size"),
            total_abs_dev=("abs_diff", lambda s: float(np.abs(s).sum())),
            net_dev=("abs_diff", lambda s: float(s.sum())),
            peak_abs_z=("z_score", lambda s: float(np.abs(s).max())),
        )
        .rename(columns={"peak_abs_z": "peak_|z|"})
        .sort_values(["total_abs_dev", "days", "peak_|z|"], ascending=False)
        .reset_index(drop=True)
    )
    return ep



def compute_calendar_matrix(df, date_col, values, agg="mean"):
  
    d = df.copy()
    d["month"] = d[date_col].dt.month
    d["day"]   = d[date_col].dt.day
    d["val"]   = values

    mat = np.full((12, 31), np.nan, dtype=float)
    for m in range(1, 13):
        dm = d[d["month"] == m]
        if dm.empty:
            continue
        for day in range(1, 32):
            sel = dm[dm["day"] == day]["val"].dropna()
            if len(sel) == 0:
                continue
            mat[m-1, day-1] = sel.mean() if agg == "mean" else sel.median()
    return mat


def plot_calendar_heatmap(matrix, title, cbar_label):
  
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d9d9d9") 

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Month")
    ax.set_xticks(np.arange(0, 31, 2))
    ax.set_xticklabels([str(i) for i in range(1, 32, 2)])
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig





def build_anomaly_report(df, date_col, value_col, flag_col="is_anomaly") -> pd.DataFrame:
    cols = [date_col, value_col, "expected", "month_std", "z_score", "abs_diff", "confidence", "month_n"]
    if "impact" in df.columns:
        cols.append("impact")
    cols = [c for c in cols if c in df.columns]
    rep = df.loc[df[flag_col], cols].copy()

    # Sort by impact desc (if present) else by |z| desc
    if "impact" in rep.columns:
        rep = rep.sort_values(["impact", "z_score"], ascending=[False, False], key=lambda s: s if s.name != "z_score" else s.abs())
    else:
        rep = rep.sort_values("z_score", ascending=False, key=lambda s: s.abs())

    rep = rep.reset_index(drop=True)
    return rep


def plot_series(df, date_col, value_col, z_used, flag_col="is_anomaly"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[date_col], df[value_col], linewidth=1.0, label="Sales")
    ax.plot(df[date_col], df["expected"], linewidth=1.0, alpha=0.6, label="Baseline (expected)")
    anom = df[df[flag_col]]
    if not anom.empty:
        ax.scatter(anom[date_col], anom[value_col], s=36, label="Anomaly")
    ax.set_title(f"Sales Over Time with Anomalies (|z| > {z_used:.2f})")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.legend()
    fig.tight_layout()
    return fig



st.title("Sales Anomaly Detector")

with st.sidebar:
    st.header("1) Upload data")
    up = st.file_uploader("Choose your CSV or Excel", type=["csv", "xlsx", "xls"])
    st.caption("Your file must include a date column and a numeric sales column.")

    st.header("2) File format")
    delimiter_choice = st.selectbox(
        "CSV delimiter",
        ["auto", ", (comma)", "; (semicolon)", "\\t (tab)"],
        index=0
    )

if up is None:
    st.info("Upload a CSV/Excel file to begin.")
    st.stop()


try:
    preview = read_uploaded_table(up, nrows=200, delimiter_choice=delimiter_choice)
except Exception as e:
    st.error(f"Could not preview the file: {e}")
    st.stop()

# Guess column names
date_guess = next((c for c in preview.columns if "date" in c.lower()), preview.columns[0])
value_guess = next((c for c in preview.columns if "sale" in c.lower()), preview.columns[-1])

with st.sidebar:
    st.header("3) Columns")
    date_col = st.selectbox("Date column", options=list(preview.columns),
                            index=list(preview.columns).index(date_guess))
    value_col = st.selectbox("Sales column", options=list(preview.columns),
                             index=list(preview.columns).index(value_guess))

    st.header("4) Baseline")
    family = st.selectbox("Baseline family", ["Monthly", "Rolling"], index=0)
    baseline_mode = st.selectbox("Baseline type", ["mean/std", "robust (median/MAD)"], index=0)

    if family == "Monthly":
        min_bucket  = st.slider("Min rows per month (gate)", 3, 60, 10, 1)
    else:
        window_days = st.slider("Rolling window (days)", 60, 720, 365, 15)
        min_hist    = st.slider("Min history points (gate)", 5, 60, 10, 1)

    st.header("5) Detection")
    z_threshold = st.slider("Z-score threshold τ (base)", 1.0, 4.0, 2.0, 0.1)

    # Alert budget τ (auto)
    use_budget = st.checkbox("Use alert budget (auto-τ)", value=False)
    budget     = st.slider("Target alerts / month", 2, 50, 10, 1, disabled=not use_budget)

    st.header("6) Persistence (noise control)")
    use_persist = st.checkbox("Require persistence", value=False)
    w = st.slider("Window (days)", 2, 10, 3, 1, disabled=not use_persist)
    k = st.slider("Min anomalous days in window", 1, 5, 2, 1, disabled=not use_persist)

    st.header("7) Suppress known dates")
    known = st.file_uploader("Upload dates to ignore (one per line)", type=["txt", "csv"], key="ignore_dates")

    # Cost-aware ranking
    st.header("8) Cost model (optional)")
    use_cost = st.checkbox("Enable cost-aware ranking", value=False)
    price_per_unit = st.number_input("Price per sales unit", min_value=0.0, value=1.0, step=0.1, disabled=not use_cost)
    margin_pct = st.slider("Gross margin (%)", 0, 100, 100, 1, disabled=not use_cost)

    run_btn = st.button("Run Detector", type="primary")


up.seek(0)

if not run_btn:
    st.write("Configure parameters in the sidebar, then click **Run Detector**.")
    st.stop()


try:
    df = load_data_from_upload(up, date_col, value_col, delimiter_choice=delimiter_choice)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

# Baseline
robust = baseline_mode.startswith("robust")
if family == "Monthly":
    mode_key = "robust" if robust else "mean"
    df_b = compute_monthly_baseline(df, date_col, value_col, min_bucket=min_bucket, mode=mode_key)
else:
    df_b = compute_rolling_baseline(df, date_col, value_col, window_days=window_days, robust=robust, min_hist=min_hist)

# Detect with base τ
df_out = detect_anomalies(df_b, date_col, value_col, z_threshold=z_threshold)

# Suppress known dates (ignore list)
if known is not None:
    try:
        kdf = pd.read_csv(known, header=None, names=["Date"])
    except Exception:
        known.seek(0)
        try:
            kdf = pd.read_csv(known, header=None, names=["Date"], sep=";")
        except Exception:
            known.seek(0)
            kdf = pd.read_csv(known, header=None, names=["Date"], sep="\t")
    kdf["Date"] = pd.to_datetime(kdf["Date"], errors="coerce")
    ignore_set = set(kdf["Date"].dropna().dt.normalize())
    df_out["ignore"] = df_out[date_col].dt.normalize().isin(ignore_set)
    df_out.loc[df_out["ignore"], "z_score"] = np.nan
    df_out.loc[df_out["ignore"], "is_anomaly"] = False

# Auto-τ (alert budget)
tau_used = float(z_threshold)
months_used = None
eligible_n = None
target_total = None
if use_budget:
    tau_auto, months_used, eligible_n, target_total = auto_tau_for_budget(df_out, date_col, budget_per_month=budget)
    if tau_auto is not None and np.isfinite(tau_auto):
        tau_used = float(tau_auto)
        # recompute base flags with auto τ
        df_out["is_anomaly"] = df_out["enough_data"].fillna(False) & (df_out["z_score"].abs() > tau_used)

# Persistence (optional)
if use_persist:
    df_out = apply_persistence(df_out, date_col, tau_used, w=w, k=k)
    flag_col = "is_anomaly_persist"
else:
    flag_col = "is_anomaly"

# Cost-aware impact
if use_cost:
    margin = float(margin_pct) / 100.0
    df_out["impact"] = (df_out["abs_diff"].abs() * float(price_per_unit) * margin).astype(float)
else:
    if "impact" in df_out.columns:
        df_out.drop(columns=["impact"], inplace=True)


A = int(df_out[flag_col].sum())
N = int(df_out.shape[0])
P = round(100.0 * A / N, 2) if N else 0.0
H = int((df_out[flag_col] & (df_out["abs_diff"] > 0)).sum())
L = int((df_out[flag_col] & (df_out["abs_diff"] < 0)).sum())

st.subheader("Results")
policy_note = (
    f"Monthly gate: n ≥ {min_bucket}" if family == "Monthly"
    else f"Rolling gate: history ≥ {min_hist} over {window_days} days"
)
auto_note = ""
if use_budget and months_used is not None:
    auto_note = (
        f"\n- **Auto-τ:** ≈ **{tau_used:.2f}** targeting ~{budget}/month "
        f"(eligible={eligible_n}, months={months_used}, target_total≈{target_total})"
    )
st.markdown(
    f"""
- **Rows (days):** **{N}**  
- **Total anomalies:** **{A}** (**{P}%**)  
- **Direction mix:** **{H}** High spikes, **{L}** Low dips  
- **Baseline:** **{family}** ({'robust' if robust else 'mean/std'}) — {policy_note}  
- **Detection:** |z| > **{tau_used:.2f}** {("(with persistence: ≥" + str(k) + " in last " + str(w) + " days)") if use_persist else ""}  
{auto_note}
"""
)



fig = plot_series(df_out, date_col, value_col, tau_used, flag_col=flag_col)
st.pyplot(fig, clear_figure=True)

# Download chart
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
st.download_button(" Download chart (PNG)", data=buf.getvalue(), file_name="sales_anomalies.png", mime="image/png")



st.subheader("Top 5 Anomalies")
table_cols = [date_col, value_col, "expected", "month_std", "z_score", "abs_diff"]
if "impact" in df_out.columns:
    table_cols.append("impact")

top = (
    df_out.assign(Month=df_out[date_col].dt.month)
          .loc[df_out[flag_col], table_cols + ["Month"]]
          .rename(columns={
              date_col: "Date",
              value_col: "Sales",
              "expected": "Mean (μ_m)",
              "month_std": "Std (σ_m)",
              "z_score": "z",
              "abs_diff": "Deviation (Δ)"
          })
)

if not top.empty:
    top["Direction"] = np.where(top["Deviation (Δ)"] >= 0, "High", "Low")
    top["|z|"] = top["z"].abs()
    if "impact" in top.columns:
        top5 = top.sort_values(["impact", "|z|"], ascending=[False, False]).head(5)
    else:
        top5 = top.sort_values("|z|", ascending=False).head(5)
    st.dataframe(top5, use_container_width=True)
else:
    st.info("No anomalies detected at the current settings.")



st.subheader("Top Episodes (consecutive anomaly windows)")
episodes = compute_episodes(df_out, date_col, flag_col=flag_col)
if not episodes.empty:
    st.dataframe(episodes.head(5), use_container_width=True)
else:
    st.caption("No multi-day episodes under current settings.")


st.subheader("Calendar Heatmap (Month × Day)")
heat_mode = st.selectbox("Heatmap value", ["Anomaly rate", "Mean |Δ|"], index=0)
if N > 0:
    if heat_mode == "Anomaly rate":
        vals = df_out[flag_col].astype(float)  # 1 for anomaly, 0 otherwise
        mat = compute_calendar_matrix(df_out, date_col, values=vals, agg="mean")
        fig_h = plot_calendar_heatmap(mat, "Anomaly Rate by Month × Day", "Rate (0–1)")
    else:
        vals = df_out["abs_diff"].abs()
        mat = compute_calendar_matrix(df_out, date_col, values=vals, agg="mean")
        fig_h = plot_calendar_heatmap(mat, "Mean |Δ| by Month × Day", "Mean |Δ|")
    st.pyplot(fig_h, clear_figure=True)
else:
    st.caption("Upload and run the detector to see the heatmap.")


st.caption(
    "Baselines: Monthly (mean/std or robust median/MAD) or Rolling (trailing window). "
    "Flags only where the gate is satisfied. Expected falls back to global median for plotting when a bucket is sparse. "
    "Auto-τ targets an alert budget; Episodes cluster consecutive alerts; Cost-aware ranking prioritizes by estimated impact. "
    "Calendar Heatmap reveals recurring patterns; Nearest Neighbors give quick 'why' clues."
)
