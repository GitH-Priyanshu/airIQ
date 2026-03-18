"""
EDA Script - Converted from eda.ipynb
Air Quality Index (AQI) Prediction & Analysis
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend so plots save to files
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

# ── helpers ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PLOT_DIR   = os.path.join(PROJECT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CSV_PATH = os.path.join(PROJECT_DIR, "data", "clean_air_quality.csv")

def save_plot(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> Plot saved: {path}")

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)
df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")
print()
print(df.head())
print()
print(df.info())
print()

# ── 2. Basic stats ───────────────────────────────────────────────────────────
print("=" * 70)
print("2. COLUMNS & MISSING VALUES")
print("=" * 70)
print(df.columns.tolist())
print()
print(df.isnull().sum())
print()

# Drop rows with any missing values (if any)
df_clean = df.dropna().copy()
print(f"Shape after dropna: {df_clean.shape}")
print()

# ── 3. PM2.5 Regression ─────────────────────────────────────────────────────
print("=" * 70)
print("3. PM2.5 REGRESSION  (target = PM2.5)")
print("=" * 70)

y_pm = df_clean["PM2.5"]
X_pm = df_clean[["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]]

X_train, X_test, y_train, y_test = train_test_split(
    X_pm, y_pm, test_size=0.2, random_state=42
)
print(f"Training size: {X_train.shape}")
print(f"Testing size : {X_test.shape}")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\n--- Linear Regression (PM2.5) ---")
print(f"MAE : {mean_absolute_error(y_test, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
print(f"R²  : {r2_score(y_test, y_pred_lr):.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n--- Random Forest (PM2.5) ---")
print(f"MAE : {mean_absolute_error(y_test, y_pred_rf):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
print(f"R²  : {r2_score(y_test, y_pred_rf):.4f}")
print()

# ── 4. AQI Regression ───────────────────────────────────────────────────────
print("=" * 70)
print("4. AQI REGRESSION  (target = AQI)")
print("=" * 70)

# The CSV already has AQI column; re-derive if you want to verify:
def calculate_aqi_pm25(pm):
    if pm <= 30:
        BP_lo, BP_hi = 0, 30; I_lo, I_hi = 0, 50
    elif pm <= 60:
        BP_lo, BP_hi = 31, 60; I_lo, I_hi = 51, 100
    elif pm <= 90:
        BP_lo, BP_hi = 61, 90; I_lo, I_hi = 101, 200
    elif pm <= 120:
        BP_lo, BP_hi = 91, 120; I_lo, I_hi = 201, 300
    elif pm <= 250:
        BP_lo, BP_hi = 121, 250; I_lo, I_hi = 301, 400
    else:
        BP_lo, BP_hi = 251, 500; I_lo, I_hi = 401, 500
    return round(((I_hi - I_lo) / (BP_hi - BP_lo)) * (pm - BP_lo) + I_lo)

# Use AQI from the CSV directly
y_aqi = df_clean["AQI"]
X_aqi = df_clean[["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]]

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_aqi, y_aqi, test_size=0.2, random_state=42
)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train_a, y_train_a)
y_pred_aqi_lr = model_lr.predict(X_test_a)
print("\n--- Linear Regression (AQI) ---")
print(f"MAE : {mean_absolute_error(y_test_a, y_pred_aqi_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_a, y_pred_aqi_lr)):.4f}")
print(f"R²  : {r2_score(y_test_a, y_pred_aqi_lr):.4f}")

# Comparison table (first 10)
comparison = pd.DataFrame({
    "Actual AQI":    y_test_a.values[:10],
    "Predicted AQI": np.round(y_pred_aqi_lr[:10], 2),
})
print("\nActual vs Predicted (first 10):")
print(comparison.to_string(index=False))

# Actual vs Predicted scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test_a, y_pred_aqi_lr)
ax.set_xlabel("Actual AQI")
ax.set_ylabel("Predicted AQI")
ax.set_title("Actual vs Predicted AQI (Linear Regression)")
save_plot(fig, "actual_vs_predicted_aqi.png")

# ── 5. Gradient Boosting ────────────────────────────────────────────────────
print("\n--- Gradient Boosting (AQI, max_depth=3) ---")
gbr = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gbr.fit(X_train_a, y_train_a)
y_pred_gbr = gbr.predict(X_test_a)
print(f"MAE : {mean_absolute_error(y_test_a, y_pred_gbr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_a, y_pred_gbr)):.4f}")
print(f"R²  : {r2_score(y_test_a, y_pred_gbr):.4f}")

# Feature importance
importance = gbr.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": X_aqi.columns,
    "Importance": importance,
}).sort_values(by="Importance", ascending=False)
print("\nFeature Importance (Gradient Boosting):")
print(feat_imp.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(feat_imp["Feature"], feat_imp["Importance"])
ax.invert_yaxis()
ax.set_title("Feature Importance (Gradient Boosting)")
save_plot(fig, "feature_importance.png")

# ── 6. Reduced Feature Model ────────────────────────────────────────────────
print("\n--- Reduced Feature Gradient Boosting (CO, NO2, OZONE, PM10) ---")
X_red = df_clean[["CO", "NO2", "OZONE", "PM10"]]
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_red, y_aqi, test_size=0.2, random_state=42
)
gbr_red = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gbr_red.fit(X_tr_r, y_tr_r)
y_pred_r = gbr_red.predict(X_te_r)
print(f"MAE : {mean_absolute_error(y_te_r, y_pred_r):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_te_r, y_pred_r)):.4f}")
print(f"R²  : {r2_score(y_te_r, y_pred_r):.4f}")

# ── 7. Hybrid Model ─────────────────────────────────────────────────────────
print("\n--- Hybrid Model (LR + GBR average) ---")
y_pred_hybrid = (y_pred_aqi_lr + y_pred_gbr) / 2
print(f"MAE : {mean_absolute_error(y_test_a, y_pred_hybrid):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_a, y_pred_hybrid)):.4f}")
print(f"R²  : {r2_score(y_test_a, y_pred_hybrid):.4f}")

# Model comparison bar chart
models   = ["Linear Regression", "Gradient Boosting", "Hybrid Model"]
r2_scores = [
    r2_score(y_test_a, y_pred_aqi_lr),
    r2_score(y_test_a, y_pred_gbr),
    r2_score(y_test_a, y_pred_hybrid),
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(models, r2_scores)
ax.set_ylabel("R² Score")
ax.set_title("Model Comparison (R² Score)")
ax.set_ylim(0.6, 0.85)
plt.xticks(rotation=20)
save_plot(fig, "model_comparison.png")

# ── 8. Cross Validation ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. CROSS-VALIDATION (5-fold)")
print("=" * 70)
X_final = df_clean[["CO", "NO2", "OZONE", "PM10"]]
y_final = df_clean["AQI"]
cv_scores = cross_val_score(LinearRegression(), X_final, y_final, cv=5, scoring="r2")
print(f"LR CV R² scores: {cv_scores}")
print(f"Average R²     : {np.mean(cv_scores):.4f}")

# ── 9. AQI Classification ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. AQI CLASSIFICATION")
print("=" * 70)

def categorize_aqi(aqi):
    if aqi <= 50:   return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else:            return "Severe"

# Use existing AQI_Category or derive it
if "AQI_Category" not in df_clean.columns:
    df_clean["AQI_Category"] = df_clean["AQI"].apply(categorize_aqi)

print(df_clean[["AQI", "AQI_Category"]].head())

X_cls = df_clean[["CO", "NO2", "OZONE", "PM10"]]
y_cls = df_clean["AQI_Category"]
X_tc, X_ec, y_tc, y_ec = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_tc, y_tc)
y_pc = clf.predict(X_ec)
print(f"\nAccuracy: {accuracy_score(y_ec, y_pc):.4f}")
print("\nClassification Report:")
print(classification_report(y_ec, y_pc, zero_division=0))

print("\n--- Balanced class weights ---")
clf2 = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
clf2.fit(X_tc, y_tc)
y_pc2 = clf2.predict(X_ec)
print(f"Accuracy: {accuracy_score(y_ec, y_pc2):.4f}")
print(classification_report(y_ec, y_pc2, zero_division=0))

# ── 10. AdaBoost & GBR (all 6 features) ─────────────────────────────────────
print("=" * 70)
print("7. ADDITIONAL MODELS (6 features → AQI)")
print("=" * 70)

X_all = df_clean[["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]]
y_all = df_clean["AQI"]
X_tr6, X_te6, y_tr6, y_te6 = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

ada = AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
ada.fit(X_tr6, y_tr6)
y_pred_ada = ada.predict(X_te6)
print("\n--- AdaBoost ---")
print(f"MAE : {mean_absolute_error(y_te6, y_pred_ada):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_te6, y_pred_ada)):.4f}")
print(f"R²  : {r2_score(y_te6, y_pred_ada):.4f}")

gbr4 = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
)
gbr4.fit(X_tr6, y_tr6)
y_pred_gbr4 = gbr4.predict(X_te6)
print("\n--- Gradient Boosting (max_depth=4) ---")
print(f"MAE : {mean_absolute_error(y_te6, y_pred_gbr4):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_te6, y_pred_gbr4)):.4f}")
print(f"R²  : {r2_score(y_te6, y_pred_gbr4):.4f}")

print("\n" + "=" * 70)
print("DONE — all cells executed successfully!")
print("=" * 70)
