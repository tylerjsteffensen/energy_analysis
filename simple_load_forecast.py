"""
Simple Load Forecast Model
===========================
Forecasts hourly electricity demand using:
  - Temperature (°F)
  - Humidity (%)
  - Historical load (lag features)
  - Calendar features (hour-of-day, day-of-week)

This is a straightforward OLS + Ridge regression approach — no complex
time-series machinery. Easy to understand, explain, and maintain.

Usage:
    python simple_load_forecast.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import os


# ──────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────

def load_caiso_load(filepath='1_month_CAISO_load.csv', tac='CA ISO-TAC'):
    """Load CAISO hourly load data."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'], utc=True)
    df = df[df['TAC_AREA_NAME'] == tac].set_index('timestamp').sort_index()
    hourly = df[['MW']].resample('h').mean().rename(columns={'MW': 'load_mw'})
    return hourly.dropna()


def load_weather(filepath='weather_data.csv'):
    """
    Load hourly weather data.

    Expected CSV columns:
        timestamp,temp_f,humidity_pct
        2025-12-01T00:00:00+00:00,55.2,62.0
        ...
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()
    return df[['temp_f', 'humidity_pct']]


def generate_sample_weather(load_index):
    """
    Generate realistic synthetic weather for SCE territory (Dec 2025).
    Winter in Southern California: ~45-70°F, 30-80% humidity.
    """
    np.random.seed(42)
    n = len(load_index)
    hour = load_index.hour

    # Temperature: cool at night, warm in afternoon
    temp_base = 55 + 10 * np.sin(2 * np.pi * (hour - 6) / 24)
    temp_f = temp_base + np.random.normal(0, 3, n)

    # Humidity: inverse of temperature (higher at night)
    humidity_base = 60 - 15 * np.sin(2 * np.pi * (hour - 6) / 24)
    humidity_pct = np.clip(humidity_base + np.random.normal(0, 5, n), 15, 95)

    weather = pd.DataFrame({
        'temp_f': temp_f.round(1),
        'humidity_pct': humidity_pct.round(1),
    }, index=load_index)

    weather.to_csv('weather_data.csv', index_label='timestamp')
    print("Generated sample weather_data.csv")
    return weather


# ──────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────

def build_features(load_df, weather_df):
    """
    Merge load + weather and create features for the regression model.

    Features:
        - temp_f, humidity_pct          (raw weather)
        - cdd, hdd                      (cooling/heating degree days, base 65°F)
        - temp_x_humidity               (heat index proxy)
        - hour sin/cos                  (diurnal cycle)
        - is_weekend                    (weekday vs weekend)
        - load_lag_24, load_lag_168     (same hour yesterday / last week)
        - load_rolling_24               (trailing 24h average)
    """
    df = load_df.join(weather_df, how='inner')

    # Weather-derived
    df['cdd'] = np.maximum(df['temp_f'] - 65, 0)
    df['hdd'] = np.maximum(65 - df['temp_f'], 0)
    df['temp_x_humidity'] = df['temp_f'] * df['humidity_pct'] / 100

    # Calendar
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Lag features (historical load)
    df['load_lag_24'] = df['load_mw'].shift(24)
    df['load_lag_168'] = df['load_mw'].shift(168)
    df['load_rolling_24'] = df['load_mw'].rolling(24).mean()

    return df.dropna()


# ──────────────────────────────────────────────────────────────────────
# 3. MODEL
# ──────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'temp_f', 'humidity_pct', 'cdd', 'hdd', 'temp_x_humidity',
    'hour_sin', 'hour_cos', 'is_weekend',
    'load_lag_24', 'load_lag_168', 'load_rolling_24',
]


def train_model(df, feature_cols=FEATURE_COLS, train_frac=0.8):
    """
    Train a Ridge regression model and evaluate on a holdout set.

    Returns: model, scaler, metrics dict, train/test DataFrames
    """
    split = int(len(df) * train_frac)
    train = df.iloc[:split]
    test = df.iloc[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    y_train = train['load_mw'].values
    y_test = test['load_mw'].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'n_train': len(train),
        'n_test': len(test),
    }

    # Feature importance (standardized coefficients)
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_,
        'abs_importance': np.abs(model.coef_),
    }).sort_values('abs_importance', ascending=False)

    test['forecast_mw'] = test_pred

    return model, scaler, metrics, coef_df, train, test


# ──────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────

def plot_results(test_df, coef_df, metrics):
    """Generate forecast vs actual plot and feature importance chart."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # --- Top: Forecast vs Actual ---
    ax = axes[0]
    ax.plot(test_df.index, test_df['load_mw'], label='Actual', linewidth=1.2)
    ax.plot(test_df.index, test_df['forecast_mw'], label='Forecast',
            linewidth=1.2, linestyle='--', alpha=0.8)
    ax.fill_between(
        test_df.index,
        test_df['forecast_mw'] - metrics['test_mae'],
        test_df['forecast_mw'] + metrics['test_mae'],
        alpha=0.15, label=f'±MAE ({metrics["test_mae"]:.0f} MW)',
    )
    ax.set_title(f'Simple Load Forecast — Test Set '
                 f'(MAPE: {metrics["test_mape"]:.2f}%, '
                 f'MAE: {metrics["test_mae"]:.0f} MW)')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom: Feature Importance ---
    ax2 = axes[1]
    colors = ['#1976D2' if c > 0 else '#E53935' for c in coef_df['coefficient']]
    ax2.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
    ax2.set_title('Feature Importance (Ridge Coefficients, standardized)')
    ax2.set_xlabel('Coefficient')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_simple_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("→ Saved: output_simple_forecast.png")


def plot_weather_vs_load(df):
    """Scatter plots showing weather-load relationships."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(df['temp_f'], df['load_mw'], s=8, alpha=0.4)
    axes[0].set_xlabel('Temperature (°F)')
    axes[0].set_ylabel('Load (MW)')
    axes[0].set_title('Load vs Temperature')

    axes[1].scatter(df['humidity_pct'], df['load_mw'], s=8, alpha=0.4, color='teal')
    axes[1].set_xlabel('Humidity (%)')
    axes[1].set_ylabel('Load (MW)')
    axes[1].set_title('Load vs Humidity')

    axes[2].scatter(df['temp_x_humidity'], df['load_mw'], s=8, alpha=0.4, color='coral')
    axes[2].set_xlabel('Temp × Humidity / 100')
    axes[2].set_ylabel('Load (MW)')
    axes[2].set_title('Load vs Heat Index Proxy')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_weather_vs_load.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("→ Saved: output_weather_vs_load.png")


# ──────────────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  SIMPLE LOAD FORECAST MODEL")
    print("  Inputs: Temperature, Humidity, Historical Load")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    load_df = load_caiso_load()
    print(f"   Load: {len(load_df)} hours "
          f"({load_df.index.min().date()} → {load_df.index.max().date()})")

    if os.path.exists('weather_data.csv'):
        weather_df = load_weather()
        print(f"   Weather: {len(weather_df)} hours (from weather_data.csv)")
    else:
        print("   weather_data.csv not found — generating sample data...")
        weather_df = generate_sample_weather(load_df.index)

    # Build features
    print("\n2. Building features...")
    df = build_features(load_df, weather_df)
    print(f"   {len(df)} usable observations, {len(FEATURE_COLS)} features:")
    for f in FEATURE_COLS:
        print(f"     - {f}")

    # Train model
    print("\n3. Training Ridge regression (80/20 split)...")
    model, scaler, metrics, coef_df, train, test = train_model(df)

    print(f"\n   RESULTS")
    print(f"   ───────────────────────────────")
    print(f"   Training:  {metrics['n_train']} hours, "
          f"MAE = {metrics['train_mae']:.0f} MW")
    print(f"   Test:      {metrics['n_test']} hours, "
          f"MAE = {metrics['test_mae']:.0f} MW, "
          f"MAPE = {metrics['test_mape']:.2f}%")

    print(f"\n   Feature importance (|coefficient|):")
    for _, row in coef_df.iterrows():
        direction = "+" if row['coefficient'] > 0 else "−"
        print(f"     {direction} {row['feature']:20s}  {row['abs_importance']:.1f}")

    # Plots
    print("\n4. Generating plots...")
    plot_results(test, coef_df, metrics)
    plot_weather_vs_load(df)

    print("\nDone. Output files:")
    print("  - output_simple_forecast.png  (forecast vs actual + feature importance)")
    print("  - output_weather_vs_load.png  (weather-load scatter plots)")
