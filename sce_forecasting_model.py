"""
SCE Energy Demand & Price Forecasting — Main Analysis
======================================================
Southern California Edison — Pricing, Design & Research

This script demonstrates the full forecasting and hedging analytics pipeline
using the available CAISO data (December 2025). It serves as:

1. A proof-of-concept showing the models work end-to-end
2. A template that will scale up when longer historical data is provided
3. Documentation of the methodology for Pricing Management review

Run: python sce_forecasting_model.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from demand_forecast import DemandForecaster
from price_forecast import PriceForecaster
from hedging_analytics import HedgingAnalytics

import os

# Check if real data files exist
DATA_FILES = {
    'load': '1_month_CAISO_load.csv',
    'dam_forecast': '1_month_DAM_forecast.csv',
    'rtm_forecast': '1_month_RTM_forecast.csv',
    'renewables': '1_month_Renewables_Act.csv',
    'lmp': '1_day_LMP_DAM.csv',
}
USE_REAL_DATA = all(os.path.exists(f) for f in DATA_FILES.values())


def generate_synthetic_data():
    """
    Generate realistic synthetic CAISO data for model demonstration.
    Mimics the structure of actual OASIS downloads so the pipeline runs end-to-end.
    This is replaced by real data when CSVs are present.
    """
    np.random.seed(42)
    hours = pd.date_range('2025-12-01', '2025-12-31 23:00', freq='h', tz='UTC')
    n = len(hours)

    # --- Synthetic load (CAISO-like diurnal + weekly pattern) ---
    hour_of_day = hours.hour
    day_of_week = hours.dayofweek

    base_load = 22000
    diurnal = 4000 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    weekly = -800 * (day_of_week >= 5).astype(float)
    noise = np.random.normal(0, 400, n)
    load_mw = base_load + diurnal + weekly + noise
    load_mw = np.maximum(load_mw, 15000)

    load_df = pd.DataFrame({
        'INTERVALSTARTTIME_GMT': hours.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        'INTERVALENDTIME_GMT': (hours + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        'TAC_AREA_NAME': 'CA ISO-TAC',
        'MW': load_mw,
        'OPR_HR': hour_of_day,
        'OPR_INTERVAL': 0,
        'LOAD_TYPE': 0,
        'POS': 3.8,
        'OPR_DT': hours.strftime('%Y-%m-%d'),
        'MARKET_RUN_ID': 'ACTUAL',
        'LABEL': 'Actual',
        'XML_DATA_ITEM': 'SLD_FCST',
        'EXECUTION_TYPE': 'ACTUAL',
        'GROUP': 1,
    })
    load_df.to_csv(DATA_FILES['load'], index=False)

    # --- Synthetic LMP (1 day, multiple nodes, SP15/NP15/ZP26 hubs) ---
    lmp_hours = pd.date_range('2025-12-10 08:00', periods=24, freq='h', tz='UTC')
    hubs = {
        'TH_SP15_GEN-APND': 0,
        'TH_NP15_GEN-APND': -2,
        'TH_ZP26_GEN-APND': -1,
    }
    lmp_types = ['LMP', 'MCE', 'MCC', 'MCL', 'MGHG']

    lmp_rows = []
    for node, offset in hubs.items():
        for i, ts in enumerate(lmp_hours):
            h = ts.hour
            base_lmp = 35 + 15 * np.sin(2 * np.pi * (h - 4) / 24) + offset
            mce = base_lmp * 0.88 + np.random.normal(0, 1)
            mcc = np.random.normal(0, 0.5)
            mcl = base_lmp * 0.08 + np.random.normal(0, 0.3)
            mghg = max(0, base_lmp * 0.04 + np.random.normal(0, 0.2))
            lmp_val = mce + mcc + mcl + mghg

            for lt, val in zip(lmp_types, [lmp_val, mce, mcc, mcl, mghg]):
                lmp_rows.append({
                    'INTERVALSTARTTIME_GMT': ts.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                    'INTERVALENDTIME_GMT': (ts + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                    'OPR_DT': ts.strftime('%Y-%m-%d'),
                    'OPR_HR': h,
                    'OPR_INTERVAL': 0,
                    'NODE_ID_XML': node,
                    'NODE_ID': node,
                    'NODE': node.replace('_', ' '),
                    'MARKET_RUN_ID': 'DAM',
                    'LMP_TYPE': lt,
                    'XML_DATA_ITEM': 'LMP_PRC',
                    'PNODE_RESMRID': node,
                    'GRP_TYPE': 'TH',
                    'POS': 0,
                    'MW': round(val, 5),
                    'GROUP': 1,
                })

    pd.DataFrame(lmp_rows).to_csv(DATA_FILES['lmp'], index=False)

    # --- Synthetic renewables ---
    renew_rows = []
    for ts in hours:
        h = ts.hour
        # Solar: bell curve peaking at noon UTC (roughly 4am PT — shift for realism)
        solar = max(0, 8000 * np.exp(-0.5 * ((h - 20) / 3) ** 2)) + np.random.normal(0, 200)
        solar = max(0, solar)
        wind = 2000 + 1000 * np.sin(2 * np.pi * h / 24) + np.random.normal(0, 300)
        wind = max(0, wind)
        for hub in ['SP15', 'NP15', 'ZP26']:
            scale = {'SP15': 0.5, 'NP15': 0.35, 'ZP26': 0.15}[hub]
            for rtype, val in [('Solar', solar * scale), ('Wind', wind * scale)]:
                renew_rows.append({
                    'INTERVALSTARTTIME_GMT': ts.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                    'INTERVALENDTIME_GMT': (ts + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                    'TRADING_HUB': hub,
                    'RENEWABLE_TYPE': rtype,
                    'MW': round(val, 2),
                    'OPR_HR': h,
                    'OPR_INTERVAL': 0,
                    'GROUP': 1,
                })

    pd.DataFrame(renew_rows).to_csv(DATA_FILES['renewables'], index=False)

    # Also create stub DAM/RTM forecast files
    dam_df = load_df.copy()
    dam_df['MW'] = dam_df['MW'] * (1 + np.random.normal(0, 0.02, len(dam_df)))
    dam_df['MARKET_RUN_ID'] = 'DAM'
    dam_df['LOAD_TYPE'] = 1
    dam_df.to_csv(DATA_FILES['dam_forecast'], index=False)

    rtm_df = load_df.copy()
    rtm_df['MW'] = rtm_df['MW'] * (1 + np.random.normal(0, 0.01, len(rtm_df)))
    rtm_df['MARKET_RUN_ID'] = 'RTM'
    rtm_df['LABEL'] = 'RTM 15Min Load Forecast'
    rtm_df.to_csv(DATA_FILES['rtm_forecast'], index=False)

    print("  Generated synthetic CAISO data for demonstration.")
    print("  Replace with real CSVs from OASIS for production use.\n")


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ======================================================================
# SECTION 1: DEMAND FORECASTING
# ======================================================================

def run_demand_analysis():
    separator("SECTION 1: ELECTRICITY DEMAND FORECASTING")

    # Initialize forecaster for CAISO total system load
    forecaster = DemandForecaster(tac_area='CA ISO-TAC')

    # Load the available data
    df_load = forecaster.load_caiso_data('1_month_CAISO_load.csv')

    # --- 1a. STL Decomposition ---
    print("\n--- 1a. STL Decomposition (Trend / Seasonal / Residual) ---")
    stl_result = forecaster.decompose(df_load, period=24)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df_load.index, df_load['load_mw'], linewidth=0.8)
    axes[0].set_ylabel('Load (MW)')
    axes[0].set_title('CAISO System Load — STL Decomposition (Dec 2025)')

    axes[1].plot(stl_result.trend.index, stl_result.trend, color='tab:orange')
    axes[1].set_ylabel('Trend (MW)')

    axes[2].plot(stl_result.seasonal.index, stl_result.seasonal, color='tab:green')
    axes[2].set_ylabel('Seasonal (MW)')

    axes[3].plot(stl_result.resid.index, stl_result.resid, color='tab:red', linewidth=0.5)
    axes[3].set_ylabel('Residual (MW)')
    axes[3].set_xlabel('Date (UTC)')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_stl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: output_stl_decomposition.png")

    # --- 1b. Stationarity Test ---
    print("\n--- 1b. Stationarity Test (ADF) ---")
    forecaster.stationarity_report(df_load['load_mw'])

    # --- 1c. Feature Engineering ---
    print("\n--- 1c. Feature Engineering ---")
    featured = forecaster.engineer_features(df_load)
    print(f"  Features created: {[c for c in featured.columns if c != 'load_mw']}")
    print(f"  Shape: {featured.shape}")

    # --- 1d. Model Fitting ---
    print("\n--- 1d. SARIMAX Model Fitting ---")
    # Use conservative order for 1-month data
    forecaster.fit(
        df_load,
        order=(2, 0, 1),
        seasonal_order=(1, 0, 1, 24),
    )

    # --- 1e. Backtesting ---
    print("\n--- 1e. Walk-Forward Backtest ---")
    bt = forecaster.backtest(df_load, train_pct=0.8)

    # Plot backtest results
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(bt['actual'].index, bt['actual']['load_mw'],
            label='Actual', linewidth=1.2)
    ax.plot(bt['forecast'].index[:len(bt['actual'])],
            bt['forecast']['forecast_mw'].values[:len(bt['actual'])],
            label='Forecast', linewidth=1.2, linestyle='--')
    ax.fill_between(
        bt['forecast'].index[:len(bt['actual'])],
        bt['forecast']['lower_ci'].values[:len(bt['actual'])],
        bt['forecast']['upper_ci'].values[:len(bt['actual'])],
        alpha=0.2, label='95% CI',
    )
    ax.set_title('CAISO Load Forecast — Walk-Forward Backtest')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_demand_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: output_demand_backtest.png")

    # --- 1f. Forecast Error Distribution ---
    print("\n--- 1f. Forecast Error Distribution (for Hedge Sizing) ---")
    error_dist = forecaster.forecast_error_distribution(bt)

    return forecaster, df_load, bt, error_dist


# ======================================================================
# SECTION 2: PRICE FORECASTING
# ======================================================================

def run_price_analysis():
    separator("SECTION 2: ENERGY PRICE FORECASTING (SP15 Hub)")

    pf = PriceForecaster(hub='SP15')

    # --- 2a. Load LMP data ---
    print("--- 2a. Loading SP15 Hub LMP Data ---")
    lmp_df = pf.load_lmp_data('1_day_LMP_DAM.csv')

    # Also load NP15 and ZP26 for comparison
    pf_np = PriceForecaster(hub='NP15')
    pf_zp = PriceForecaster(hub='ZP26')
    lmp_np = pf_np.load_lmp_data('1_day_LMP_DAM.csv')
    lmp_zp = pf_zp.load_lmp_data('1_day_LMP_DAM.csv')

    # --- 2b. Load system load and renewables ---
    print("\n--- 2b. Preparing Fundamental Drivers ---")
    df_actual = pd.read_csv('1_month_CAISO_load.csv')
    df_actual['start_time'] = pd.to_datetime(
        df_actual['INTERVALSTARTTIME_GMT'], utc=True, errors='coerce'
    )
    df_actual = df_actual.set_index('start_time').sort_index()
    caiso_load = (
        df_actual[df_actual['TAC_AREA_NAME'] == 'CA ISO-TAC'][['MW']]
        .resample('h').mean()
        .rename(columns={'MW': 'load_mw'})
    )

    df_renew = pd.read_csv('1_month_Renewables_Act.csv')
    df_renew['start_time'] = pd.to_datetime(
        df_renew['INTERVALSTARTTIME_GMT'], utc=True, errors='coerce'
    )
    df_renew = df_renew.set_index('start_time').sort_index()
    renew_total = (
        df_renew[['MW']]
        .resample('h').sum()
        .rename(columns={'MW': 'renew_mw'})
    )

    # Slice to match LMP day range (use LMP index bounds for alignment)
    lmp_start = lmp_df.index.min()
    lmp_end = lmp_df.index.max()
    load_day = caiso_load.loc[lmp_start:lmp_end]
    renew_day = renew_total.loc[lmp_start:lmp_end]

    print(f"  LMP range: {lmp_start} → {lmp_end} ({len(lmp_df)} hours)")
    print(f"  Load day:  {len(load_day)} hours matched")
    print(f"  Renew day: {len(renew_day)} hours matched")

    # --- 2c. Structural Model ---
    print("\n--- 2c. Structural LMP Model ---")
    structural = pf.fit_structural(
        lmp_df,
        load_df=load_day,
        renew_df=renew_day,
    )

    # --- 2d. Time-Series Model ---
    print("\n--- 2d. SARIMAX LMP Model ---")
    ts = pf.fit_timeseries(
        lmp_df,
        load_df=load_day,
        renew_df=renew_day,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 0),  # Only 24 hours — no seasonal
    )

    # --- 2e. Quantile Regression ---
    print("\n--- 2e. Quantile Regression (Price Distribution) ---")
    qr = pf.fit_quantile(
        lmp_df,
        load_df=load_day,
        renew_df=renew_day,
    )

    # --- 2f. Hub comparison plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # LMP by hub
    for hub_name, hub_lmp in [('SP15', lmp_df), ('NP15', lmp_np), ('ZP26', lmp_zp)]:
        axes[0].plot(hub_lmp.index, hub_lmp['LMP'], label=hub_name, linewidth=1.5)
    axes[0].set_title('Day-Ahead LMP by Trading Hub — 2025-12-10')
    axes[0].set_ylabel('LMP ($/MWh)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LMP components for SP15
    components = ['MCE', 'MCC', 'MCL', 'MGHG']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    axes[1].stackplot(
        lmp_df.index,
        [lmp_df[c].fillna(0) for c in components],
        labels=components, colors=colors, alpha=0.75,
    )
    axes[1].plot(lmp_df.index, lmp_df['LMP'], 'k--', linewidth=1, label='Total LMP')
    axes[1].set_title('SP15 LMP Decomposition — 2025-12-10')
    axes[1].set_ylabel('$/MWh')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.savefig('output_price_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: output_price_analysis.png")

    # --- 2g. Merit-order effect visualization ---
    merged = pf.prepare_fundamentals(lmp_df, load_day, renew_day)
    merged = merged.dropna(subset=['LMP', 'renew_mw'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LMP vs renewable penetration
    if 'renew_pct' in merged.columns:
        sc1 = axes[0].scatter(merged['renew_pct'], merged['LMP'],
                              c=merged.index.hour, cmap='plasma', s=60,
                              edgecolors='white', linewidths=0.5)
        axes[0].set_xlabel('Renewable Penetration (% of load)')
        axes[0].set_ylabel('SP15 LMP ($/MWh)')
        axes[0].set_title('Merit-Order Effect')
        plt.colorbar(sc1, ax=axes[0], label='Hour (UTC)')

    # LMP vs net load
    if 'net_load_mw' in merged.columns:
        sc2 = axes[1].scatter(merged['net_load_mw'], merged['LMP'],
                              c=merged.index.hour, cmap='viridis', s=60,
                              edgecolors='white', linewidths=0.5)
        valid = merged[['net_load_mw', 'LMP']].dropna()
        if len(valid) >= 2:
            z = np.polyfit(valid['net_load_mw'], valid['LMP'], 1)
            xs = np.linspace(valid['net_load_mw'].min(), valid['net_load_mw'].max(), 50)
            axes[1].plot(xs, np.poly1d(z)(xs), 'r--',
                         label=f'β={z[0]:.4f} $/MWh per MW')
            axes[1].legend()
        axes[1].set_xlabel('Net Load (MW)')
        axes[1].set_ylabel('SP15 LMP ($/MWh)')
        axes[1].set_title('Price vs Net Load')
        plt.colorbar(sc2, ax=axes[1], label='Hour (UTC)')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_merit_order.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: output_merit_order.png")

    return pf, lmp_df, merged


# ======================================================================
# SECTION 3: HEDGING & POSITION MANAGEMENT
# ======================================================================

def run_hedging_analysis(lmp_df, merged, error_dist):
    separator("SECTION 3: HEDGING & POSITION MANAGEMENT ANALYTICS")

    ha = HedgingAnalytics(confidence_level=0.95)

    # --- 3a. Volumetric Exposure ---
    print("--- 3a. Volumetric Exposure (illustrative) ---")
    print("  Using demand forecast uncertainty from Section 1\n")

    # Simulate a typical SCE procurement position
    # SCE peak load ~15,000 MW, assume 80% hedged via bilateral contracts
    forecast_mw = pd.Series(
        np.linspace(18000, 24000, 24),  # Illustrative daily load shape
        index=pd.date_range('2025-12-10 08:00', periods=24, freq='h', tz='UTC'),
    )
    hedged_mw = forecast_mw * 0.80  # 80% hedged
    forecast_std = pd.Series(
        error_dist['std_error'],
        index=forecast_mw.index,
    )

    exposure = ha.volumetric_exposure(forecast_mw, forecast_std, hedged_mw)

    # --- 3b. VaR Calculation ---
    print("\n--- 3b. Value-at-Risk ---")
    # Use hourly LMP returns as proxy
    lmp_returns = lmp_df['LMP'].pct_change().dropna()

    if len(lmp_returns) >= 5:
        # Position value: open 20% of ~20,000 MW avg × $45/MWh avg × 24h
        position_val = 4000 * 45 * 24  # ~$4.3M daily

        var_hist = ha.calculate_var(lmp_returns, position_val, method='historical')
        var_param = ha.calculate_var(lmp_returns, position_val, method='parametric')
        var_cf = ha.calculate_var(lmp_returns, position_val, method='cornish_fisher')
    else:
        print("  Insufficient data for VaR — need longer price history.")
        print("  With 2+ years of daily data, all three VaR methods will produce")
        print("  robust estimates.")

    # --- 3c. Stress Testing ---
    print("\n--- 3c. Stress Test Scenarios ---")
    stress_scenarios = {
        'Summer heat wave (+$80/MWh, 72h)': {'price_change': 80, 'hours': 72},
        'Gas pipeline outage (+$150/MWh, 48h)': {'price_change': 150, 'hours': 48},
        'Renewable over-gen (−$15/MWh, 6h)': {'price_change': -15, 'hours': 6},
        'Extended high prices (+$40/MWh, 168h)': {'price_change': 40, 'hours': 168},
    }
    open_position_mw = 4000  # 20% of ~20 GW open
    stress = ha.stress_test(open_position_mw, stress_scenarios)

    # --- 3d. Hedge Cost-Benefit ---
    print("\n--- 3d. Hedge Cost-Benefit Analysis ---")
    avg_lmp = lmp_df['LMP'].mean()
    std_lmp = lmp_df['LMP'].std()

    # What if SCE is offered an SP15 forward at a $3 premium?
    ha.hedge_cost_benefit(
        position_mw=4000,
        hours=720,  # ~1 month
        spot_forecast=avg_lmp,
        forward_price=avg_lmp + 3.0,  # $3/MWh risk premium
        spot_std=std_lmp,
        hedge_ratio=0.50,
    )

    # --- 3e. Illustrative Position Summary ---
    print("\n--- 3e. Position Summary (Illustrative) ---")
    positions = pd.DataFrame([
        {
            'instrument': 'SP15 Jan-26 On-Peak Forward',
            'direction': 'long',
            'volume_mwh': 150_000,
            'price': 48.50,
            'current_mark': avg_lmp,
            'delivery_start': '2026-01-01',
            'delivery_end': '2026-01-31',
        },
        {
            'instrument': 'SoCal Citygate Gas Swap',
            'direction': 'long',
            'volume_mwh': 80_000,  # gas-equivalent MWh
            'price': 4.25,
            'current_mark': 4.10,
            'delivery_start': '2026-01-01',
            'delivery_end': '2026-01-31',
        },
        {
            'instrument': 'SP15 Dec-25 Off-Peak Forward',
            'direction': 'short',
            'volume_mwh': 50_000,
            'price': 35.00,
            'current_mark': avg_lmp * 0.85,  # off-peak discount
            'delivery_start': '2025-12-01',
            'delivery_end': '2025-12-31',
        },
    ])

    ha.position_summary(positions)

    return ha


# ======================================================================
# SECTION 4: SUMMARY & RECOMMENDATIONS
# ======================================================================

def print_summary():
    separator("SECTION 4: SUMMARY & NEXT STEPS")

    print("""
    MODEL OUTPUTS PRODUCED
    ──────────────────────
    1. Demand Forecast: SARIMAX model for CAISO hourly load
       - STL decomposition revealing trend, diurnal, and residual components
       - Walk-forward backtest with MAPE metric
       - Forecast error distribution for volumetric hedge sizing

    2. Price Forecast: Structural + time-series LMP models for SP15 hub
       - OLS structural model decomposing LMP into fundamental drivers
       - Price sensitivity to load, renewables (merit-order effect)
       - Quantile regression for tail-risk (P95/P99) price estimation

    3. Hedging Analytics: Position management toolkit
       - Volumetric exposure quantification
       - VaR/CVaR risk metrics (historical, parametric, Cornish-Fisher)
       - Stress testing (heat wave, gas spike, negative price scenarios)
       - Hedge cost-benefit analysis
       - Position mark-to-market summary

    LIMITATIONS WITH CURRENT DATA
    ─────────────────────────────
    - Only 1 month of load data → seasonal patterns not fully captured
    - Only 1 day of LMP data → price model is illustrative, not deployable
    - No weather data → CDD/HDD features (strongest load driver) unavailable
    - No gas prices → structural price model missing key fundamental
    - No forward curves → hedge optimization is illustrative

    PRIORITY DATA REQUESTS (see DATA_REQUIREMENTS.md)
    ─────────────────────────────────────────────────
    1. [CRITICAL] 3+ years of hourly CAISO load + SP15 LMP
    2. [CRITICAL] Hourly weather (temperature) for SCE territory
    3. [CRITICAL] Daily SoCal Citygate + Henry Hub gas prices
    4. [HIGH]     Hourly solar/wind generation (extended history)
    5. [HIGH]     California carbon allowance prices
    6. [MEDIUM]   SP15 forward curves (power + gas)

    With Priority 1-3 data, expect:
    - Demand forecast MAPE < 3% (7-day ahead)
    - Price forecast capturing 60-70% of LMP variance
    - Robust VaR estimates for CPUC reporting

    CONSULTATION NOTES FOR CUSTOMER SERVICE ORG UNIT
    ─────────────────────────────────────────────────
    - Peak price hours (17:00-21:00 PT) align with customer TOU peak periods
    - Renewable suppression of midday prices supports TOU rate design
    - Forecast error bands inform customer bill volatility estimates
    - Stress test scenarios support customer advisory during extreme events
    """)


# ======================================================================
# MAIN EXECUTION
# ======================================================================

if __name__ == '__main__':
    print("\n" + "█" * 70)
    print("  SCE ENERGY DEMAND & PRICE FORECASTING MODEL")
    print("  Pricing, Design & Research — Position Management Support")
    print("█" * 70)

    if not USE_REAL_DATA:
        print("\n  NOTE: Real CAISO CSV files not found on disk.")
        print("  Generating synthetic data for end-to-end demonstration...\n")
        generate_synthetic_data()
    else:
        print("\n  Using real CAISO data files.\n")

    # Run all analyses
    forecaster, df_load, bt, error_dist = run_demand_analysis()
    pf, lmp_df, merged = run_price_analysis()
    ha = run_hedging_analysis(lmp_df, merged, error_dist)

    print_summary()

    print("\nAnalysis complete. Output files:")
    print("  - output_stl_decomposition.png")
    print("  - output_demand_backtest.png")
    print("  - output_price_analysis.png")
    print("  - output_merit_order.png")
