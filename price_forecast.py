"""
Energy Commodity Price Forecasting Module
==========================================
Southern California Edison — Pricing, Design & Research

Provides econometric models for CAISO Day-Ahead and Real-Time LMP forecasting,
with structural decomposition of price drivers (gas, load, renewables, congestion).

Key models:
1. Structural OLS — LMP as a function of fundamental drivers (gas, load, renewables)
2. SARIMAX — Time-series model with exogenous fundamental regressors
3. Regime-switching — Captures price spikes vs normal-state behavior

Designed to support:
- Day-ahead price forecasting for scheduling and bidding
- Spark-spread estimation for gas-fired generation portfolio
- Price volatility estimation for hedge sizing and VaR

Usage:
    >>> from price_forecast import PriceForecaster
    >>> pf = PriceForecaster(hub='SP15')
    >>> pf.fit_structural(lmp_df, gas_df, load_df, renew_df)
    >>> pf.predict(steps=24)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


class PriceForecaster:
    """
    Econometric electricity price forecaster for CAISO hub LMPs.

    Implements three complementary modeling approaches:

    1. **Structural model** (OLS / GLS):
       LMP = f(gas_price, system_load, renewable_gen, hour_of_day, congestion_proxy)
       - Transparent, interpretable price decomposition
       - Useful for scenario analysis ("what if gas rises $1/MMBtu?")
       - Supports Pricing team's rate case testimony

    2. **Time-series model** (SARIMAX):
       Captures autocorrelation and mean-reversion in LMP residuals after
       removing structural drivers
       - Better short-term (1-24h) forecast accuracy
       - Provides confidence intervals for position management

    3. **Quantile regression**:
       Models tail risk (P90/P95/P99 prices) for VaR and stress testing
       - Essential for sizing financial hedges
       - Captures asymmetric price spikes
    """

    def __init__(self, hub: str = 'SP15'):
        self.hub = hub
        self.structural_model = None
        self.ts_model = None
        self.ts_model_fit = None
        self.quantile_models = {}
        self.train_data = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def load_lmp_data(self, filepath: str) -> pd.DataFrame:
        """Load CAISO LMP data and extract hub-level prices."""
        df = pd.read_csv(filepath)
        df['start_time'] = pd.to_datetime(
            df['INTERVALSTARTTIME_GMT'], utc=True, errors='coerce'
        )
        df = df.set_index('start_time').sort_index()

        # Pivot to wide format
        df_wide = df.pivot_table(
            index=['start_time', 'NODE_ID'],
            columns='LMP_TYPE',
            values='MW',
            aggfunc='first',
        ).reset_index()

        # Filter to trading hub nodes
        hub_kw = f'TH_{self.hub}'
        df_hub = df_wide[df_wide['NODE_ID'].str.contains(hub_kw, na=False)].copy()

        # Average across hub nodes per interval
        hub_ts = (
            df_hub.groupby('start_time')[['LMP', 'MCE', 'MCC', 'MCL', 'MGHG']]
            .mean()
        )

        print(f"Loaded {self.hub} hub LMP: {len(hub_ts)} intervals")
        print(f"  LMP range: ${hub_ts['LMP'].min():.2f} – ${hub_ts['LMP'].max():.2f}/MWh")

        return hub_ts

    def prepare_fundamentals(
        self,
        lmp_df: pd.DataFrame,
        load_df: Optional[pd.DataFrame] = None,
        renew_df: Optional[pd.DataFrame] = None,
        gas_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge LMP with fundamental driver datasets.

        Parameters
        ----------
        lmp_df : Hub-level LMP time series (from load_lmp_data)
        load_df : System load with 'load_mw' column
        renew_df : Renewable generation with 'renew_mw' column
        gas_df : Gas prices with 'gas_price' column ($/MMBtu)

        Returns
        -------
        Merged DataFrame aligned on hourly timestamps
        """
        merged = lmp_df.copy()

        # Calendar features
        merged['hour'] = merged.index.hour
        merged['dow'] = merged.index.dayofweek
        merged['is_weekend'] = (merged['dow'] >= 5).astype(int)

        # Fourier terms for diurnal price pattern
        for k in [1, 2]:
            merged[f'sin_24_{k}'] = np.sin(2 * np.pi * k * merged['hour'] / 24)
            merged[f'cos_24_{k}'] = np.cos(2 * np.pi * k * merged['hour'] / 24)

        # Merge load
        if load_df is not None:
            if 'load_mw' not in load_df.columns and 'MW' in load_df.columns:
                load_df = load_df.rename(columns={'MW': 'load_mw'})
            load_hourly = load_df[['load_mw']].resample('h').mean()
            merged = merged.join(load_hourly, how='left')

            # Net load (if renewables available)
            if renew_df is not None:
                if 'renew_mw' not in renew_df.columns and 'MW' in renew_df.columns:
                    renew_df = renew_df.rename(columns={'MW': 'renew_mw'})
                renew_hourly = renew_df[['renew_mw']].resample('h').sum()
                merged = merged.join(renew_hourly, how='left')
                merged['net_load_mw'] = merged['load_mw'] - merged['renew_mw'].fillna(0)
                merged['renew_pct'] = (
                    merged['renew_mw'] / merged['load_mw'] * 100
                ).round(2)

        # Merge gas price (forward-fill daily to hourly)
        if gas_df is not None:
            if 'gas_price' not in gas_df.columns:
                # Try to detect the price column
                price_cols = [c for c in gas_df.columns if 'price' in c.lower()]
                if price_cols:
                    gas_df = gas_df.rename(columns={price_cols[0]: 'gas_price'})
            gas_hourly = gas_df[['gas_price']].resample('h').ffill()
            merged = merged.join(gas_hourly, how='left')

            # Implied heat rate (LMP / gas price) — key metric for gas-fired gen
            if 'gas_price' in merged.columns:
                merged['implied_heat_rate'] = np.where(
                    merged['gas_price'] > 0,
                    merged['LMP'] / merged['gas_price'],
                    np.nan,
                )

        # Lag features
        merged['lmp_lag_1'] = merged['LMP'].shift(1)
        merged['lmp_lag_24'] = merged['LMP'].shift(24)

        return merged

    # ------------------------------------------------------------------
    # Model 1: Structural OLS
    # ------------------------------------------------------------------

    def fit_structural(
        self,
        lmp_df: pd.DataFrame,
        load_df: Optional[pd.DataFrame] = None,
        renew_df: Optional[pd.DataFrame] = None,
        gas_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """
        Fit structural econometric model:
            LMP = β₀ + β₁·gas + β₂·load + β₃·renewables + β₄·hour_dummies + ε

        This model is interpretable and suitable for rate case testimony,
        scenario analysis, and long-term price forecasting.
        """
        merged = self.prepare_fundamentals(lmp_df, load_df, renew_df, gas_df)
        merged = merged.dropna(subset=['LMP'])
        self.train_data = merged

        # Build regressor matrix
        feature_cols = ['is_weekend', 'sin_24_1', 'cos_24_1', 'sin_24_2', 'cos_24_2']

        if 'gas_price' in merged.columns:
            feature_cols.append('gas_price')
        if 'load_mw' in merged.columns:
            feature_cols.append('load_mw')
        if 'net_load_mw' in merged.columns:
            feature_cols.append('net_load_mw')
        elif 'renew_mw' in merged.columns:
            feature_cols.append('renew_mw')
        # Only include lag features if they have enough non-null values
        if 'lmp_lag_24' in merged.columns and merged['lmp_lag_24'].notna().sum() > 5:
            feature_cols.append('lmp_lag_24')

        # Remove features that are entirely NaN after merge
        feature_cols = [c for c in feature_cols if merged[c].notna().sum() > 5]

        data = merged[['LMP'] + feature_cols].dropna()

        if len(data) < 5:
            print(f"WARNING: Only {len(data)} valid observations after merge. "
                  f"Need more data for robust estimation.")
            print(f"  Available columns: {list(merged.columns)}")
            print(f"  NaN counts:\n{merged[['LMP'] + feature_cols].isna().sum()}")
            return None

        y = data['LMP']
        X = sm.add_constant(data[feature_cols])

        self.structural_model = sm.OLS(y, X).fit(cov_type='HC1')  # robust SEs

        if verbose:
            print("=" * 60)
            print(f"STRUCTURAL LMP MODEL — {self.hub} Hub")
            print("=" * 60)
            print(self.structural_model.summary())
            print(f"\nR²: {self.structural_model.rsquared:.4f}")
            print(f"Adj R²: {self.structural_model.rsquared_adj:.4f}")

            # Economic interpretation
            print("\n--- Price Sensitivity Estimates ---")
            params = self.structural_model.params
            if 'gas_price' in params:
                print(f"  Gas price:  +$1/MMBtu → +${params['gas_price']:.2f}/MWh in LMP")
            if 'load_mw' in params:
                print(f"  System load: +1,000 MW → +${params['load_mw']*1000:.2f}/MWh in LMP")
            if 'net_load_mw' in params:
                print(f"  Net load: +1,000 MW → +${params['net_load_mw']*1000:.2f}/MWh in LMP")
            if 'renew_mw' in params:
                print(f"  Renewables: +1,000 MW → ${params['renew_mw']*1000:+.2f}/MWh in LMP")

        return self.structural_model

    # ------------------------------------------------------------------
    # Model 2: SARIMAX time-series
    # ------------------------------------------------------------------

    def fit_timeseries(
        self,
        lmp_df: pd.DataFrame,
        load_df: Optional[pd.DataFrame] = None,
        renew_df: Optional[pd.DataFrame] = None,
        gas_df: Optional[pd.DataFrame] = None,
        order: tuple = (2, 0, 1),
        seasonal_order: tuple = (1, 0, 1, 24),
        verbose: bool = True,
    ):
        """
        Fit SARIMAX model for short-term LMP forecasting.

        Captures autocorrelation structure in prices after removing
        fundamental drivers via exogenous regressors.
        """
        merged = self.prepare_fundamentals(lmp_df, load_df, renew_df, gas_df)
        merged = merged.dropna(subset=['LMP'])

        exog_cols = ['is_weekend', 'sin_24_1', 'cos_24_1']
        if 'gas_price' in merged.columns:
            exog_cols.append('gas_price')
        if 'net_load_mw' in merged.columns:
            exog_cols.append('net_load_mw')
        elif 'load_mw' in merged.columns:
            exog_cols.append('load_mw')

        data = merged[['LMP'] + exog_cols].dropna()
        endog = data['LMP']
        exog = data[exog_cols]

        print(f"Fitting SARIMAX{order}x{seasonal_order} for {self.hub} LMP...")
        print(f"  Exogenous: {exog_cols}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ts_model = SARIMAX(
                endog,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.ts_model_fit = self.ts_model.fit(disp=False, maxiter=200)

        if verbose:
            print(self.ts_model_fit.summary().tables[0])
            print(f"\nAIC: {self.ts_model_fit.aic:.1f}")
            print(f"BIC: {self.ts_model_fit.bic:.1f}")

        return self.ts_model_fit

    # ------------------------------------------------------------------
    # Model 3: Quantile regression (tail risk)
    # ------------------------------------------------------------------

    def fit_quantile(
        self,
        lmp_df: pd.DataFrame,
        load_df: Optional[pd.DataFrame] = None,
        renew_df: Optional[pd.DataFrame] = None,
        gas_df: Optional[pd.DataFrame] = None,
        quantiles: list = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    ):
        """
        Fit quantile regression models for price distribution estimation.

        Critical for:
        - VaR calculation (P95/P99 price exposure)
        - Sizing financial hedges (call options, collars)
        - Stress testing procurement portfolio
        """
        merged = self.prepare_fundamentals(lmp_df, load_df, renew_df, gas_df)
        merged = merged.dropna(subset=['LMP'])

        feature_cols = ['is_weekend', 'sin_24_1', 'cos_24_1']
        if 'load_mw' in merged.columns:
            feature_cols.append('load_mw')
        if 'gas_price' in merged.columns:
            feature_cols.append('gas_price')

        data = merged[['LMP'] + feature_cols].dropna()
        y = data['LMP']
        X = sm.add_constant(data[feature_cols])

        print(f"Fitting quantile regressions at τ = {quantiles}...")

        for q in quantiles:
            qr = QuantReg(y, X).fit(q=q)
            self.quantile_models[q] = qr

        # Report price percentile estimates
        print("\n--- Price Percentile Estimates (Unconditional) ---")
        for q in quantiles:
            pred = self.quantile_models[q].predict(X)
            print(f"  Q{int(q*100):02d}: ${pred.mean():.2f}/MWh (avg across sample)")

        return self.quantile_models

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_structural(
        self,
        future_X: pd.DataFrame,
    ) -> pd.Series:
        """Predict LMP using the structural model given future fundamentals."""
        if self.structural_model is None:
            raise RuntimeError("Structural model not fitted.")

        X = sm.add_constant(future_X)
        return self.structural_model.predict(X)

    def predict_timeseries(
        self,
        steps: int = 24,
        exog_future: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Predict LMP using the SARIMAX model with confidence intervals."""
        if self.ts_model_fit is None:
            raise RuntimeError("Time-series model not fitted.")

        fc = self.ts_model_fit.get_forecast(steps=steps, exog=exog_future, alpha=alpha)

        return pd.DataFrame({
            'forecast_lmp': fc.predicted_mean,
            'lower_ci': fc.conf_int().iloc[:, 0],
            'upper_ci': fc.conf_int().iloc[:, 1],
        })

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------

    def scenario_analysis(
        self,
        base_case: pd.DataFrame,
        scenarios: dict,
    ) -> pd.DataFrame:
        """
        Run what-if scenarios on the structural model.

        Parameters
        ----------
        base_case : DataFrame of base-case fundamentals (same cols as training)
        scenarios : dict of {name: {col: multiplier_or_adder}} adjustments

        Example:
            scenarios = {
                'Gas +$2':       {'gas_price': ('+', 2.0)},
                'Load +10%':     {'load_mw': ('*', 1.10)},
                'Solar doubles': {'renew_mw': ('*', 2.0)},
            }
        """
        if self.structural_model is None:
            raise RuntimeError("Fit structural model first.")

        results = {'Base Case': self.predict_structural(base_case).mean()}

        for name, adjustments in scenarios.items():
            scenario_X = base_case.copy()
            for col, (op, val) in adjustments.items():
                if col in scenario_X.columns:
                    if op == '+':
                        scenario_X[col] = scenario_X[col] + val
                    elif op == '*':
                        scenario_X[col] = scenario_X[col] * val
            results[name] = self.predict_structural(scenario_X).mean()

        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['avg_lmp'])
        result_df['delta_vs_base'] = result_df['avg_lmp'] - result_df.loc['Base Case', 'avg_lmp']

        print("\n--- Scenario Analysis Results ---")
        print(result_df.round(2))

        return result_df

    # ------------------------------------------------------------------
    # Spark spread analysis
    # ------------------------------------------------------------------

    def spark_spread(
        self,
        merged_df: pd.DataFrame,
        heat_rate: float = 7.0,
        vom: float = 3.5,
    ) -> pd.DataFrame:
        """
        Calculate spark spreads for gas-fired generation economics.

        Spark Spread = LMP - (Gas Price × Heat Rate) - VOM

        Parameters
        ----------
        merged_df : DataFrame with 'LMP' and 'gas_price' columns
        heat_rate : Plant heat rate in MMBtu/MWh (default 7.0 for CCGT)
        vom : Variable O&M cost in $/MWh
        """
        if 'gas_price' not in merged_df.columns:
            raise ValueError("Gas price data required for spark spread analysis.")

        ss = merged_df[['LMP', 'gas_price']].dropna().copy()
        ss['fuel_cost'] = ss['gas_price'] * heat_rate
        ss['spark_spread'] = ss['LMP'] - ss['fuel_cost'] - vom
        ss['in_the_money'] = (ss['spark_spread'] > 0).astype(int)

        print(f"\n--- Spark Spread Analysis (HR={heat_rate}, VOM=${vom}) ---")
        print(f"  Avg LMP:          ${ss['LMP'].mean():.2f}/MWh")
        print(f"  Avg fuel cost:    ${ss['fuel_cost'].mean():.2f}/MWh")
        print(f"  Avg spark spread: ${ss['spark_spread'].mean():.2f}/MWh")
        print(f"  % hours in-money: {ss['in_the_money'].mean()*100:.1f}%")
        print(f"  Max spark spread: ${ss['spark_spread'].max():.2f}/MWh")
        print(f"  Min spark spread: ${ss['spark_spread'].min():.2f}/MWh")

        return ss
