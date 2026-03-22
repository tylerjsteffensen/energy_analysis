"""
Electricity Demand Forecasting Module
======================================
Southern California Edison — Pricing, Design & Research

Provides SARIMAX-based demand forecasting for CAISO / SCE-TAC hourly load,
with feature engineering for weather, calendar effects, and lag structures.

Designed to support:
- Short-term (1-7 day) hourly load forecasts for position management
- Forecast error characterization for volumetric hedge sizing
- Peak demand estimation for capacity procurement

Usage with existing repo data (1 month):
    >>> from demand_forecast import DemandForecaster
    >>> forecaster = DemandForecaster()
    >>> df = forecaster.load_caiso_data('1_month_CAISO_load.csv')
    >>> forecaster.fit(df)
    >>> forecast = forecaster.predict(steps=168)  # 7-day ahead
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


class DemandForecaster:
    """
    Hourly electricity demand forecaster using SARIMAX with engineered features.

    The model captures:
    - Diurnal (24h) and weekly (168h) seasonality
    - Day-of-week and holiday effects
    - Temperature-driven load (CDD/HDD) when weather data is available
    - Autoregressive structure in load forecast errors
    """

    def __init__(self, tac_area: str = 'CA ISO-TAC', freq: str = 'h'):
        self.tac_area = tac_area
        self.freq = freq
        self.model = None
        self.model_fit = None
        self.scaler = StandardScaler()
        self.train_data = None
        self.feature_cols = []

    # ------------------------------------------------------------------
    # Data loading & preparation
    # ------------------------------------------------------------------

    def load_caiso_data(self, filepath: str) -> pd.DataFrame:
        """Load and parse CAISO load CSV from OASIS format."""
        df = pd.read_csv(filepath)
        df['start_time'] = pd.to_datetime(
            df['INTERVALSTARTTIME_GMT'], utc=True, errors='coerce'
        )
        df = df.set_index('start_time').sort_index()

        # Filter to target TAC area
        if 'TAC_AREA_NAME' in df.columns:
            df = df[df['TAC_AREA_NAME'] == self.tac_area].copy()

        # Resample to hourly (handles 5-min/15-min data)
        hourly = df[['MW']].resample(self.freq).mean().dropna()
        hourly.columns = ['load_mw']

        print(f"Loaded {len(hourly)} hourly observations for {self.tac_area}")
        print(f"  Range: {hourly.index.min()} → {hourly.index.max()}")
        print(f"  Load range: {hourly['load_mw'].min():.0f} – "
              f"{hourly['load_mw'].max():.0f} MW")

        return hourly

    def engineer_features(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build exogenous feature matrix for demand modeling.

        Parameters
        ----------
        df : DataFrame with DatetimeIndex and 'load_mw' column
        weather_df : Optional DataFrame with 'temp_f' column (hourly, same index)

        Returns
        -------
        DataFrame with load_mw + engineered feature columns
        """
        out = df.copy()

        # --- Calendar features ---
        out['hour'] = out.index.hour
        out['dow'] = out.index.dayofweek  # 0=Mon, 6=Sun
        out['month'] = out.index.month
        out['is_weekend'] = (out['dow'] >= 5).astype(int)

        # Fourier terms for daily cycle (captures smooth diurnal shape)
        for k in [1, 2, 3]:
            out[f'sin_24_{k}'] = np.sin(2 * np.pi * k * out['hour'] / 24)
            out[f'cos_24_{k}'] = np.cos(2 * np.pi * k * out['hour'] / 24)

        # Fourier terms for weekly cycle
        hour_of_week = out['dow'] * 24 + out['hour']
        for k in [1, 2]:
            out[f'sin_168_{k}'] = np.sin(2 * np.pi * k * hour_of_week / 168)
            out[f'cos_168_{k}'] = np.cos(2 * np.pi * k * hour_of_week / 168)

        # --- Lag features ---
        out['load_lag_24'] = out['load_mw'].shift(24)   # Same hour yesterday
        out['load_lag_168'] = out['load_mw'].shift(168)  # Same hour last week
        out['load_rolling_24'] = out['load_mw'].rolling(24).mean()

        # --- Weather features (if available) ---
        if weather_df is not None and 'temp_f' in weather_df.columns:
            weather = weather_df[['temp_f']].reindex(out.index, method='nearest')
            out['temp_f'] = weather['temp_f']

            # CDD/HDD (base 65°F is standard for SCE territory)
            out['cdd'] = np.maximum(out['temp_f'] - 65, 0)
            out['hdd'] = np.maximum(65 - out['temp_f'], 0)

            # Quadratic CDD (captures nonlinear AC load at extreme heat)
            out['cdd_sq'] = out['cdd'] ** 2

            # Temperature-hour interaction (AC load peaks in afternoon)
            out['cdd_x_hour'] = out['cdd'] * np.where(
                (out['hour'] >= 12) & (out['hour'] <= 20), 1, 0
            )

        return out

    # ------------------------------------------------------------------
    # Stationarity diagnostics
    # ------------------------------------------------------------------

    def stationarity_report(self, series: pd.Series) -> dict:
        """Run ADF test and report stationarity diagnostics."""
        result = adfuller(series.dropna(), maxlag=48, autolag='AIC')
        report = {
            'adf_stat': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_obs': result[3],
            'is_stationary': result[1] < 0.05,
        }
        print(f"ADF test statistic: {report['adf_stat']:.4f}")
        print(f"p-value: {report['p_value']:.6f}")
        print(f"Stationary at 5%: {report['is_stationary']}")
        return report

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        order: tuple = (2, 0, 1),
        seasonal_order: tuple = (1, 0, 1, 24),
        verbose: bool = True,
    ):
        """
        Fit SARIMAX model on the prepared load data.

        Parameters
        ----------
        df : DataFrame with 'load_mw' column and DatetimeIndex
        weather_df : Optional hourly weather data
        order : ARIMA(p,d,q) order
        seasonal_order : Seasonal (P,D,Q,s) — s=24 for hourly data
        verbose : Print model summary
        """
        featured = self.engineer_features(df, weather_df)
        featured = featured.dropna()

        endog = featured['load_mw']
        self.feature_cols = [
            c for c in featured.columns
            if c not in ['load_mw', 'hour', 'dow', 'month']
        ]
        exog = featured[self.feature_cols] if self.feature_cols else None

        if exog is not None and len(exog.columns) > 0:
            # Scale exogenous features for numerical stability
            exog_scaled = pd.DataFrame(
                self.scaler.fit_transform(exog),
                index=exog.index,
                columns=exog.columns,
            )
        else:
            exog_scaled = None

        self.train_data = featured

        print(f"Fitting SARIMAX{order}x{seasonal_order} on "
              f"{len(endog)} observations...")
        if exog_scaled is not None:
            print(f"  Exogenous features: {list(exog_scaled.columns)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SARIMAX(
                endog,
                exog=exog_scaled,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = self.model.fit(disp=False, maxiter=200)

        if verbose:
            print(self.model_fit.summary().tables[0])
            print(f"\nAIC: {self.model_fit.aic:.1f}")
            print(f"BIC: {self.model_fit.bic:.1f}")

        # In-sample diagnostics
        resid = self.model_fit.resid.dropna()
        print(f"\nResidual diagnostics:")
        print(f"  Mean: {resid.mean():.2f} MW")
        print(f"  Std:  {resid.std():.2f} MW")
        print(f"  MAE:  {mean_absolute_error(endog.iloc[-len(resid):], endog.iloc[-len(resid):] - resid):.2f} MW")

        return self.model_fit

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def predict(
        self,
        steps: int = 24,
        exog_future: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Generate out-of-sample forecast with confidence intervals.

        Parameters
        ----------
        steps : Number of hourly steps to forecast
        exog_future : Future exogenous features (must match training features)
        alpha : Confidence interval significance level (default 95% CI)

        Returns
        -------
        DataFrame with columns: forecast, lower_ci, upper_ci
        """
        if self.model_fit is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        if exog_future is not None:
            exog_future_scaled = pd.DataFrame(
                self.scaler.transform(exog_future[self.feature_cols]),
                index=exog_future.index,
                columns=self.feature_cols,
            )
        else:
            exog_future_scaled = None

        forecast = self.model_fit.get_forecast(
            steps=steps, exog=exog_future_scaled, alpha=alpha
        )

        result = pd.DataFrame({
            'forecast_mw': forecast.predicted_mean,
            'lower_ci': forecast.conf_int().iloc[:, 0],
            'upper_ci': forecast.conf_int().iloc[:, 1],
        })

        return result

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.8,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Walk-forward backtest: train on first train_pct, forecast the rest.

        Returns dict with forecast DataFrame and accuracy metrics.
        """
        n = len(df)
        split = int(n * train_pct)

        train = df.iloc[:split]
        test = df.iloc[split:]

        weather_train = None
        if weather_df is not None:
            weather_train = weather_df.iloc[:split]

        self.fit(train, weather_df=weather_train, verbose=False)

        # Build exogenous features for the test period
        weather_test = None
        if weather_df is not None:
            weather_test = weather_df.iloc[split:]
        test_featured = self.engineer_features(df, weather_df=weather_df).iloc[split:]
        test_featured = test_featured.dropna()
        exog_future = test_featured[self.feature_cols] if self.feature_cols else None

        if exog_future is not None and len(exog_future.columns) > 0:
            exog_future_scaled = pd.DataFrame(
                self.scaler.transform(exog_future),
                index=exog_future.index,
                columns=exog_future.columns,
            )
            steps = len(exog_future_scaled)
            fc = self.model_fit.get_forecast(steps=steps, exog=exog_future_scaled, alpha=0.05)
            fc_df = pd.DataFrame({
                'forecast_mw': fc.predicted_mean,
                'lower_ci': fc.conf_int().iloc[:, 0],
                'upper_ci': fc.conf_int().iloc[:, 1],
            })
        else:
            fc_df = self.predict(steps=len(test))
        fc = fc_df

        actual = test['load_mw'].values
        predicted = fc['forecast_mw'].values[:len(actual)]

        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100

        print(f"\nBacktest results ({len(test)} hours out-of-sample):")
        print(f"  MAE:  {mae:.1f} MW")
        print(f"  MAPE: {mape:.2f}%")

        return {
            'forecast': fc,
            'actual': test,
            'mae': mae,
            'mape': mape,
        }

    # ------------------------------------------------------------------
    # STL Decomposition (trend / seasonal / residual)
    # ------------------------------------------------------------------

    def decompose(self, df: pd.DataFrame, period: int = 24) -> object:
        """
        STL decomposition of the load time series.

        Useful for:
        - Identifying trend shifts (economic growth, EV adoption)
        - Quantifying seasonal amplitude (peak/off-peak spread)
        - Isolating residual volatility for hedge sizing
        """
        stl = STL(df['load_mw'].dropna(), period=period, robust=True)
        result = stl.fit()
        print(f"STL decomposition (period={period}h):")
        print(f"  Trend range:    {result.trend.min():.0f} – {result.trend.max():.0f} MW")
        print(f"  Seasonal amp:   {result.seasonal.std():.0f} MW (std)")
        print(f"  Residual std:   {result.resid.std():.0f} MW")
        return result

    # ------------------------------------------------------------------
    # Forecast error distribution (for hedge sizing)
    # ------------------------------------------------------------------

    def forecast_error_distribution(self, backtest_result: dict) -> dict:
        """
        Characterize forecast error distribution for volumetric hedging.

        Returns percentile-based error bands that Pricing can use to size
        hedge volumes above/below the point forecast.
        """
        actual = backtest_result['actual']['load_mw'].values
        predicted = backtest_result['forecast']['forecast_mw'].values[:len(actual)]
        errors = actual - predicted

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(errors, percentiles)

        dist = {
            'mean_error': errors.mean(),
            'std_error': errors.std(),
            'skewness': pd.Series(errors).skew(),
            'kurtosis': pd.Series(errors).kurtosis(),
            'percentiles': dict(zip(percentiles, pct_values)),
        }

        print("\nForecast error distribution (MW):")
        print(f"  Mean: {dist['mean_error']:.1f}")
        print(f"  Std:  {dist['std_error']:.1f}")
        print(f"  Skew: {dist['skewness']:.2f}")
        print(f"  Kurt: {dist['kurtosis']:.2f}")
        print("  Percentiles:")
        for p, v in dist['percentiles'].items():
            print(f"    P{p:02d}: {v:+.0f} MW")

        return dist
