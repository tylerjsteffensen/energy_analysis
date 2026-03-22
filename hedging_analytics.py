"""
Position Management & Hedging Analytics Module
================================================
Southern California Edison — Pricing, Design & Research

Provides quantitative tools for energy position management, hedge optimization,
and risk measurement in the CAISO wholesale market.

Designed to support:
- Volumetric exposure quantification (demand forecast uncertainty)
- Price exposure measurement (VaR, CVaR, Expected Shortfall)
- Optimal hedge ratio calculation (minimum-variance and utility-based)
- Mark-to-market of existing hedge positions
- Regulatory reporting support (CPUC procurement adequacy)

Usage:
    >>> from hedging_analytics import HedgingAnalytics
    >>> ha = HedgingAnalytics()
    >>> ha.calculate_var(price_series, position_mw=1000)
    >>> ha.optimal_hedge_ratio(spot_prices, forward_prices)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy import stats
from scipy.optimize import minimize_scalar


class HedgingAnalytics:
    """
    Energy position management and hedging toolkit for SCE.

    Covers three pillars of energy risk management:
    1. **Exposure quantification** — How much volume and price risk is open?
    2. **Risk measurement** — VaR, CVaR, stress tests
    3. **Hedge optimization** — Optimal hedge ratios, instrument selection
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    # ==================================================================
    # 1. EXPOSURE QUANTIFICATION
    # ==================================================================

    def volumetric_exposure(
        self,
        forecast_mw: pd.Series,
        forecast_std: pd.Series,
        hedged_mw: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calculate hourly volumetric exposure (open position).

        Open Position = Forecast Load − Hedged Volume

        Parameters
        ----------
        forecast_mw : Hourly demand forecast (MW)
        forecast_std : Hourly forecast standard deviation (MW)
        hedged_mw : Hourly hedged volume (MW), or None if fully open

        Returns
        -------
        DataFrame with open position and probabilistic exposure bands
        """
        if hedged_mw is None:
            hedged_mw = pd.Series(0, index=forecast_mw.index)

        z = stats.norm.ppf(self.confidence_level)

        exposure = pd.DataFrame({
            'forecast_mw': forecast_mw,
            'hedged_mw': hedged_mw,
            'open_position_mw': forecast_mw - hedged_mw,
            'forecast_std': forecast_std,
            'open_upper': (forecast_mw + z * forecast_std) - hedged_mw,
            'open_lower': (forecast_mw - z * forecast_std) - hedged_mw,
        })

        total_open = exposure['open_position_mw'].sum()
        total_forecast = exposure['forecast_mw'].sum()
        hedge_ratio = hedged_mw.sum() / total_forecast if total_forecast > 0 else 0

        print(f"\n--- Volumetric Exposure Summary ---")
        print(f"  Total forecast:    {total_forecast:,.0f} MWh")
        print(f"  Total hedged:      {hedged_mw.sum():,.0f} MWh")
        print(f"  Open position:     {total_open:,.0f} MWh")
        print(f"  Hedge ratio:       {hedge_ratio:.1%}")
        print(f"  Max hourly open:   {exposure['open_position_mw'].max():,.0f} MW")
        print(f"  Avg hourly open:   {exposure['open_position_mw'].mean():,.0f} MW")

        return exposure

    def price_exposure(
        self,
        open_position_mw: pd.Series,
        price_forecast: pd.Series,
        price_std: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculate dollar-denominated price exposure of open positions.

        Exposure ($) = Open Position (MW) × Price ($/MWh)
        """
        z = stats.norm.ppf(self.confidence_level)

        exposure = pd.DataFrame({
            'open_mw': open_position_mw,
            'price_forecast': price_forecast,
            'cost_forecast': open_position_mw * price_forecast,
            'cost_upper': open_position_mw * (price_forecast + z * price_std),
            'cost_lower': open_position_mw * (price_forecast - z * price_std),
        })

        total_cost = exposure['cost_forecast'].sum()
        worst_case = exposure['cost_upper'].sum()

        print(f"\n--- Price Exposure Summary ---")
        print(f"  Expected cost:   ${total_cost:,.0f}")
        print(f"  {self.confidence_level:.0%} worst case: ${worst_case:,.0f}")
        print(f"  Exposure range:  ${exposure['cost_lower'].sum():,.0f} – "
              f"${worst_case:,.0f}")

        return exposure

    # ==================================================================
    # 2. RISK MEASUREMENT
    # ==================================================================

    def calculate_var(
        self,
        returns: pd.Series,
        position_value: float,
        method: str = 'historical',
        holding_period: int = 1,
    ) -> dict:
        """
        Calculate Value-at-Risk for an energy position.

        Parameters
        ----------
        returns : Historical daily price returns (or log returns)
        position_value : Current mark-to-market value of position ($)
        method : 'historical', 'parametric', or 'cornish_fisher'
        holding_period : Holding period in days (for scaling)

        Returns
        -------
        dict with VaR, CVaR, and distribution parameters
        """
        clean = returns.dropna()
        alpha = 1 - self.confidence_level

        if method == 'historical':
            var_pct = np.percentile(clean, alpha * 100)
            cvar_pct = clean[clean <= var_pct].mean()

        elif method == 'parametric':
            mu = clean.mean()
            sigma = clean.std()
            var_pct = mu + sigma * stats.norm.ppf(alpha)
            # Analytical CVaR for normal distribution
            cvar_pct = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

        elif method == 'cornish_fisher':
            # Adjusts for skewness and kurtosis (important for energy prices)
            mu = clean.mean()
            sigma = clean.std()
            skew = clean.skew()
            kurt = clean.kurtosis()
            z = stats.norm.ppf(alpha)
            z_cf = (z
                    + (z**2 - 1) * skew / 6
                    + (z**3 - 3*z) * kurt / 24
                    - (2*z**3 - 5*z) * skew**2 / 36)
            var_pct = mu + sigma * z_cf
            # Approximate CVaR
            cvar_pct = clean[clean <= mu + sigma * z_cf].mean()

        else:
            raise ValueError(f"Unknown method: {method}")

        # Scale by holding period (sqrt-T rule)
        scale = np.sqrt(holding_period)
        var_dollar = abs(var_pct * scale * position_value)
        cvar_dollar = abs(cvar_pct * scale * position_value) if not np.isnan(cvar_pct) else np.nan

        result = {
            'var_pct': var_pct * scale,
            'var_dollar': var_dollar,
            'cvar_pct': cvar_pct * scale if not np.isnan(cvar_pct) else np.nan,
            'cvar_dollar': cvar_dollar,
            'method': method,
            'confidence': self.confidence_level,
            'holding_period_days': holding_period,
            'return_mean': clean.mean(),
            'return_std': clean.std(),
            'return_skew': clean.skew(),
            'return_kurtosis': clean.kurtosis(),
        }

        print(f"\n--- Value-at-Risk ({method}, {self.confidence_level:.0%}, "
              f"{holding_period}-day) ---")
        print(f"  Position value: ${position_value:,.0f}")
        print(f"  VaR:  {result['var_pct']:.2%} = ${var_dollar:,.0f}")
        print(f"  CVaR: {result['cvar_pct']:.2%} = ${cvar_dollar:,.0f}")
        print(f"  Return distribution: μ={clean.mean():.4f}, σ={clean.std():.4f}, "
              f"skew={clean.skew():.2f}, kurt={clean.kurtosis():.2f}")

        return result

    def stress_test(
        self,
        position_mw: float,
        scenarios: dict,
    ) -> pd.DataFrame:
        """
        Run price stress test scenarios on an open position.

        Parameters
        ----------
        position_mw : Open position size in MW
        scenarios : dict of {name: {'price_change': $/MWh, 'hours': int}}

        Example:
            scenarios = {
                'Heat wave (+$50/MWh, 72h)':     {'price_change': 50, 'hours': 72},
                'Gas spike (+$100/MWh, 24h)':     {'price_change': 100, 'hours': 24},
                'Neg prices (−$20/MWh, 8h)':      {'price_change': -20, 'hours': 8},
                'Sustained high (+$30/MWh, 168h)': {'price_change': 30, 'hours': 168},
            }
        """
        results = []
        for name, params in scenarios.items():
            cost_impact = position_mw * params['price_change'] * params['hours']
            results.append({
                'scenario': name,
                'price_change_per_mwh': params['price_change'],
                'duration_hours': params['hours'],
                'position_mw': position_mw,
                'total_impact': cost_impact,
            })

        df = pd.DataFrame(results).set_index('scenario')

        print(f"\n--- Stress Test Results (position: {position_mw:,.0f} MW) ---")
        for _, row in df.iterrows():
            sign = '+' if row['total_impact'] >= 0 else ''
            print(f"  {row.name}: {sign}${row['total_impact']:,.0f}")

        return df

    # ==================================================================
    # 3. HEDGE OPTIMIZATION
    # ==================================================================

    def optimal_hedge_ratio(
        self,
        spot_returns: pd.Series,
        hedge_returns: pd.Series,
        method: str = 'minimum_variance',
    ) -> dict:
        """
        Calculate optimal hedge ratio using spot and hedge instrument returns.

        Methods:
        - 'minimum_variance': β from OLS regression (classic Johnson hedge)
        - 'ols': Same as minimum_variance
        - 'correlation': ρ × (σ_spot / σ_hedge)

        Parameters
        ----------
        spot_returns : Price returns of the exposure (e.g., SP15 spot)
        hedge_returns : Price returns of the hedge instrument (e.g., SP15 forward)

        Returns
        -------
        dict with hedge ratio, effectiveness, and diagnostics
        """
        # Align series
        aligned = pd.DataFrame({
            'spot': spot_returns,
            'hedge': hedge_returns,
        }).dropna()

        if len(aligned) < 10:
            raise ValueError(f"Insufficient data: {len(aligned)} observations")

        rho = aligned['spot'].corr(aligned['hedge'])
        sigma_spot = aligned['spot'].std()
        sigma_hedge = aligned['hedge'].std()

        if method in ('minimum_variance', 'ols'):
            # OLS: spot = α + β·hedge + ε → β* = Cov(spot,hedge)/Var(hedge)
            cov = aligned['spot'].cov(aligned['hedge'])
            var_hedge = aligned['hedge'].var()
            h_star = cov / var_hedge if var_hedge > 0 else 0
        elif method == 'correlation':
            h_star = rho * (sigma_spot / sigma_hedge) if sigma_hedge > 0 else 0
        else:
            raise ValueError(f"Unknown method: {method}")

        # Hedge effectiveness: R² = 1 - Var(hedged)/Var(unhedged)
        hedged_returns = aligned['spot'] - h_star * aligned['hedge']
        effectiveness = 1 - hedged_returns.var() / aligned['spot'].var()

        result = {
            'hedge_ratio': h_star,
            'hedge_effectiveness': effectiveness,
            'correlation': rho,
            'sigma_spot': sigma_spot,
            'sigma_hedge': sigma_hedge,
            'method': method,
            'n_obs': len(aligned),
        }

        print(f"\n--- Optimal Hedge Ratio ({method}) ---")
        print(f"  h* = {h_star:.4f}")
        print(f"  Hedge effectiveness (R²): {effectiveness:.4f}")
        print(f"  Correlation: {rho:.4f}")
        print(f"  σ_spot:  {sigma_spot:.4f}")
        print(f"  σ_hedge: {sigma_hedge:.4f}")
        print(f"  Observations: {len(aligned)}")

        return result

    def hedge_cost_benefit(
        self,
        position_mw: float,
        hours: int,
        spot_forecast: float,
        forward_price: float,
        spot_std: float,
        hedge_ratio: float = 1.0,
    ) -> dict:
        """
        Cost-benefit analysis of hedging at the forward price vs staying open.

        Parameters
        ----------
        position_mw : Volume to hedge (MW)
        hours : Number of hours in the delivery period
        spot_forecast : Expected spot price ($/MWh)
        forward_price : Forward/swap price offered ($/MWh)
        spot_std : Standard deviation of spot price ($/MWh)
        hedge_ratio : Fraction of position to hedge (0-1)
        """
        volume_mwh = position_mw * hours
        hedged_volume = volume_mwh * hedge_ratio
        open_volume = volume_mwh * (1 - hedge_ratio)

        # Expected costs
        expected_spot_cost = volume_mwh * spot_forecast
        hedged_cost = hedged_volume * forward_price + open_volume * spot_forecast

        # Risk premium (cost of hedging vs expected spot)
        risk_premium = forward_price - spot_forecast
        hedge_premium_total = hedged_volume * risk_premium

        # Risk reduction
        z = stats.norm.ppf(self.confidence_level)
        unhedged_worst = volume_mwh * (spot_forecast + z * spot_std)
        hedged_worst = (hedged_volume * forward_price
                        + open_volume * (spot_forecast + z * spot_std))
        risk_reduction = unhedged_worst - hedged_worst

        result = {
            'volume_mwh': volume_mwh,
            'hedge_ratio': hedge_ratio,
            'expected_spot_cost': expected_spot_cost,
            'hedged_cost': hedged_cost,
            'risk_premium_per_mwh': risk_premium,
            'risk_premium_total': hedge_premium_total,
            'unhedged_worst_case': unhedged_worst,
            'hedged_worst_case': hedged_worst,
            'risk_reduction': risk_reduction,
        }

        print(f"\n--- Hedge Cost-Benefit Analysis ---")
        print(f"  Volume: {position_mw:,.0f} MW × {hours} hours = "
              f"{volume_mwh:,.0f} MWh")
        print(f"  Hedge ratio: {hedge_ratio:.0%}")
        print(f"  Spot forecast:   ${spot_forecast:.2f}/MWh")
        print(f"  Forward price:   ${forward_price:.2f}/MWh")
        print(f"  Risk premium:    ${risk_premium:+.2f}/MWh "
              f"(total: ${hedge_premium_total:+,.0f})")
        print(f"  Expected cost (open):   ${expected_spot_cost:,.0f}")
        print(f"  Expected cost (hedged): ${hedged_cost:,.0f}")
        print(f"  {self.confidence_level:.0%} worst case (open):   "
              f"${unhedged_worst:,.0f}")
        print(f"  {self.confidence_level:.0%} worst case (hedged): "
              f"${hedged_worst:,.0f}")
        print(f"  Risk reduction:  ${risk_reduction:,.0f}")

        return result

    # ==================================================================
    # 4. PORTFOLIO-LEVEL ANALYTICS
    # ==================================================================

    def position_summary(
        self,
        positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a position management summary report.

        Parameters
        ----------
        positions : DataFrame with columns:
            - instrument: Name (e.g., 'SP15 Jan Forward', 'SoCal Gas Swap')
            - direction: 'long' or 'short'
            - volume_mwh: Position size
            - price: Contracted price
            - current_mark: Current market price
            - delivery_start: Start of delivery period
            - delivery_end: End of delivery period
        """
        pos = positions.copy()
        pos['sign'] = pos['direction'].map({'long': 1, 'short': -1})
        pos['notional'] = pos['volume_mwh'] * pos['price']
        pos['mark_to_market'] = pos['sign'] * pos['volume_mwh'] * (
            pos['current_mark'] - pos['price']
        )
        pos['unrealized_pnl_pct'] = (
            (pos['current_mark'] - pos['price']) / pos['price'] * 100
        ).round(2)

        total_mtm = pos['mark_to_market'].sum()
        total_notional = pos['notional'].sum()

        print(f"\n{'='*60}")
        print(f"POSITION MANAGEMENT SUMMARY")
        print(f"{'='*60}")
        for _, p in pos.iterrows():
            print(f"\n  {p['instrument']} ({p['direction'].upper()})")
            print(f"    Volume:      {p['volume_mwh']:,.0f} MWh")
            print(f"    Contract:    ${p['price']:.2f}/MWh")
            print(f"    Current:     ${p['current_mark']:.2f}/MWh")
            print(f"    MTM P&L:     ${p['mark_to_market']:+,.0f} "
                  f"({p['unrealized_pnl_pct']:+.2f}%)")

        print(f"\n  {'—'*40}")
        print(f"  Total notional:    ${total_notional:,.0f}")
        print(f"  Total MTM P&L:     ${total_mtm:+,.0f}")

        return pos

    def correlation_matrix(
        self,
        price_series: dict,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix across energy commodities.

        Useful for identifying natural hedges and diversification benefits.

        Parameters
        ----------
        price_series : dict of {name: pd.Series} with aligned price data
            e.g., {'SP15 LMP': ..., 'SoCal Gas': ..., 'Carbon': ...}
        """
        df = pd.DataFrame(price_series).dropna()
        returns = df.pct_change().dropna()
        corr = returns.corr().round(3)

        print(f"\n--- Commodity Return Correlations ---")
        print(corr)
        print(f"\n  (Based on {len(returns)} daily return observations)")

        return corr
