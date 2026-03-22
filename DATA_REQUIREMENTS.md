# Data Requirements for SCE Energy Demand & Price Forecasting Model

## Overview

This document specifies the datasets needed to build and calibrate statistical and
econometric models for (1) electricity demand forecasting, (2) energy commodity price
forecasting, and (3) position management / hedging analytics — all in the context of
Southern California Edison's service territory within the CAISO market.

---

## 1. ALREADY AVAILABLE IN THIS REPO

| Dataset | File | Granularity | Coverage |
|---------|------|-------------|----------|
| CAISO Day-Ahead LMP (all nodes) | `1_day_LMP_DAM.csv` | Hourly × node | 2025-12-10 (1 day) |
| CAISO Actual System Load | `1_month_CAISO_load.csv` | Hourly × TAC | Dec 2025 (1 month) |
| CAISO DAM Load Forecast | `1_month_DAM_forecast.csv` | Hourly × TAC | Dec 2025 (1 month) |
| CAISO RTM Load Forecast | `1_month_RTM_forecast.csv` | 15-min × TAC | Dec 2025 (partial) |
| CAISO Renewables Actual | `1_month_Renewables_Act.csv` | 5/15-min × hub × type | Dec 2025 (1 month) |

---

## 2. DATA NEEDED — PRIORITY 1 (Critical for Model Calibration)

These datasets are essential to train robust demand and price forecasting models.
**At minimum, 2-3 years of historical data is needed** for seasonal pattern estimation.

### 2a. Extended Historical CAISO Load & LMP

| Dataset | Source | Granularity | Requested Range |
|---------|--------|-------------|-----------------|
| **CAISO actual system load** | OASIS / EIA-930 | Hourly × TAC area | Jan 2022 – Dec 2025 |
| **CAISO Day-Ahead LMP** (SP15 hub) | OASIS | Hourly | Jan 2022 – Dec 2025 |
| **CAISO Real-Time LMP** (SP15 hub) | OASIS | 5-min or 15-min | Jan 2023 – Dec 2025 |
| **SCE-TAC specific load** | OASIS | Hourly | Jan 2022 – Dec 2025 |

> **Why SP15?** SCE's service territory maps to the SP15 trading hub. Position
> management and hedging at SCE is benchmarked against SP15 hub prices.

### 2b. Weather Data (Demand Driver #1)

| Dataset | Source | Variables | Requested Range |
|---------|--------|-----------|-----------------|
| **Hourly temperature** for SCE territory | NOAA ISD / Visual Crossing | Temp (°F), Humidity (%), Wind Speed | Jan 2022 – Dec 2025 |
| Representative stations | LAX, Riverside, Burbank, Palm Springs | Same | Same |

> **Key derived features:** Cooling Degree Days (CDD), Heating Degree Days (HDD),
> Temperature-Humidity Index (THI). These are the strongest demand drivers in SCE territory
> where summer cooling load dominates.

### 2c. Natural Gas Prices (Price Driver #1)

| Dataset | Source | Granularity | Requested Range |
|---------|--------|-------------|-----------------|
| **SoCal Citygate daily spot** | ICE / Platts / Bloomberg | Daily | Jan 2022 – Dec 2025 |
| **Henry Hub daily spot** | EIA / CME | Daily | Jan 2022 – Dec 2025 |
| **SoCal Citygate forward curve** | ICE / CME NYMEX | Monthly strips | Current |
| **Henry Hub forward curve** | CME NYMEX | Monthly strips | Current |

> **Why gas prices?** Natural gas sets the marginal cost for ~40-50% of CAISO dispatch hours.
> The SoCal Citygate basis (premium over Henry Hub) captures pipeline constraints into
> Southern California — a critical hedging risk factor.

---

## 3. DATA NEEDED — PRIORITY 2 (Improves Model Accuracy)

### 3a. Renewable Generation (Extended History)

| Dataset | Source | Granularity | Requested Range |
|---------|--------|-------------|-----------------|
| **CAISO solar generation** | OASIS Renewables | Hourly | Jan 2022 – Dec 2025 |
| **CAISO wind generation** | OASIS Renewables | Hourly | Jan 2022 – Dec 2025 |
| **Behind-the-meter solar estimate** | CAISO / CEC | Daily/Monthly | 2022 – 2025 |

> **Why?** The merit-order effect (visible in your `lmp_vs_generation.ipynb`) is the
> primary driver of midday price suppression. Solar ramps create hedging exposure in the
> evening peak (the "duck curve" net-load ramp).

### 3b. GHG Allowance Prices

| Dataset | Source | Granularity | Requested Range |
|---------|--------|-------------|-----------------|
| **California Carbon Allowance (CCA)** | ICE / CARB | Daily | Jan 2022 – Dec 2025 |

> **Why?** The MGHG component of LMP directly reflects carbon costs. Cap-and-trade
> compliance costs flow through to wholesale prices and affect hedging economics.

### 3c. Calendar & Economic Variables

| Dataset | Source | Notes |
|---------|--------|-------|
| **CAISO holiday calendar** | CAISO OASIS | NERC holidays + CA-specific |
| **EV adoption / electrification metrics** | CEC | Annual, SCE territory |

---

## 4. DATA NEEDED — PRIORITY 3 (Advanced / Optional)

### 4a. Transmission & Congestion

| Dataset | Source | Granularity |
|---------|--------|-------------|
| **Path 15 / Path 26 scheduled flows** | OASIS | Hourly |
| **SP15 congestion revenue rights (CRRs)** | CAISO CRR auction | Monthly |

### 4b. Forward Market / Bilateral

| Dataset | Source | Granularity |
|---------|--------|-------------|
| **SP15 on-peak / off-peak forward prices** | ICE / brokers | Daily, Monthly strips |
| **SP15 heat rate** (implied) | Derived: SP15 power / SoCal gas | Daily |

### 4c. Demand-Side

| Dataset | Source | Notes |
|---------|--------|-------|
| **SCE customer count by rate class** | SCE / CPUC | Annual |
| **SCE demand response program enrollment** | SCE | Annual |

---

## 5. FILE FORMAT & NAMING CONVENTIONS

Please provide data as **CSV files** with the following conventions:

```
data/
├── load/
│   ├── caiso_actual_load_2022_2025.csv
│   └── sce_tac_load_2022_2025.csv
├── prices/
│   ├── sp15_dam_lmp_2022_2025.csv
│   ├── sp15_rtm_lmp_2023_2025.csv
│   ├── socal_citygate_gas_2022_2025.csv
│   ├── henry_hub_gas_2022_2025.csv
│   └── cca_carbon_2022_2025.csv
├── generation/
│   ├── caiso_solar_gen_2022_2025.csv
│   ├── caiso_wind_gen_2022_2025.csv
│   └── btm_solar_estimate_2022_2025.csv
├── weather/
│   └── sce_territory_weather_2022_2025.csv
└── forward_curves/
    ├── sp15_power_forward.csv
    ├── socal_gas_forward.csv
    └── henry_hub_forward.csv
```

**Minimum required columns:**
- `datetime` or `timestamp` (ISO 8601, UTC or Pacific with timezone label)
- `value` or domain-specific column (MW, $/MWh, $/MMBtu, etc.)

---

## 6. WHAT THE MODELS WILL PRODUCE

With the Priority 1 data alone, the framework will deliver:

1. **Demand Forecast** — 7-day ahead hourly CAISO/SCE load forecast using SARIMAX with
   weather, calendar, and lag features (target MAPE < 3%)
2. **Price Forecast** — Day-ahead SP15 hub LMP forecast using structural econometric model
   (gas price, load, renewables, congestion drivers)
3. **Hedging Analytics** — Volumetric exposure quantification, optimal hedge ratios,
   spark-spread analysis, and Value-at-Risk (VaR) for open positions

With Priority 2-3 data, the models additionally provide:
- Renewable-adjusted net load forecasting
- Carbon cost pass-through modeling
- Congestion risk premium estimation
- Forward curve vs fundamental fair-value analysis
