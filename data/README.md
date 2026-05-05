# Data

## Sources

All electricity market data is sourced from the ENTSO-E Transparency Platform via the Fraunhofer ISE Energy-Charts API (`https://api.energy-charts.info`).

## Countries
AT, BE, CH, CZ, DE, DK, ES, FR, HU, IT, NL, NO, PL, SE, SK (15 countries)

## Period
2019-01-01 to 2024-12-31 (hourly resolution, ~52,600 observations per country)

## Features
- Day-ahead price (EUR/MWh)
- Total load (MW)
- Generation by fuel type (nuclear, coal, gas, wind, solar, hydro, other)
- Carbon intensity (kg CO2/MWh) — computed from fuel-specific generation shares and IPCC emission factors
- Cross-border flows (MW)
- Net transfer capacity (MW)
- TTF gas price (EUR/MWh)
- EU ETS carbon price (EUR/tCO2)

## Download Instructions

1. Install the data download dependencies: `pip install requests pandas`
2. Run the download script: `python src/dataset.py --download`
3. Raw data will be saved to `data/raw/`
4. Processed data will be saved to `data/processed/`

## Preprocessing
- Missing values (~2.3% of hourly records): linear interpolation for gaps ≤3 hours; weekly repeat for longer gaps
- Daily gas/ETS prices broadcast to hourly resolution
- All features normalized to zero mean and unit variance (training set statistics)
