"""
Download and preprocess Kenneth French data.

Downloads:
  - 48 Industry Portfolios (daily returns)
  - Fama-French 3 Factors + Momentum (daily)

Paper reference:
  Bonaccolto et al. (2019), Section 3.1
  - Period 1: 2006-01-03 ~ 2008-12-31 (755 trading days)
  - Period 2: 2011-01-03 ~ 2015-12-31 (1258 observations)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PERIOD_1 = ("2006-01-03", "2008-12-31")
PERIOD_2 = ("2011-01-03", "2015-12-31")


def download_french_data() -> None:
    """Download 48 Industry Portfolios and FF factors from Kenneth French."""
    try:
        import pandas_datareader.data as web
    except ImportError:
        raise ImportError(
            "pandas_datareader is required. Install with: "
            "pip install pandas-datareader"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 48 Industry Portfolios (daily, value-weighted returns)
    logger.info("Downloading 48 Industry Portfolios (daily)...")
    ind48 = web.DataReader(
        "48_Industry_Portfolios_daily", "famafrench", start="2005-01-01"
    )
    # ind48 is a dict; key 0 = value-weighted returns
    df_ind = ind48[0]
    df_ind.to_csv(RAW_DIR / "48_industry_portfolios_daily.csv")
    logger.info(f"Saved: {RAW_DIR / '48_industry_portfolios_daily.csv'}")
    logger.info(f"  Shape: {df_ind.shape}, Date range: {df_ind.index[0]} ~ {df_ind.index[-1]}")

    # Fama-French 3 Factors (daily)
    logger.info("Downloading Fama-French 3 Factors (daily)...")
    ff3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start="2005-01-01")
    df_ff3 = ff3[0]
    df_ff3.to_csv(RAW_DIR / "ff3_factors_daily.csv")
    logger.info(f"Saved: {RAW_DIR / 'ff3_factors_daily.csv'}")

    # Momentum Factor (daily)
    logger.info("Downloading Momentum Factor (daily)...")
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start="2005-01-01")
    df_mom = mom[0]
    df_mom.to_csv(RAW_DIR / "momentum_factor_daily.csv")
    logger.info(f"Saved: {RAW_DIR / 'momentum_factor_daily.csv'}")

    logger.info("All downloads complete.")


def load_and_preprocess() -> Dict[str, pd.DataFrame]:
    """
    Load raw CSV files and preprocess into analysis-ready DataFrames.

    Returns:
        Dictionary with keys:
          - 'returns_p1': Daily returns for period 1 (48 industries)
          - 'returns_p2': Daily returns for period 2
          - 'factors_p1': FF4 factors for period 1
          - 'factors_p2': FF4 factors for period 2
          - 'weekly_returns_p1': Weekly returns for period 1
          - 'weekly_returns_p2': Weekly returns for period 2
          - 'weekly_factors_p1': Weekly FF4 factors for period 1
          - 'weekly_factors_p2': Weekly FF4 factors for period 2
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    df_ind = pd.read_csv(
        RAW_DIR / "48_industry_portfolios_daily.csv",
        index_col=0, parse_dates=True
    )
    df_ff3 = pd.read_csv(
        RAW_DIR / "ff3_factors_daily.csv",
        index_col=0, parse_dates=True
    )
    df_mom = pd.read_csv(
        RAW_DIR / "momentum_factor_daily.csv",
        index_col=0, parse_dates=True
    )

    # Convert from percentage to decimal
    df_ind = df_ind / 100.0
    df_ff3 = df_ff3 / 100.0
    df_mom = df_mom / 100.0

    # Merge FF3 + Momentum → FF4
    df_factors = df_ff3[["Mkt-RF", "SMB", "HML"]].join(
        df_mom.rename(columns={df_mom.columns[0]: "MOM"}),
        how="inner"
    )
    # Also keep RF for computing excess returns if needed
    df_factors["RF"] = df_ff3["RF"]

    # Split by period
    results = {}

    for label, (start, end) in [("p1", PERIOD_1), ("p2", PERIOD_2)]:
        mask_ind = (df_ind.index >= start) & (df_ind.index <= end)
        mask_fac = (df_factors.index >= start) & (df_factors.index <= end)

        ret = df_ind.loc[mask_ind].copy()
        fac = df_factors.loc[mask_fac].copy()

        # Align dates
        common_dates = ret.index.intersection(fac.index)
        ret = ret.loc[common_dates]
        fac = fac.loc[common_dates]

        # Drop columns with all -99.99 or -999 (missing industry)
        ret = ret.loc[:, (ret > -0.9).all()]

        logger.info(f"Period {label}: {len(ret)} trading days, {ret.shape[1]} industries")

        results[f"returns_{label}"] = ret
        results[f"factors_{label}"] = fac

        # Weekly returns (Wednesday-to-Wednesday or business week)
        weekly_ret = ret.resample("W-FRI").apply(
            lambda x: (1 + x).prod() - 1
        )
        weekly_fac = fac.resample("W-FRI").apply(
            lambda x: (1 + x).prod() - 1 if x.name != "RF" else x.mean()
        )
        # Fix RF: use mean of daily RF for weekly
        weekly_fac["RF"] = fac["RF"].resample("W-FRI").mean()

        results[f"weekly_returns_{label}"] = weekly_ret
        results[f"weekly_factors_{label}"] = weekly_fac

        logger.info(f"  Weekly: {len(weekly_ret)} weeks")

    # Save processed data
    for key, df in results.items():
        path = PROCESSED_DIR / f"{key}.csv"
        df.to_csv(path)
        logger.info(f"Saved: {path}")

    return results


def compute_descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for comparison with paper Table C.5-C.6.

    Args:
        returns: DataFrame of daily returns (T × K).

    Returns:
        DataFrame with statistics per asset.
    """
    stats = pd.DataFrame(index=returns.columns)
    stats["Mean"] = returns.mean()
    stats["Std"] = returns.std()
    stats["Min"] = returns.min()
    stats["Q1"] = returns.quantile(0.25)
    stats["Median"] = returns.median()
    stats["Q3"] = returns.quantile(0.75)
    stats["Max"] = returns.max()
    stats["Skewness"] = returns.skew()
    stats["Kurtosis"] = returns.kurtosis()
    stats["IQR"] = stats["Q3"] - stats["Q1"]
    stats["N"] = returns.count()
    return stats


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("=== Step 1: Download data from Kenneth French ===")
    download_french_data()

    logger.info("\n=== Step 2: Load and preprocess ===")
    data = load_and_preprocess()

    logger.info("\n=== Step 3: Descriptive statistics ===")
    for period in ["p1", "p2"]:
        stats = compute_descriptive_stats(data[f"returns_{period}"])
        stats_path = PROCESSED_DIR / f"descriptive_stats_{period}.csv"
        stats.to_csv(stats_path)
        logger.info(f"\nPeriod {period} summary:")
        logger.info(f"\n{stats.describe().round(6)}")
