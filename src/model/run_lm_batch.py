"""Batch LM jump detection + HAR-RV decomposition for all industry indices.

Reads 5-min and daily data from data/raw/, runs the LM+HAR pipeline
for each index, and saves results to data/raw/lm_results/.

Usage:
    python src/model/run_lm_batch.py
"""

import glob
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from lm_har import (
    _pivot_intraday_to_daily,
    compute_lm_jump_indicators,
    compute_daily_jump_stats,  
    compute_realized_volatility,
    decompose_jump_volatility,
    compute_downside_risk,
    assemble_and_export,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MIN5_DIR = os.path.join(PROJECT_DIR, "data", "raw", "min5_data")
DAILY_DIR = os.path.join(PROJECT_DIR, "data", "raw", "daily_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "raw", "lm_results")

WINDOW = 270
SIGNIFICANCE = 0.05
PERIODS_PER_DAY = 49
START_DATE = "2009-03-01 09:30:00"
START_DATE_DAILY = "20090301"


# ---------------------------------------------------------------------------
# Data loading (adapted for this dataset format)
# ---------------------------------------------------------------------------

def load_min5(path: str) -> pd.Series:
    """Load 5-min close prices, sorted by time, indexed by datetime strings."""
    df = pd.read_csv(path)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    invalid_count = int((df["close"] <= 0).sum())
    if invalid_count:
        logger.warning("Dropping %d non-positive close prices from %s", invalid_count, path)
    df = df[df["close"] > 0].sort_values("trade_time")
    prices = df.set_index("trade_time")["close"].dropna()
    prices.index = prices.index.astype(str)
    prices.name = "close"
    return prices


def load_daily(path: str) -> pd.Series:
    """Load daily close prices, indexed by YYYYMMDD strings."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    prices = df.set_index("date")["close"].dropna()
    return prices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single(code: str) -> pd.DataFrame:
    """Run LM+HAR pipeline for a single index code."""
    min5_path = os.path.join(MIN5_DIR, f"{code}_5min.csv")
    daily_path = os.path.join(DAILY_DIR, f"{code}_daily.csv")

    intraday_prices = load_min5(min5_path)
    daily_prices = load_daily(daily_path)
    daily_log_returns = np.log(daily_prices).diff(1)
    value_col = "close"

    jump_indicator, log_returns = compute_lm_jump_indicators(
        intraday_prices, START_DATE, WINDOW, SIGNIFICANCE,
    )

    (
        daily_jump_count, _, daily_jump_return_sum,
        abs_jump_pivot, signed_jump_pivot, jump_size_pivot,
    ) = compute_daily_jump_stats(jump_indicator, log_returns, value_col)

    rv_daily, rv_weekly, rv_monthly, return_pivot = compute_realized_volatility(
        log_returns, value_col,
    )

    vol_components = decompose_jump_volatility(
        rv_daily, daily_jump_count, abs_jump_pivot, return_pivot,
        signed_jump_pivot, jump_size_pivot, PERIODS_PER_DAY,
    )

    downside = compute_downside_risk(
        daily_jump_return_sum, daily_log_returns, START_DATE_DAILY,
    )

    output_path = os.path.join(OUTPUT_DIR, f"{code}_lm_har.csv")
    return assemble_and_export(
        vol_components, downside,
        rv_daily, rv_weekly, rv_monthly,
        daily_log_returns, START_DATE_DAILY, output_path,
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    min5_files = sorted(glob.glob(os.path.join(MIN5_DIR, "*_5min.csv")))
    codes = [os.path.basename(f).replace("_5min.csv", "") for f in min5_files]

    logger.info("Found %d indices: %s", len(codes), codes)
    logger.info("Output dir: %s", OUTPUT_DIR)

    for code in codes:
        t0 = time.time()
        try:
            result = run_single(code)
            logger.info(
                "%s: %d rows, %.1fs", code, len(result), time.time() - t0,
            )
        except Exception as e:
            logger.error("%s failed: %s", code, e)
            raise

    logger.info("All done. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
