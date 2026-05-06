"""Lee-Mykland (2008) jump detection + HAR-RV volatility decomposition.

Reads intraday and daily price data, detects jumps via the LM test statistic,
decomposes realized volatility into continuous/jump (positive & negative) components,
computes downside risk measures, and exports HAR model features to CSV.

Usage:
    python src/model/lm_har.py
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1: Helper
# ---------------------------------------------------------------------------

def _pivot_intraday_to_daily(
    series: pd.Series,
    value_col: str,
) -> pd.DataFrame:
    """Reshape an intraday Series (datetime-string index) into (time_of_day x trading_day).

    Args:
        series: Intraday data indexed by datetime strings like '2023-03-01 09:35:00'.
        value_col: Column name for the values in the pivot.

    Returns:
        DataFrame with rows = intraday time slots, columns = trading days (YYYYMMDD).
    """
    df = series.reset_index()
    df.columns = ["datetime_str", value_col]
    dt = pd.to_datetime(df["datetime_str"])
    df["trading_day"] = dt.dt.strftime("%Y%m%d")
    df["intraday_time"] = dt.dt.strftime("%H:%M:%S")
    pivot = df.pivot_table(
        index="intraday_time",
        columns="trading_day",
        values=value_col,
        aggfunc="first",
    )
    time_order = df["intraday_time"].unique()
    pivot = pivot.reindex(time_order)
    return pivot


# ---------------------------------------------------------------------------
# Section 2: Data loading
# ---------------------------------------------------------------------------

def load_data(
    intraday_path: str,
    daily_path: str,
    price_col_index: int = 3,
    daily_col_index: int = 0,
    daily_date_format: str = "%d/%m/%Y",
) -> tuple[pd.Series, pd.Series]:
    """Load intraday and daily price series.

    Args:
        intraday_path: Path to the intraday Excel file.
        daily_path: Path to the daily CSV file.
        price_col_index: Column index for intraday price.
        daily_col_index: Column index for daily price.
        daily_date_format: strptime format for daily CSV date column.

    Returns:
        intraday_prices: Intraday price Series indexed by datetime strings.
        daily_prices: Daily price Series indexed by YYYYMMDD strings.
    """
    # Intraday data (高频数据)
    data = pd.read_excel(intraday_path)
    time_col = data.columns[0]  # '时间' column
    data = data.rename(columns={time_col: "日期"})
    data["日期"] = data["日期"].astype(str)
    data = data.set_index("日期")
    intraday_prices = data.iloc[:, price_col_index].dropna()

    # Daily data (日线数据)
    data_day = pd.read_csv(daily_path)
    date_col = data_day.columns[0]  # '日期' column
    data_day[date_col] = pd.to_datetime(
        data_day[date_col], format=daily_date_format
    ).dt.strftime("%Y%m%d")
    data_day = data_day.set_index(date_col)
    daily_prices = data_day.iloc[:, daily_col_index].dropna()

    return intraday_prices, daily_prices


# ---------------------------------------------------------------------------
# Section 3: LM jump detection
# ---------------------------------------------------------------------------

def compute_lm_jump_indicators(
    intraday_prices: pd.Series,
    start_date: str,
    window: int = 270,
    significance: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    """Compute Lee-Mykland (2008) jump indicators from intraday prices.

    Steps:
        1. Log returns: r_t = ln(P_t / P_{t-1})
        2. Bipower variation: BV_t = |r_t| * |r_{t-1}|, rolling mean over (window-1)
        3. LM statistic: L_t = r_t / sigma_{t-1}
        4. Gumbel standardization: T_t = (|L_t| - C_n) / S_n
        5. Jump if T_t > beta_star at given significance

    Args:
        intraday_prices: Intraday price series.
        start_date: Start datetime for counting n (e.g. '2023-03-01 09:30:00').
        window: Rolling window for bipower variation (default: 270).
        significance: Significance level (default: 0.05).

    Returns:
        jump_indicator: Series of {-1, 0, +1, NaN} for jump direction.
        log_returns: Full intraday log-return series.
    """
    log_returns = np.log(intraday_prices).diff(1)
    r_subset = log_returns.loc[start_date:]
    n = len(r_subset)

    sign = np.sign(log_returns)

    # Bipower variation and instantaneous volatility
    bv = log_returns.abs() * log_returns.shift(1).abs()
    sigma = np.sqrt(bv.rolling(window - 1).mean())

    # LM test statistic
    lm_stat = log_returns / sigma.shift(1)

    # Gumbel distribution parameters
    c = (2 / np.pi) ** 0.5
    s_n = 1 / (c * (2 * np.log(n)) ** 0.5)
    c_n = (
        (2 * np.log(n)) ** 0.5 / c
        - np.log(np.pi * np.log(n)) / (2 * c * (2 * np.log(n)) ** 0.5)
    )
    beta_star = -np.log(-np.log(1 - significance))

    # Standardized test statistic
    test_stat = (lm_stat.abs() - c_n) / s_n

    # Vectorized jump detection (no aliasing, NaN propagation automatic)
    is_jump = test_stat > beta_star
    jump_indicator = is_jump.astype(float) * sign
    jump_indicator[~is_jump & test_stat.notna()] = 0.0

    return jump_indicator, log_returns


# ---------------------------------------------------------------------------
# Section 4: Daily jump statistics
# ---------------------------------------------------------------------------

def compute_daily_jump_stats(
    jump_indicator: pd.Series,
    log_returns: pd.Series,
    value_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute daily jump counts, squared sizes, and return sums.

    Args:
        jump_indicator: Intraday jump indicators {-1, 0, +1, NaN}.
        log_returns: Intraday log-return series.
        value_col: Column name for pivot tables.

    Returns:
        daily_jump_count: Number of jumps per trading day.
        daily_jump_sq_sum: Sum of squared jump sizes per day.
        daily_jump_return_sum: Sum of jump returns per day.
        abs_jump_pivot: (time x day) absolute jump indicators.
        signed_jump_pivot: (time x day) signed jump indicators.
        jump_size_pivot: (time x day) jump sizes = |J| * r.
    """
    abs_jump_pivot = _pivot_intraday_to_daily(jump_indicator.abs(), value_col)
    signed_jump_pivot = _pivot_intraday_to_daily(jump_indicator, value_col)

    jump_size = jump_indicator.abs() * log_returns
    jump_size_pivot = _pivot_intraday_to_daily(jump_size, value_col)

    daily_jump_count = (signed_jump_pivot ** 2).sum()
    daily_jump_sq_sum = (jump_size_pivot ** 2).sum()
    daily_jump_return_sum = jump_size_pivot.sum()

    return (
        daily_jump_count,
        daily_jump_sq_sum,
        daily_jump_return_sum,
        abs_jump_pivot,
        signed_jump_pivot,
        jump_size_pivot,
    )


# ---------------------------------------------------------------------------
# Section 5: Realized volatility
# ---------------------------------------------------------------------------

def compute_realized_volatility(
    log_returns: pd.Series,
    value_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Compute daily, weekly, and monthly realized volatility.

    RV_daily = sum(r_i^2) over intraday periods.
    RV_weekly = 5-day rolling mean.   RV_monthly = 22-day rolling mean.

    Args:
        log_returns: Intraday log-return series.
        value_col: Column name for pivot.

    Returns:
        rv_daily, rv_weekly, rv_monthly: RV at three horizons.
        return_pivot: (time x day) intraday returns for decomposition.
    """
    return_pivot = _pivot_intraday_to_daily(log_returns, value_col)

    rv_daily = (return_pivot ** 2).sum()
    rv_weekly = rv_daily.rolling(5).mean()
    rv_monthly = rv_daily.rolling(22).mean()

    return rv_daily, rv_weekly, rv_monthly, return_pivot


# ---------------------------------------------------------------------------
# Section 6: Jump volatility decomposition
# ---------------------------------------------------------------------------

def decompose_jump_volatility(
    rv_daily: pd.Series,
    daily_jump_count: pd.Series,
    abs_jump_pivot: pd.DataFrame,
    return_pivot: pd.DataFrame,
    signed_jump_pivot: pd.DataFrame,
    jump_size_pivot: pd.DataFrame,
    periods_per_day: int = 48,
) -> dict[str, pd.Series]:
    """Decompose RV into continuous (CSV) and jump (JSV) components.

    Methodology (Tauchen-Zhou 2011):
        1. meanvol = average r^2 of non-jump periods
        2. PJ_{t,i} = jump_size^2 - meanvol  (purified jump volatility)
        3. JSV = sum(PJ)           CSV = RV - JSV
        4. Split JSV into positive (upward) and negative (downward) by jump sign

    Args:
        rv_daily: Daily realized volatility.
        daily_jump_count: Number of jumps per day.
        abs_jump_pivot: (time x day) absolute jump indicators.
        return_pivot: (time x day) intraday returns.
        signed_jump_pivot: (time x day) signed indicators.
        jump_size_pivot: (time x day) jump sizes.
        periods_per_day: Intraday periods per day (default: 48).

    Returns:
        Dictionary with keys: csv_d/w/m, jsv_pos_d/w/m, jsv_neg_d/w/m.
    """
    no_jump_count = periods_per_day - daily_jump_count
    no_jump_r2 = ((1 - abs_jump_pivot) * (return_pivot ** 2)).sum()
    meanvol = no_jump_r2 / no_jump_count

    # Purified jump volatility (vectorized, replaces nested loop)
    k2 = jump_size_pivot ** 2
    has_jump = k2 > 0
    meanvol_aligned = meanvol.reindex(k2.columns)
    pj = k2.copy()
    pj[has_jump] = k2[has_jump].subtract(meanvol_aligned, axis="columns")
    pj[~has_jump & k2.notna()] = 0.0

    jsv_daily = pj.sum()
    csv_daily = rv_daily - jsv_daily

    # Split by jump sign (vectorized, replaces nested loop)
    jsv_pre = signed_jump_pivot * pj
    jsv_pos_daily = jsv_pre.clip(lower=0).sum()
    jsv_neg_daily = (-jsv_pre).clip(lower=0).sum()

    return {
        "csv_d": csv_daily,
        "csv_w": csv_daily.rolling(5).mean(),
        "csv_m": csv_daily.rolling(22).mean(),
        "jsv_pos_d": jsv_pos_daily,
        "jsv_pos_w": jsv_pos_daily.rolling(5).mean(),
        "jsv_pos_m": jsv_pos_daily.rolling(22).mean(),
        "jsv_neg_d": jsv_neg_daily,
        "jsv_neg_w": jsv_neg_daily.rolling(5).mean(),
        "jsv_neg_m": jsv_neg_daily.rolling(22).mean(),
    }


# ---------------------------------------------------------------------------
# Section 7: Downside risk
# ---------------------------------------------------------------------------

def compute_downside_risk(
    daily_jump_return_sum: pd.Series,
    daily_log_returns: pd.Series,
    start_date_daily: str,
) -> dict[str, pd.Series]:
    """Compute downside (negative) jump and continuous returns with 1-day lag.

    Args:
        daily_jump_return_sum: Sum of jump returns per day.
        daily_log_returns: Daily log-return series.
        start_date_daily: Start date in YYYYMMDD format.

    Returns:
        Dictionary with keys: jrt_d/w/m, crt_d/w/m.
    """
    jrt = daily_jump_return_sum.clip(upper=0)
    crt = (daily_log_returns - daily_jump_return_sum).clip(upper=0)

    def _lag_and_trim(s: pd.Series) -> pd.Series:
        return s.shift(1).loc[start_date_daily:]

    return {
        "jrt_d": _lag_and_trim(jrt),
        "jrt_w": _lag_and_trim(jrt.rolling(5).mean()),
        "jrt_m": _lag_and_trim(jrt.rolling(22).mean()),
        "crt_d": _lag_and_trim(crt),
        "crt_w": _lag_and_trim(crt.rolling(5).mean()),
        "crt_m": _lag_and_trim(crt.rolling(22).mean()),
    }


# ---------------------------------------------------------------------------
# Section 8: Assembly and export
# ---------------------------------------------------------------------------

def assemble_and_export(
    vol_components: dict[str, pd.Series],
    downside: dict[str, pd.Series],
    rv_daily: pd.Series,
    rv_weekly: pd.Series,
    rv_monthly: pd.Series,
    daily_log_returns: pd.Series,
    start_date_daily: str,
    output_path: str,
) -> pd.DataFrame:
    """Assemble all HAR-RV features, lag by 1 day, and export to CSV.

    Args:
        vol_components: Output of decompose_jump_volatility().
        downside: Output of compute_downside_risk().
        rv_daily, rv_weekly, rv_monthly: Realized volatility at three horizons.
        daily_log_returns: Daily log-return series.
        start_date_daily: Start date in YYYYMMDD format.
        output_path: Output CSV path.

    Returns:
        Assembled DataFrame (also saved to output_path).
    """
    # Volatility components — lag by 1 day
    vol_df = pd.DataFrame(vol_components).shift(1).loc[start_date_daily:]

    # RV — lag by 1 day
    rv_lagged = pd.DataFrame({
        "RV_sum_d": rv_daily.shift(1),
        "RV_sum_w": rv_weekly.shift(1),
        "RV_sum_m": rv_monthly.shift(1),
    }).loc[start_date_daily:]

    # Current-day RV and returns (not lagged) — use explicit names to avoid
    # pd.concat merging Series with colliding .name attributes
    extras = pd.DataFrame({
        "RV_sum_now": rv_daily.loc[start_date_daily:],
        "r_now": daily_log_returns.loc[start_date_daily:],
        "r_d": daily_log_returns.shift(1).loc[start_date_daily:],
    })

    # Downside risk (already lagged inside compute_downside_risk)
    down_df = pd.DataFrame(downside)

    result = pd.concat([vol_df, rv_lagged, extras, down_df], axis=1)
    result.columns = [
        "CSVt_d", "CSVt_w", "CSVt_m",
        "JSVt_zheng_d", "JSVt_zheng_w", "JSVt_zheng_m",
        "JSVt_fu_d", "JSVt_fu_w", "JSVt_fu_m",
        "RV_sum_d", "RV_sum_w", "RV_sum_m",
        "RV_sum_now", "r_now", "r_d",
        "jrt_d", "jrt_w", "jrt_m",
        "crt_d", "crt_w", "crt_m",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path)
    logger.info("Saved %d rows to %s", len(result), output_path)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    intraday_path: str = "./data/上证50高频数据.xlsx",
    daily_path: str = "./data/上证50指数历史数据.csv",
    output_path: str = "./data/LM_HAR_volatility.csv",
    start_date: str = "2023-03-01 09:30:00",
    start_date_daily: str = "20230301",
    window: int = 270,
    periods_per_day: int = 48,
    significance: float = 0.05,
    price_col_index: int = 3,
    daily_col_index: int = 0,
) -> pd.DataFrame:
    """Run the full LM jump detection + HAR-RV decomposition pipeline."""
    intraday_prices, daily_prices = load_data(
        intraday_path, daily_path, price_col_index, daily_col_index,
    )
    value_col = intraday_prices.name or "price"
    daily_log_returns = np.log(daily_prices).diff(1)

    jump_indicator, log_returns = compute_lm_jump_indicators(
        intraday_prices, start_date, window, significance,
    )

    (
        daily_jump_count,
        daily_jump_sq_sum,
        daily_jump_return_sum,
        abs_jump_pivot,
        signed_jump_pivot,
        jump_size_pivot,
    ) = compute_daily_jump_stats(jump_indicator, log_returns, value_col)

    rv_daily, rv_weekly, rv_monthly, return_pivot = compute_realized_volatility(
        log_returns, value_col,
    )

    vol_components = decompose_jump_volatility(
        rv_daily, daily_jump_count, abs_jump_pivot, return_pivot,
        signed_jump_pivot, jump_size_pivot, periods_per_day,
    )

    downside = compute_downside_risk(
        daily_jump_return_sum, daily_log_returns, start_date_daily,
    )

    return assemble_and_export(
        vol_components, downside,
        rv_daily, rv_weekly, rv_monthly,
        daily_log_returns, start_date_daily, output_path,
    )


if __name__ == "__main__":
    main()
