"""
整理Wind行业数据为统一CSV格式
生成周度收益率序列和GARCH(1,1)条件波动率序列供DY模型使用

数据管线:
  Wind .xlsx → 日度价格 → 周度重采样 → 周收益率 + GARCH周波动率
  → 下游 TVP-VAR-DY.R 读取, 生成时变网络
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from arch import arch_model

# 设置路径 - 直接使用相对路径
data_dir = Path("wind下载")
output_dir = Path(".")

def read_industry_data(file_path):
    """读取单个行业Excel文件"""
    df = pd.read_excel(file_path)
    return df

def extract_close_price(df, industry_name):
    """
    提取收盘价
    Wind数据格式：第一列是日期，第二列是收盘价（算数平均）
    """
    # 获取列名
    cols = df.columns.tolist()
    date_col = cols[0]  # 第一列是日期
    price_col = cols[1]  # 第二列是收盘价

    # 提取数据
    df_clean = df[[date_col, price_col]].copy()
    df_clean.columns = ['date', 'price']
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean = df_clean.sort_values('date')
    df_clean = df_clean.set_index('date')
    df_clean.columns = [industry_name]

    return df_clean

def process_all_industries():
    """处理所有行业数据"""

    # 获取所有Excel文件
    excel_files = list(data_dir.glob("*.xlsx"))
    print(f"找到 {len(excel_files)} 个行业文件")

    all_prices = []
    industry_names = []

    for file_path in sorted(excel_files):
        # 从文件名提取行业名称
        industry_name = file_path.stem.split('-')[0]
        print(f"处理: {industry_name}")

        try:
            df = read_industry_data(file_path)
            price_series = extract_close_price(df, industry_name)
            all_prices.append(price_series)
            industry_names.append(industry_name)
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 合并所有价格数据
    prices_df = pd.concat(all_prices, axis=1)
    prices_df = prices_df.sort_index()

    # 删除全为NA的列
    prices_df = prices_df.dropna(axis=1, how='all')

    print(f"\n价格数据形状: {prices_df.shape}")
    print(f"时间范围: {prices_df.index[0]} 到 {prices_df.index[-1]}")
    print(f"行业列表: {list(prices_df.columns)}")

    return prices_df

def calculate_returns(prices_df):
    """计算收益率（周度或日度，取决于输入频率）"""
    returns_df = prices_df.pct_change().dropna()
    return returns_df


def resample_to_weekly(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    将日度价格重采样为周度（每周最后一个交易日的收盘价）

    周频可大幅加速 TVP-VAR 估计，同时保留主要波动信息。
    使用 'W-FRI' 锚定每周五，取该周最后一个非空观测。

    Args:
        prices_df: 日度价格 DataFrame (date 索引, 列为行业)

    Returns:
        weekly_prices: 周度价格 DataFrame
    """
    weekly_prices = prices_df.resample('W-FRI').last().dropna(how='all')
    return weekly_prices

def calculate_volatility_garch(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 GARCH(1,1) 计算条件波动率（年化）

    优势:
    - 输出维度与输入收益率完全一致（无行损失）
    - 捕捉波动率聚集效应（volatility clustering）
    - 金融计量标准方法

    Args:
        returns_df: 收益率 DataFrame (date 索引, 列为行业)

    Returns:
        volatility_df: 年化条件波动率 DataFrame (与 returns_df 同维度)
    """
    volatility_df = pd.DataFrame(index=returns_df.index)
    n_industries = len(returns_df.columns)

    print(f"  使用 GARCH(1,1) 估计条件波动率 ({n_industries} 个行业)...")

    for i, col in enumerate(returns_df.columns):
        series = returns_df[col].dropna()
        # arch 库使用百分比收益率以提升数值稳定性
        series_pct = series * 100

        try:
            model = arch_model(
                series_pct,
                vol='Garch', p=1, q=1,
                mean='Constant',
                rescale=False
            )
            result = model.fit(disp='off', show_warning=False)
            # 条件波动率: 百分比 → 小数, 周度 → 年化
            cond_vol = result.conditional_volatility / 100 * np.sqrt(52)
            volatility_df[col] = cond_vol
        except Exception as e:
            # GARCH 不收敛时使用 EWMA 回退
            print(f"    ⚠ {col} GARCH不收敛, 使用EWMA回退: {e}")
            ewma_var = returns_df[col].ewm(span=4).var()
            volatility_df[col] = np.sqrt(ewma_var) * np.sqrt(52)

        if (i + 1) % 5 == 0 or i == 0 or i == n_industries - 1:
            print(f"    GARCH(1,1): {col} ({i+1}/{n_industries})")

    return volatility_df

def save_data_for_r(prices_df, returns_df, volatility_df):
    """保存数据供R代码使用"""

    # 创建输出目录
    output_path = output_dir / "processed"
    output_path.mkdir(exist_ok=True)

    # 1. 保存价格数据
    prices_df.to_csv(output_path / "industry_prices.csv")
    print(f"\n保存: {output_path / 'industry_prices.csv'}")

    # 2. 保存收益率数据
    returns_df.to_csv(output_path / "industry_returns.csv")
    print(f"保存: {output_path / 'industry_returns.csv'}")

    # 3. 保存波动率数据
    volatility_df.to_csv(output_path / "industry_volatility.csv")
    print(f"保存: {output_path / 'industry_volatility.csv'}")

    # 4. 保存行业名称列表
    with open(output_path / "industry_names.txt", 'w', encoding='utf-8') as f:
        for name in returns_df.columns:
            f.write(name + '\n')
    print(f"保存: {output_path / 'industry_names.txt'}")

    # 5. 保存基本信息
    info = {
        'n_industries': len(returns_df.columns),
        'n_observations_returns': len(returns_df),
        'n_observations_volatility': len(volatility_df),
        'frequency': 'weekly',
        'volatility_method': 'GARCH(1,1)',
        'start_date': str(returns_df.index[0]),
        'end_date': str(returns_df.index[-1]),
        'industries': list(returns_df.columns)
    }

    import json
    with open(output_path / "data_info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"保存: {output_path / 'data_info.json'}")

    return output_path

def main():
    print("="*60)
    print("Wind行业数据整理")
    print("="*60)

    # 1. 处理所有行业数据
    prices_df = process_all_industries()

    # 2. 周度重采样
    print("\n重采样为周度数据...")
    weekly_prices = resample_to_weekly(prices_df)
    print(f"周度价格数据形状: {weekly_prices.shape}")

    # 3. 计算周收益率
    print("\n计算周收益率...")
    returns_df = calculate_returns(weekly_prices)
    print(f"收益率数据形状: {returns_df.shape}")

    # 4. 计算波动率 (GARCH)
    print("\n计算GARCH(1,1)条件波动率...")
    volatility_df = calculate_volatility_garch(returns_df)
    print(f"波动率数据形状: {volatility_df.shape}")

    # 维度一致性检查
    assert returns_df.shape == volatility_df.shape, \
        f"维度不匹配: returns {returns_df.shape} vs volatility {volatility_df.shape}"
    assert (returns_df.index == volatility_df.index).all(), \
        "日期索引不一致"
    print(f"[OK] 收益率与波动率维度一致: {returns_df.shape}")

    # 5. 保存数据
    print("\n保存数据...")
    output_path = save_data_for_r(prices_df, returns_df, volatility_df)

    # 6. 输出统计信息
    print("\n" + "="*60)
    print("数据统计")
    print("="*60)
    print(f"行业数量: {len(returns_df.columns)}")
    print(f"观测期数: {len(returns_df)}")
    print(f"时间范围: {returns_df.index[0].strftime('%Y-%m-%d')} 至 {returns_df.index[-1].strftime('%Y-%m-%d')}")

    print("\n收益率统计（前5个行业）:")
    print(returns_df.iloc[:, :5].describe().round(4))

    print("\n波动率统计（前5个行业）:")
    print(volatility_df.iloc[:, :5].describe().round(4))

    print(f"\n数据已保存到: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
