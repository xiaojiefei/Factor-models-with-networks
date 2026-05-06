import os
import time
import pandas as pd
from datetime import datetime, timedelta
import tushare as ts
import akshare as ak
import tushare.pro.client as client

# ======================
# 1. 配置与路径初始化
# ======================
TOKEN = '0f864583902803d86234572d3f1a7d6b1267aefe3ef2d22d4b8171e4'
client.DataApi._DataApi__http_url = "http://tushare.xyz"
# 调高 timeout 到 60 秒
pro = ts.pro_api(TOKEN, timeout=60)

DAILY_DIR = './data/raw/daily_data'
MIN5_DIR = './data/raw/min5_data'
os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(MIN5_DIR, exist_ok=True)

industry_codes = [f"{i:06d}" for i in range(32, 42)]

def get_min5_data_with_retry(ts_code, start_date, end_date, max_retries=3):
    """
    带有重试机制和分段下载的函数
    """
    all_dfs = []
    curr_start = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    final_end = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    
    while curr_start < final_end:
        step_end = min(curr_start + timedelta(days=30), final_end)
        s_str = curr_start.strftime('%Y-%m-%d %H:%M:%S')
        e_str = step_end.strftime('%Y-%m-%d %H:%M:%S')
        
        success = False
        for i in range(max_retries):
            try:
                df = pro.idx_mins(ts_code=ts_code, freq='5min', start_date=s_str, end_date=e_str)
                if df is not None:
                    if not df.empty:
                        all_dfs.append(df)
                    success = True
                    break
            except Exception as e:
                print(f"    [尝试 {i+1}] {ts_code} 请求超时或失败，正在重试...")
                time.sleep(2) # 失败后等2秒再试
        
        if not success:
            print(f"    [严重错误] {ts_code} 在 {s_str} 时间段多次请求失败，跳过该段数据")
            
        curr_start = step_end + timedelta(seconds=1)
        time.sleep(0.5) # 每次成功下载一段后稍微歇一下，防止被封
    
    return pd.concat(all_dfs).drop_duplicates() if all_dfs else pd.DataFrame()

def main():
    for code in industry_codes:
        print(f"正在处理指数: {code}...")
        
        # 1. Akshare 日度 (通常很稳)
        try:
            df_daily = ak.stock_zh_index_daily(symbol=f"sh{code}")
            if not df_daily.empty:
                df_daily.to_csv(os.path.join(DAILY_DIR, f"{code}_daily.csv"))
                print(f"  [成功] 日度数据已保存")
        except Exception as e:
            print(f"  [失败] 日度数据 {code}: {e}")

        # 2. Tushare 5分钟 (增加重试)
        try:
            df_min5 = get_min5_data_with_retry(
                ts_code=f"{code}.SH",
                start_date='2000-01-01 09:30:00',
                end_date='2026-04-24 15:00:00'
            )
            if not df_min5.empty:
                df_min5.to_csv(os.path.join(MIN5_DIR, f"{code}_5min.csv"), index=False)
                print(f"  [成功] 5分钟数据保存 (共{len(df_min5)}行)")
        except Exception as e:
            print(f"  [失败] 5分钟数据 {code}: {e}")

if __name__ == "__main__":
    main()