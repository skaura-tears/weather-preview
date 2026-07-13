"""Spearman 秩相关系数分析 — 6 个目标变量与 15 个特征（含特征工程）的相关性"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ── 读取数据 + 特征工程（与 train-model.py 一致）──
df = pd.read_csv('data/csv/清洗后数据.csv', parse_dates=['date'])

# build_features
df['doy_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
df['doy_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)
df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)
df['pressure_change_1d'] = df['surface_pressure'].diff(1).fillna(0)

# 6 个目标变量
targets = ['temperature_max', 'temperature_min', 'temperature_mean',
           'humidity_mean', 'wind_speed_max', 'precipitation']

# 15 个特征（= 10 原始数值 + 5 特征工程）
all_features = [
    'temperature_max', 'temperature_min', 'temperature_mean',
    'precipitation', 'wind_speed_max', 'humidity_mean',
    'surface_pressure', 'shortwave_radiation',
    'vapour_pressure_deficit', 'dew_point', 'wind_gusts_max',
    'wind_dir_cos', 'pressure_change_1d',
    'doy_sin', 'doy_cos',
]

print("=" * 72)
print("Spearman 秩相关系数分析（15 特征，含特征工程）")
print("=" * 72)

for target in targets:
    # 用除目标本身外的所有特征
    feats = [f for f in all_features if f != target]

    print(f"\n{'─' * 64}")
    print(f"【{target}】")
    print(f"{'─' * 64}")
    results = []
    for feat in feats:
        valid = df[[target, feat]].dropna()
        if len(valid) < 10:
            continue
        corr, p = spearmanr(valid[target], valid[feat])
        results.append((feat, corr, p))

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, corr, p in results:
        bar = '█' * int(abs(corr) * 20)
        sign = '+' if corr > 0 else ' '
        tag = ''
        if feat in ('doy_sin', 'doy_cos', 'wind_dir_sin', 'wind_dir_cos', 'pressure_change_1d'):
            tag = ' [★工程]'
        print(f"  {feat:<28s} {sign}{corr:6.3f}  {bar}{tag}")

print("\n" + "=" * 72)
print("[★工程] = 特征工程创建的变量")
print("完成")

