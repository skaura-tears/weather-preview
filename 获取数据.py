import numpy as np
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta

# 设置南阳市的经纬度（南阳中心坐标）
LATITUDE = 32.9907
LONGITUDE = 112.5283

# 配置API客户端（带缓存和重试机制）
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
def preprocess_weather_data(df):
    """
    对爬取的原始天气数据进行预处理：g    +..
    1. 日期格式化和排序
    2. 缺失值插值
    3. 温度和湿度合理性检查
    4. 剩余缺失值填充
    """
    # 复制数据，避免修改原始引用
    df_processed = df.copy()

    # 确保日期列正确并按时间排序
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed = df_processed.sort_values('date').reset_index(drop=True)

    # 需要处理的数值列
    numeric_cols = ['temperature_max', 'temperature_min', 'temperature_mean',
                    'precipitation', 'wind_speed_max', 'humidity_mean',
                    'surface_pressure', 'shortwave_radiation',
                    'vapour_pressure_deficit', 'dew_point',
                    'wind_direction', 'wind_gusts_max']
    # 只保留实际存在的列
    numeric_cols = [col for col in numeric_cols if col in df_processed.columns]

    # 1. 缺失值线性插值
    df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(
        method='linear', limit_direction='both'#对首个空缺值不做处理，线性插值要满足前后都有数据的条件
    )

    # 2. 温度和湿度合理性检查（超出合理范围设为NaN，再插值）
    # 温度合理范围：-25℃ ~ 48℃
    temp_cols = ['temperature_max', 'temperature_min', 'temperature_mean']
    temp_cols = [col for col in temp_cols if col in df_processed.columns]
    for col in temp_cols:
        invalid = (df_processed[col] < -25) | (df_processed[col] > 48)
        if invalid.any():
            print(f"   {col} 中发现 {invalid.sum()} 条不合理值，已置为缺失")
            df_processed.loc[invalid, col] = np.nan#loc接受bool型变量的时候会选中true对应的行

    # 湿度合理范围：0~100%
    if 'humidity_mean' in df_processed.columns:
        invalid_hum = (df_processed['humidity_mean'] < 0) | (df_processed['humidity_mean'] > 100)
        if invalid_hum.any():
            print(f"   湿度中发现 {invalid_hum.sum()} 条不合理值，已置为缺失")
            df_processed.loc[invalid_hum, 'humidity_mean'] = np.nan

    # 再次插值处理新产生的缺失值
    df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(
        method='linear', limit_direction='both'
    )

    # 3. 若首尾仍有缺失，使用前向/后向填充
    if df_processed[numeric_cols].isnull().any().any():#第一个any返回series索引，第二个any对series进行判断返回bool型，判断这个dataframe里面是否有nan值
        df_processed[numeric_cols] = df_processed[numeric_cols].ffill().bfill()

    return df_processed

def fetch_nanyang_weather(start_date, end_date):
    """
    获取南阳市指定日期范围的天气数据，并自动进行预处理
    """
    # API参数设置（保持不变）
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                  "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean",
                  "shortwave_radiation_sum", "vapour_pressure_deficit_max",
                  "wind_direction_10m_dominant", "wind_gusts_10m_max"],
        "hourly": ["surface_pressure", "dew_point_2m"],  # Archive 不支持日级，用小时级聚合
        "timezone": "Asia/Shanghai"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]#取单城市即南阳市的数据

    print(f"获取数据成功！站点: {response.Latitude()}°N, {response.Longitude()}°E, 海拔: {response.Elevation()} m")

    daily = response.Daily()#从api中获取的数据中提取日级别的气象数据
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),#生成一个时间频率对象freq，标记当前时间序列是日频
            inclusive="left"#表示区间是左闭右开
        ),#这里date的value数据类型是Datetimeindex
        "temperature_max": daily.Variables(0).ValuesAsNumpy(),
        "temperature_min": daily.Variables(1).ValuesAsNumpy(),
        "temperature_mean": daily.Variables(2).ValuesAsNumpy(),
        "precipitation": daily.Variables(3).ValuesAsNumpy(),
        "wind_speed_max": daily.Variables(4).ValuesAsNumpy(),
        "humidity_mean": daily.Variables(5).ValuesAsNumpy(),
        "shortwave_radiation": daily.Variables(6).ValuesAsNumpy(),
        "vapour_pressure_deficit": daily.Variables(7).ValuesAsNumpy(),#空气还能吸多少水，反应空气干燥程度
        "wind_direction": daily.Variables(8).ValuesAsNumpy(),
        "wind_gusts_max": daily.Variables(9).ValuesAsNumpy(),  # 阵风
    }

    # ---- 小时级数据 → 聚合成日均值 ----
    hourly = response.Hourly()
    hourly_pressure = hourly.Variables(0).ValuesAsNumpy()  # (n_days * 24,)
    hourly_dewpoint = hourly.Variables(1).ValuesAsNumpy()
    n_days = len(daily_data["date"])
    daily_pressure = hourly_pressure.reshape(n_days, 24).mean(axis=1)  # 日均气压这里会先把一维的numpy重塑为2维然后再求平均压缩为1维
    daily_dewpoint = hourly_dewpoint.reshape(n_days, 24).mean(axis=1)   # 日均露点温度
    daily_data["surface_pressure"] = daily_pressure
    daily_data["dew_point"] = daily_dewpoint

    df_raw = pd.DataFrame(data=daily_data)
    print(f"原始数据记录数：{len(df_raw)}")
    os.makedirs('data/csv',exist_ok=True)
    df_raw.to_csv('data/csv/原始数据.csv', index=False)#index=False表示保存的时候去掉多余的数字列

    # 调用预处理函数
    df_clean = preprocess_weather_data(df_raw)
    print(f"预处理后数据记录数：{len(df_clean)}")

    return df_clean



if __name__ == "__main__":
    # 计算日期范围（近5年）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=15 * 365)).strftime('%Y-%m-%d')

    print(f"开始获取南阳市天气数据: {start_date} 至 {end_date}")

    # 获取数据
    df_weather = fetch_nanyang_weather(start_date, end_date)

    # 保存为CSV
    df_weather.to_csv('data/csv/清洗后数据.csv', index=False)
    print(type(df_weather['temperature_max'][2]))
    print(f"数据已保存，共 {len(df_weather)} 条记录")

    # 查看前几行
    print("\n数据预览：")
    print(df_weather.head())

