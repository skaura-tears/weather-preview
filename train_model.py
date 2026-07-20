import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
import joblib
import os
import json
import pickle

# ============================ GPU 设备检测 ============================
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"DirectML GPU 已启用: {torch_directml.device_name(0)}")
except ImportError:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cuda':
        print(f"CUDA GPU 已启用: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到 GPU，使用 CPU")
FEATURE_COLS = [
    'temperature_max', 'temperature_min', 'temperature_mean',
    'precipitation', 'wind_speed_max', 'humidity_mean',
    'surface_pressure', 'shortwave_radiation',
    'vapour_pressure_deficit', 'dew_point', 'wind_gusts_max',#vapour_pressure_deficit反应空气干燥程度，越大越干燥。wind_gusts_max阵风
     'pressure_change_1d','doy_sin', 'doy_cos',
]
TARGET_COLS = ['temperature_max', 'temperature_min', 'temperature_mean',
               'humidity_mean', 'wind_speed_max', 'precipitation']
seq_len=7

# ============================ 特征工程 ============================
def build_features(df):
    df['doy_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)#numpy只接受弧度制这里括号就是把日期转化为角度后转化为弧度，弧度=角度*π/180
    df['doy_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)#映射成0-2π的数据来计算cos值，消除首尾数值断层
    df['pressure_change_1d'] = df['surface_pressure'].diff(1).fillna(0)#第一天的数据填0
    return df

# ============================ PyTorch 模型 ============================
class WeatherLSTM(nn.Module):
    """杂交架构：温度自回归解码 + 湿度/风速/降水直接多步预测"""
    def __init__(self, n_features, hidden1=64, hidden2=32, hidden3=16,
                 dropout_rate=0.3, output_steps=7, output_vars=6):
        super().__init__()
        # Shared Encoder
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Temperature branch: 自回归 decoder
        self.lstm3_temp = nn.LSTM(hidden2, hidden3, batch_first=True)
        self.decoder_lstm = nn.LSTM(3, hidden3, batch_first=True)
        self.head_temp = nn.Linear(hidden3, 3)

        # Other branch: 直接多步预测 (湿度+风速+降水)
        self.lstm3_other = nn.LSTM(hidden2, hidden3, batch_first=True)
        self.head_other = nn.Linear(hidden3, output_steps * 3)

        self.output_steps = output_steps
        self.output_vars = output_vars

    def forward(self, x):
        # Shared encoder
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Temperature: 自回归逐日解码
        x_temp, (h_t, c_t) = self.lstm3_temp(x)
        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, 1, 3, device=x.device)
        decoder_hidden = (h_t, c_t)

        temp_outputs = []
        for _ in range(self.output_steps):
            out, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            pred = self.head_temp(out)
            temp_outputs.append(pred)
            decoder_input = pred.detach()

        out_temp = torch.cat(temp_outputs, dim=1)  # (batch, 7, 3)

        # Other: 直接多步预测
        x_other, _ = self.lstm3_other(x)
        out_other = self.head_other(x_other[:, -1, :]).view(batch_size, self.output_steps, 3)

        return torch.cat([out_temp, out_other], dim=2)  # (batch, 7, 6)

# ============================ 训练模型 ============================
def train_model(skip_cv=False):
    print("开始训练模型...")
    print(f"当前设备: {DEVICE}")
    print(f"序列长度: {seq_len} 天")

    df = pd.read_csv('data/csv/清洗后数据.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 固定测试集（2023年）
    start_date, end_date = '2023-01-01', '2023-12-31'
    df = build_features(df).dropna()
    # 划分训练、验证集 、固定测试集
    train_val_df = df[(df['date'] < start_date) | (df['date'] > end_date)].copy()
    n=len(train_val_df)
    div=int((n*0.8))
    train_df,val_df=train_val_df[:div],train_val_df[div:]
    test_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    test_df.to_csv('data/csv/fixed_test.csv', index=False, encoding='utf-8-sig')
    if len(test_df) == 0:
        raise ValueError("固定测试集无数据")

    #训练集标准化
    scaler_X = StandardScaler()
    scaled_X_train = scaler_X.fit_transform(train_df[FEATURE_COLS])
    scaler_y = {}
    scaled_y_train = np.zeros((len(train_df), len(TARGET_COLS)))
    for i, col in enumerate(TARGET_COLS):
        data = train_df[[col]].values
        scaler = MinMaxScaler() if col == 'precipitation' else StandardScaler()
        scaled_y_train[:, i] = scaler.fit_transform(data).flatten()
        scaler_y[col] = scaler
    #验证集数据标准化
    scaled_X_val = scaler_X.transform(val_df[FEATURE_COLS])
    scaled_y_val = np.zeros((len(val_df), len(TARGET_COLS)))
    for i, col in enumerate(TARGET_COLS):
        data = val_df[[col]].values
        scaled_y_val[:, i] = scaler_y[col].transform(data).flatten()
    makedirs('data/models',exist_ok=True)
    joblib.dump(scaler_X, 'data/models/scaler_X.pkl')
    joblib.dump(scaler_y, 'data/models/scaler_y.pkl')

    scaled_X_test = scaler_X.transform(test_df[FEATURE_COLS])
    scaled_y_test = np.zeros((len(test_df), len(TARGET_COLS)))
    for i, col in enumerate(TARGET_COLS):
        scaled_y_test[:, i] = scaler_y[col].transform(test_df[[col]].values).flatten()
    # 构造序列
    def make_seq(X, y, seq_len=seq_len, n_future=7):
        Xs, ys = [], []
        for i in range(seq_len, len(X) - n_future + 1):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i:i + n_future])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)#这里是因为pytorch使用float32

    X_train, y_train = make_seq(scaled_X_train, scaled_y_train)
    X_val, y_val = make_seq(scaled_X_val, scaled_y_val)
    X_test_seq, y_test_seq = make_seq(scaled_X_test, scaled_y_test)
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 固定测试集序列数: {len(X_test_seq)}")
    # ---- 获取超参数 ----
    n_features = len(FEATURE_COLS)
    best_params_file = 'data/json/best_params.json'

    if skip_cv and os.path.exists(best_params_file):
        with open(best_params_file) as f:
            best_params = json.load(f)
        print(f"使用缓存参数: {best_params}")
    else:
        print("\n===== 超参数搜索 =====")
        net = NeuralNetRegressor(
            module=WeatherLSTM, module__n_features=n_features,
            criterion=nn.HuberLoss, criterion__delta=1.5,
            optimizer=optim.Adam, optimizer__lr=1.5e-4,
            max_epochs=50, batch_size=32, verbose=0, device=DEVICE,
        )
        param_grid = {
            'criterion__delta': [1.0, 1.5, 3.0],
            'optimizer__lr': [1e-4, 1.5e-4, 3e-4],
            'module__dropout_rate': [0.1, 0.2, 0.3],
            'batch_size': [16, 32],
        }
        def scorer(est, X, y):
            yp = est.predict(X)
            return -mean_absolute_error(y.reshape(-1, y.shape[-1]), yp.reshape(-1, yp.shape[-1]))

        search = RandomizedSearchCV(estimator=net, param_distributions=param_grid,
                                    n_iter=6, cv=TimeSeriesSplit(n_splits=3),
                                    scoring=scorer, random_state=22, n_jobs=1, verbose=1)#  n_iter=6：随机抽取6组参数来试
        search.fit(X_train, y_train)
        best_params = search.best_params_
        os.makedirs('data/json',exist_ok=True)
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)#保存最优参数
        print(f"最佳参数: {best_params}")
        print(f"最佳验证MAE: {-search.best_score_:.4f}")

    # ---- 最终训练 ----
    print("\n===== 最终训练 =====")
    X_train_all = np.concatenate([X_train, X_val])
    y_train_all = np.concatenate([y_train, y_val])
    n_train = len(X_train)

    def fixed_split(dataset, y=None):#这里的dataset是通过 skorch打包后的数据集对象
        cls = type(dataset)
        return cls(dataset.X[:n_train], dataset.y[:n_train]), cls(dataset.X[n_train:], dataset.y[n_train:])

    final_net = NeuralNetRegressor(
        module=WeatherLSTM, module__n_features=n_features,
        module__dropout_rate=best_params['module__dropout_rate'],
        criterion=nn.HuberLoss, criterion__delta=best_params['criterion__delta'],
        optimizer=optim.Adam, optimizer__lr=best_params['optimizer__lr'],
        max_epochs=80, batch_size=best_params['batch_size'],
        verbose=1, device=DEVICE, train_split=fixed_split,
        callbacks=[EarlyStopping(patience=5, threshold=1e-2, threshold_mode='abs')],
    )
    final_net.fit(X_train_all, y_train_all)

    # ---- 评估 ----
    y_pred_scaled = final_net.predict(X_test_seq)
    yt = y_test_seq.reshape(-1, y_test_seq.shape[-1])#拿到 y_test_seq的维度（元祖）后用索引拿到最后一维
    yp = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    current_results = {}
    for i, col in enumerate(TARGET_COLS):#i在这里代表的是列索引，后面把i当做列索引来用，位置决定用途
        true = scaler_y[col].inverse_transform(yt[:, i].reshape(-1, 1)).flatten()
        pred = scaler_y[col].inverse_transform(yp[:, i].reshape(-1, 1)).flatten()
        current_results[col] = round(float(np.mean(np.abs(true - pred))), 4)#numpy不能用于json序列化，round保留4位小数
    current_avg_mae = float(round(np.mean(list(current_results.values())), 4))#float只接受一个参数把numpy.float数据转化为python原生float
    print(f"当前模型 MAE: {current_avg_mae}")
    for k, v in current_results.items():
        print(f"  {k}: {v}")

    # ---- 保存测试集预测（day-0，不重叠）用于前端对比图 ----
    test_dates = test_df['date'].iloc[seq_len:seq_len + len(X_test_seq)]
    test_pred_data = {
        'dates': test_dates.dt.strftime('%Y-%m-%d').tolist(),
        'actual': {},
        'predicted': {},
    }
    for i, col in enumerate(TARGET_COLS):
        true_day0 = scaler_y[col].inverse_transform(
            y_test_seq[:, 0, i].reshape(-1, 1)).flatten()
        pred_day0 = scaler_y[col].inverse_transform(
            y_pred_scaled[:, 0, i].reshape(-1, 1)).flatten()
        test_pred_data['actual'][col] = [round(float(v), 4) for v in true_day0]
        test_pred_data['predicted'][col] = [round(float(v), 4) for v in pred_day0]
    with open('data/json/current_test_pred.json', 'w', encoding='utf-8') as f:
        json.dump(test_pred_data, f, ensure_ascii=False)
    print(f"测试集预测数据已保存: {len(test_dates)} 个数据点")

    # ---- 保存 ----
    os.makedirs('static', exist_ok=True)
    hist = final_net.history
    with open('data/json/training_history.json', 'w', encoding='utf-8') as f:
        json.dump({
            'loss': [float(x) for x in hist[:, 'train_loss']],
            'val_loss': [float(x) for x in hist[:, 'valid_loss']],
        }, f, ensure_ascii=False, indent=2)

    final_net.set_params(train_split=None)
    with open('data/models/current_model.pkl', 'wb') as f:
        pickle.dump(final_net, f)

    pd.DataFrame([(k, v) for k, v in current_results.items()], columns=['Metric', 'MAE'])\
        .to_csv('data/csv/current_mae.csv', index=False, encoding='utf-8-sig')

    # ---- 对比历史最佳 ----
    best_mae_file = 'data/json/best_mae.json'
    best_mae = float('inf')
    if os.path.exists(best_mae_file):
        try:
            with open(best_mae_file) as f:
                best_mae = json.load(f).get('best_mae', float('inf'))
            print(f"历史最佳 MAE: {best_mae:.4f}")
        except (json.JSONDecodeError, TypeError):
            pass

    if current_avg_mae < best_mae:
        print(f"优于历史最佳 ({best_mae:.4f})，请在前端决定是否替换")
    elif best_mae != float('inf'):
        print(f"未优于历史最佳 ({best_mae:.4f})，仍可手动替换")

    if not os.path.exists('data/models/saved_weather_model.pkl'):
        print("首次训练，自动保存为历史最佳模型")
        with open('data/models/saved_weather_model.pkl', 'wb') as f:
            pickle.dump({'model': final_net, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
        with open(best_mae_file, 'w') as f:
            json.dump({'best_mae': float(current_avg_mae)}, f)
        pd.DataFrame([(k, v) for k, v in current_results.items()], columns=['Metric', 'MAE'])\
            .to_csv('data/csv/mae.csv', index=False, encoding='utf-8-sig')
        # 同步保存最佳模型的测试集预测
        import shutil
        if os.path.exists('data/json/current_test_pred.json'):
            shutil.copy('data/json/current_test_pred.json', 'data/json/best_test_pred.json')

    # 读取历史最佳模型的逐指标 MAE（用于前端对比）
    best_comparison = {}
    best_mae_csv = 'data/csv/mae.csv'
    if os.path.exists(best_mae_csv):
        try:
            best_df = pd.read_csv(best_mae_csv)
            for _, row in best_df.iterrows():#iterrows是把dataframe的数据一行一行取出来，返回行索引加该行数据
                best_comparison[row['Metric']] = float(row['MAE'])
        except Exception:
            pass

    # 保存待审批对比信息
    pending_data = {
        'current_avg_mae': float(current_avg_mae),
        'best_avg_mae': float(best_mae) if best_mae != float('inf') else None,
        'is_improved': bool(current_avg_mae < best_mae),
        'comparison': {
            'current': {k: float(v) for k, v in current_results.items()},
            'best': best_comparison
        }
    }
    with open('data/json/pending_model.json', 'w', encoding='utf-8') as f:
        json.dump(pending_data, f, ensure_ascii=False, indent=2)

    # 预测未来7天
    predict_future(model=final_net, scaler_X=scaler_X, scaler_y=scaler_y, df=df, seq_len=seq_len, features=FEATURE_COLS)
    print("\n训练完成！")

# ============================ 预测 ============================
def predict_future(model=None, scaler_X=None, scaler_y=None, df=None, seq_len=7, features=None):
    if model is None:
        if not os.path.exists('data/models/saved_weather_model.pkl'):
            print("未找到模型文件")
            return
        with open('data/models/saved_weather_model.pkl', 'rb') as f:
            saved = pickle.load(f)
        if isinstance(saved, dict):#判断save是否是字典格式
            model = saved['model']
            scaler_X = saved['scaler_X']
            scaler_y = saved['scaler_y']
        else:
            # 兼容旧格式（仅保存了模型，未打包缩放器）
            model = saved
            scaler_X = joblib.load('data/models/scaler_X.pkl')
            scaler_y = joblib.load('data/models/scaler_y.pkl')
        if hasattr(model, 'module_') and getattr(DEVICE, 'type', '') == 'cuda':
            for layer in model.module_.modules():
                if isinstance(layer, nn.LSTM):
                    layer.flatten_parameters()
        df = pd.read_csv('data/csv/清洗后数据.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)#drop=True代表丢弃旧索引
        df = build_features(df)

    scaled_X = scaler_X.transform(df[FEATURE_COLS])
    last_seq = scaled_X[-seq_len:].reshape(1, seq_len, len(FEATURE_COLS)).astype(np.float32)#pytorch默认float32
    pred_scaled = model.predict(last_seq)

    pred = np.zeros((7, len(TARGET_COLS)))
    for i, col in enumerate(TARGET_COLS):
        pred[:, i] = scaler_y[col].inverse_transform(pred_scaled[0, :, i].reshape(-1, 1)).flatten()
        if col == 'precipitation':
            pred[:, i] = np.maximum(pred[:, i], 0)#预测值如果小于0，强制输出0

    last_date = df['date'].iloc[-1]
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
    result = pd.DataFrame({
        'date': dates,
        'temp_max': pred[:, 0].round(1),
        'temp_min': pred[:, 1].round(1),
        'temp_avg': pred[:, 2].round(1),
        'humidity': pred[:, 3].round(0).astype(int),
        'wind': pred[:, 4].round(1),
        'precip': pred[:, 5].round(1),
    })

    print("\n未来7天天气预报:")
    for _, r in result.iterrows():
        print(f"{r['date'].strftime('%m-%d')} 最高:{r['temp_max']} 最低:{r['temp_min']} "
              f"平均:{r['temp_avg']} 湿度:{r['humidity']}% 风速:{r['wind']}km/h 降水:{r['precip']}mm")
    result.to_csv('data/csv/prediction_results.csv', index=False)
    return result

# ============================ 主菜单 ============================
def main():
    has_model = os.path.exists('data/models/saved_weather_model.pkl')
    if has_model:
        choice = input("1. 直接预测  2. 重新训练  3. 重新训练+超参数搜索\n请选择 (1/2/3): ")
    else:
        choice = input("未找到模型。\n2. 重新训练  3. 重新训练+超参数搜索\n请选择 (2/3): ")

    if choice == '1' :
        predict_future()
    elif choice == '2':
        train_model(skip_cv=True)
    elif choice == '3':
        train_model(skip_cv=False)

if __name__ == "__main__":
    main()
