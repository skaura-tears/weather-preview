from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
import sys
import pickle
import joblib
from datetime import datetime

# 注册 WeatherLSTM 到 __main__，让 pickle 能反序列化 train-model.py 保存的模型
from agent.tools import WeatherLSTM
sys.modules['__main__'].WeatherLSTM = WeatherLSTM

app = Flask(__name__)

MODEL_EVALUATION_DATA = None


def initialize_model_evaluation_data():
    """应用启动时预加载模型评估数据到内存"""
    global MODEL_EVALUATION_DATA

    try:
        print("[OK] 预加载模型评估数据到内存...")

        # 读取训练历史文件
        if os.path.exists('data/json/training_history.json'):
            with open('data/json/training_history.json', 'r', encoding='utf-8') as f:
                training_history = json.load(f)

            print(f"[OK] 成功加载训练历史数据，包含 {len(training_history.get('loss', training_history.get('train_loss', [])))} 个epochs")

            # 准备返回的数据结构
            MODEL_EVALUATION_DATA = {
                'loss_history': training_history,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            print("[OK] 模型评估数据已缓存到内存")
        else:
            print("[WARN] training_history.json 文件不存在，使用空数据")
            MODEL_EVALUATION_DATA = {
                'loss_history': {'loss': [], 'val_loss': []},
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    except Exception as e:
        print(f"[ERROR] 预加载模型评估数据失败: {e}")
        MODEL_EVALUATION_DATA = {
            'loss_history': {'loss': [], 'val_loss': []},
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        }


@app.route('/model_evaluation_data')
def get_model_evaluation_data():
    """快速返回内存中的模型评估数据"""
    if MODEL_EVALUATION_DATA is None:
        # 如果数据还未加载，尝试加载一次
        initialize_model_evaluation_data()

    if MODEL_EVALUATION_DATA:
        return jsonify(MODEL_EVALUATION_DATA)

    # 如果加载失败，返回错误
    return jsonify({
        'error': '模型评估数据未准备好，请刷新页面',
        'suggestion': '如果问题持续，请检查training_history.json文件'
    }), 503


# 首页路由 - 根据访问路径返回不同的页面
@app.route('/')
@app.route('/model_evaluation.html')
def model_evaluation():
    return render_template('model_evaluation.html')


@app.route('/weather_prediction.html')
def weather_prediction():
    return render_template('weather_prediction.html')


@app.route('/historical_analysis.html')
def historical_analysis():
    return render_template('historical_analysis.html')


# 预测数据API端点
@app.route('/prediction_data')
def get_prediction_data():
    try:
        pred_df = pd.read_csv('data/csv/prediction_results.csv', encoding='utf-8-sig')

        # 确保日期格式正确
        pred_df['date'] = pd.to_datetime(pred_df['date']).dt.strftime('%Y-%m-%d')

        # 准备预测数据
        prediction_data = {
            'prediction_dates': pred_df['date'].tolist(),
            'temperature_max': pred_df['temp_max'].tolist(),
            'temperature_min': pred_df['temp_min'].tolist(),
            'temperature_mean': pred_df['temp_avg'].tolist(),
            'humidity_mean': pred_df['humidity'].tolist(),
            'precipitation': pred_df['precip'].tolist(),
            'wind_speed_max': pred_df['wind'].tolist()
        }

        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({'error': f'加载预测数据失败: {str(e)}'}), 500


# 历史数据API端点
@app.route('/historical_data')
def get_historical_data():
    try:
        # 获取筛选参数
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        feature = request.args.get('feature', 'temperature_mean')  # 默认平均温度

        # 读取历史数据
        df = pd.read_csv('data/csv/清洗后数据.csv', encoding='utf-8-sig')

        # 校验特征参数，无效时退回默认值
        if feature not in df.columns:
            feature = 'temperature_mean'

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 应用日期筛选
        if start_date and end_date:
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df = df.loc[mask]

        # 按日期排序
        df = df.sort_values('date')

        # 准备返回数据
        data = {
            'dates': df['date'].tolist(),
            'feature_data': df[feature].tolist(),
            'feature_name': feature
        }

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'读取历史数据失败: {str(e)}'}), 500


# 健康检查端点
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'available_endpoints': [
            '/model_evaluation_data',
            '/prediction_data',
            '/historical_data'
        ]
    })
# ==================== 模型管理 API ====================

@app.route('/api/pending_model')
def get_pending_model_info():
    """返回待定模型与历史最佳的对比信息"""
    try:
        if os.path.exists('data/json/pending_model.json'):
            with open('data/json/pending_model.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            # 无待定模型时，回退到 mae.csv 展示当前最佳模型的 MAE
            if os.path.exists('data/csv/mae.csv'):
                mae_df = pd.read_csv('data/csv/mae.csv', encoding='utf-8-sig')
                mae_dict = {}
                for _, row in mae_df.iterrows():
                    mae_dict[row['Metric']] = float(row['MAE'])
                avg = round(sum(mae_dict.values()) / len(mae_dict), 4) if mae_dict else None
                return jsonify({
                    'current_avg_mae': avg,
                    'best_avg_mae': None,   # null → 前端隐藏替换按钮
                    'comparison': {
                        'best': mae_dict,
                        'current': mae_dict
                    }
                })
            return jsonify({'current_avg_mae': None, 'message': '暂无模型数据'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/replace_best_model', methods=['POST'])
def replace_best_model():
    """将当前模型替换为历史最佳模型"""
    try:
        if not os.path.exists('./data/models/current_model.pkl'):
            return jsonify({'success': False, 'message': '未找到当前模型文件'}), 400

        import pickle
        with open('data/models/current_model.pkl', 'rb') as f:
            current_model = pickle.load(f)
        scaler_X = joblib.load('data/models/scaler_X.pkl')
        scaler_y = joblib.load('data/models/scaler_y.pkl')
        with open('data/models/saved_weather_model.pkl', 'wb') as f:
            pickle.dump({'model': current_model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

        if os.path.exists('data/json/pending_model.json'):
            with open('data/json/pending_model.json', 'r', encoding='utf-8') as f:
                pending = json.load(f)
            current_avg = pending['current_avg_mae']
            with open('data/json/best_mae.json', 'w', encoding='utf-8') as f:
                json.dump({'best_mae': current_avg}, f)

        if os.path.exists('data/csv/current_mae.csv'):
            df = pd.read_csv('data/csv/current_mae.csv')
            df.to_csv('data/csv/mae.csv', index=False, encoding='utf-8-sig')

        if os.path.exists('data/json/pending_model.json'):
            os.remove('data/json/pending_model.json')

        # 同步复制测试集预测数据
        import shutil
        if os.path.exists('data/json/current_test_pred.json'):
            shutil.copy('data/json/current_test_pred.json', 'data/json/best_test_pred.json')

        message = '已成功替换为历史最佳模型。'
        if os.path.exists('data/json/predict_source.json'):
            with open('data/json/predict_source.json', 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            if source_data.get('source') == 'best':
                message = ('历史最佳模型已更新。'
                           '当前预测曲线仍为旧模型数据，请手动运行一次预测以查看新模型效果。')
        # -----------------------------------------------

        return jsonify({'success': True, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==================== AI 聊天 API ====================

# 测试集对比数据 API
@app.route('/api/test_comparison')
def get_test_comparison():
    """返回测试集真实值 + 当前模型预测 + 历史最佳模型预测"""
    result = {'dates': [], 'actual': {}, 'current': {}, 'best': {}}
    try:
        if os.path.exists('data/json/current_test_pred.json'):
            with open('data/json/current_test_pred.json', 'r', encoding='utf-8') as f:
                c = json.load(f)
            result['dates'] = c['dates']
            result['actual'] = c['actual']
            result['current'] = c['predicted']
        if os.path.exists('data/json/best_test_pred.json'):
            with open('data/json/best_test_pred.json', 'r', encoding='utf-8') as f:
                b = json.load(f)
            result['best'] = b['predicted']
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify(result)


@app.route('/api/model_metrics')
def get_model_metrics():
    """从 current_test_pred.json 计算 MSE, RMSE, MAE, R²"""
    import numpy as np
    path = 'data/json/current_test_pred.json'
    if not os.path.exists(path):
        return jsonify({'error': 'current_test_pred.json 不存在，请先训练模型'}), 404
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    metrics = {}
    metric_order = ['temperature_max', 'temperature_min', 'temperature_mean',
                    'humidity_mean', 'wind_speed_max', 'precipitation']
    for key in metric_order:
        if key not in data['actual'] or key not in data['predicted']:
            continue
        actual = np.array(data['actual'][key])
        pred = np.array(data['predicted'][key])
        mae = float(np.mean(np.abs(actual - pred)))
        mse = float(np.mean((actual - pred) ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
        metrics[key] = {'MAE': round(mae, 4), 'MSE': round(mse, 4),
                        'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

    return jsonify({'metrics': metrics, 'dates': data.get('dates', [])})


# ==================== AI 聊天 API ====================

# 存储多轮对话上下文（内存中，重启丢失）
chat_sessions = {}
MAX_SESSIONS = 100
SESSION_TTL = 1800  # 30 分钟无活动自动清理


def _cleanup_sessions():
    """清理过期和超量的会话"""
    now = datetime.now().timestamp()
    # 清理过期
    expired = [sid for sid, (_, _, ts) in chat_sessions.items() if now - ts > SESSION_TTL]
    for sid in expired:
        del chat_sessions[sid]
    # 清理超量（保留最新的）
    if len(chat_sessions) > MAX_SESSIONS:
        sorted_sessions = sorted(chat_sessions.items(), key=lambda x: x[1][2], reverse=True)
        for sid, _ in sorted_sessions[MAX_SESSIONS:]:
            del chat_sessions[sid]


@app.route('/chat')
def chat_page():
    """返回聊天页面"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """处理聊天消息，返回 AI 回复"""
    from agent.agent import chat

    data = request.get_json()
    user_msg = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    model_key = data.get('model_key', None)  # None = 使用默认模型

    if not user_msg:
        return jsonify({'reply': '请输入消息', 'tool_calls': []})

    # memory_list 快捷通道：检测到关键词直接返回知识库列表，跳过 LLM
    MEMORY_KEYWORDS = ["记忆库", "知识库列表", "memory list", "看记忆", "有哪些记忆", "memory_list", "列出知识库"]
    if any(kw in user_msg for kw in MEMORY_KEYWORDS):
        from agent.tools import memory_list
        r = memory_list()
        return jsonify({'reply': r['result'], 'tool_calls': [], 'tokens': 0})

    # 获取或创建会话上下文
    _cleanup_sessions()
    if session_id not in chat_sessions:
        chat_sessions[session_id] = (None, None, datetime.now().timestamp())

    # 切换模型时清上下文
    prev_model, prev_msgs, _ = chat_sessions[session_id]
    if model_key != prev_model:
        prev_msgs = None

    try:
        result = chat(user_msg, messages=prev_msgs, model_key=model_key)
        # 保存上下文供下一轮使用
        chat_sessions[session_id] = (model_key, result['messages'], datetime.now().timestamp())
        return jsonify({
            'reply': result['reply'],
            'reasoning': result.get('reasoning', ''),
            'tool_calls': result['tool_calls'],
            'tokens': result['tokens'],
            'model': model_key or 'deepseek',
        })
    except Exception as e:
        return jsonify({'reply': f'出错了: {str(e)}', 'tool_calls': []}), 500


if __name__ == '__main__':
    print("[INIT] 初始化应用数据...")
    initialize_model_evaluation_data()

    # 调试模式下显示所有路由
    if app.debug:
        print("Available routes:")
        for rule in app.url_map.iter_rules():
            print(f"{rule.endpoint}: {rule.rule}")

    app.run(debug=True, host='0.0.0.0', port=5000)