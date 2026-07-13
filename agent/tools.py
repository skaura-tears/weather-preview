"""
Agent 工具 — 天气查询、预测、知识库。
每个工具返回 {"success": bool, "result": str, "error": str | None}
"""

import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# ====== 模型类定义（torch.load 需要） ======
import torch.nn as nn

# 从 train_model 导入，杜绝类定义重复
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from train_model import WeatherLSTM

# ====== 路径 ======
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = Path(__file__).resolve().parent / "memory"
DATA_CSV = PROJECT_ROOT / "data" / "csv" / "清洗后数据.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "saved_weather_model.pkl"
SCALER_X_PATH = PROJECT_ROOT / "data" / "models" / "scaler_X.pkl"
SCALER_Y_PATH = PROJECT_ROOT / "data" / "models" / "scaler_y.pkl"
PRED_CSV = PROJECT_ROOT / "data" / "csv" / "prediction_results.csv"
BEST_MAE_PATH = PROJECT_ROOT / "data" / "json" / "best_mae.json"

FEATURE_COLS = [
    'temperature_max', 'temperature_min', 'temperature_mean',
    'precipitation', 'wind_speed_max', 'humidity_mean',
    'surface_pressure', 'shortwave_radiation',
    'vapour_pressure_deficit', 'dew_point', 'wind_gusts_max',
     'pressure_change_1d','doy_sin', 'doy_cos',
]
TARGET_COLS = ['temperature_max', 'temperature_min', 'temperature_mean',
               'humidity_mean', 'wind_speed_max', 'precipitation']

# ====== 缓存 ======
_df_cache = None
_model_cache = None
_scaler_X_cache = None
_scaler_Y_cache = None
_device_cache = None
_module_registered = False


def _load_data():
    """懒加载历史天气数据"""
    global _df_cache
    if _df_cache is None:
        df = pd.read_csv(DATA_CSV)
        df['date'] = pd.to_datetime(df['date'])
        _df_cache = df.sort_values('date').reset_index(drop=True)
    return _df_cache


# ============================================================
# ① 读取项目文件
# ============================================================
def read_file(path: str, offset: int = 0, limit: int = 100, **kwargs) -> dict:
    """
    读取项目文件内容。用于验证代码中的事实（特征维度、参数值等）。

    Args:
        path: 文件相对于项目根目录的路径，如 'train-model.py'、'data/json/best_mae.json'
        offset: 起始行号
        limit: 最大行数
    """
    try:
        filepath = (PROJECT_ROOT / path).resolve()
        # 禁止读取敏感文件
        if filepath.name in ('.env', '.gitconfig') or filepath.suffix in ('.pem', '.key'):
            return {"success": False, "result": "", "error": "此文件包含敏感信息，不允许读取"}
        if not str(filepath).startswith(str(PROJECT_ROOT.resolve())):
            return {"success": False, "result": "", "error": "禁止访问项目目录外的路径"}
        if not filepath.exists():
            return {"success": False, "result": "", "error": f"文件不存在: {path}"}
        if filepath.is_dir():
            files = sorted(f.relative_to(filepath).as_posix() for f in filepath.rglob("*") if f.is_file())
            return {"success": True, "result": "\n".join(files[:50]), "error": None}

        content = filepath.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        total = len(lines)
        actual_limit = limit if limit and limit > 0 else 100
        actual_limit = min(actual_limit, 500)
        sliced = lines[offset:offset + actual_limit]
        numbered = [f"{i + offset + 1:>4}\t{line}" for i, line in enumerate(sliced)]
        return {
            "success": True,
            "result": f"【{path}】共 {total} 行，显示 {len(sliced)} 行\n" + "\n".join(numbered),
            "error": None,
        }
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


# ============================================================
# ② 获取当前日期
# ============================================================
def get_date(**kwargs) -> dict:
    """返回今天的日期"""
    now = datetime.now()
    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    return {
        "success": True,
        "result": f"今天是 {now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]}",
        "error": None,
    }


# ============================================================
# ② 查询历史天气（某一天）
# ============================================================
def get_weather(date_str: str = "", **kwargs) -> dict:
    """
    查询南阳市指定日期的天气数据。

    Args:
        date_str: 日期 YYYY-MM-DD，如 '2026-07-05'。留空则返回今天。
    """
    try:
        df = _load_data()

        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        target = pd.to_datetime(date_str)
        row = df[df['date'].dt.date == target.date()]

        if row.empty:
            closest = df.iloc[(df['date'] - target).abs().argsort()[:1]]
            closest_date = closest['date'].iloc[0].strftime('%Y-%m-%d')
            return {
                "success": True,
                "result": f"⚠ {date_str} 无数据，最近可用日期: {closest_date}\n\n{_format_weather_row(closest)}",
                "error": None,
            }

        return {
            "success": True,
            "result": _format_weather_row(row),
            "error": None,
        }

    except Exception as e:
        return {"success": False, "result": "", "error": f"查询失败: {e}"}


# ============================================================
# ③ 日期范围天气概览
# ============================================================
def get_weather_range(start: str, end: str = "", **kwargs) -> dict:
    """
    查询一段时间的天气统计 + 逐日详情。

    Args:
        start: 起始日期 YYYY-MM-DD
        end: 结束日期 YYYY-MM-DD，留空则为 start 后 7 天
    """
    try:
        df = _load_data()
        start_dt = pd.to_datetime(start)
        if not end:
            end_dt = start_dt + timedelta(days=6)
        else:
            end_dt = pd.to_datetime(end)

        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        period = df[mask]

        if period.empty:
            return {"success": True, "result": f"{start} 至 {end} 无数据", "error": None}

        s = period
        lines = [
            f"📅 {start} 至 {end} 天气概览 ({len(period)}天)",
            "",
            f"  最高温  {s['temperature_max'].max():.1f}℃ / 最低 {s['temperature_min'].min():.1f}℃",
            f"  均温    {s['temperature_mean'].mean():.1f}℃",
            f"  降水    累计 {s['precipitation'].sum():.1f}mm，{s[s['precipitation'] > 0].shape[0]} 天有雨",
            f"  湿度    {s['humidity_mean'].mean():.0f}% (范围 {s['humidity_mean'].min():.0f}-{s['humidity_mean'].max():.0f})",
            f"  风速    平均 {s['wind_speed_max'].mean():.1f} km/h (最大 {s['wind_speed_max'].max():.1f})",
            "",
            "逐日:",
        ]

        for _, r in period.iterrows():
            date_s = r['date'].strftime('%m/%d')
            rain = f"💧{r['precipitation']:.1f}mm" if r['precipitation'] > 0 else "☀"
            lines.append(
                f"  {date_s} {rain:10s}  {r['temperature_max']:5.1f}/{r['temperature_min']:5.1f}℃  "
                f"湿{r['humidity_mean']:3.0f}%  风{r['wind_speed_max']:5.1f}"
            )

        return {"success": True, "result": "\n".join(lines), "error": None}

    except Exception as e:
        return {"success": False, "result": "", "error": f"查询失败: {e}"}


# ============================================================
# ④ 天气预测（LSTM 模型）
# ============================================================
def predict_weather(days: int = 7, **kwargs) -> dict:
    """
    使用 LSTM 深度学习模型预测南阳市未来天气。

    Args:
        days: 预测天数 1-7，默认 7
    """
    if days < 1 or days > 7:
        days = min(max(days, 1), 7)

    try:
        global _model_cache, _scaler_X_cache, _scaler_Y_cache, _device_cache, _module_registered

        if not _module_registered:
            sys.modules['__main__'].WeatherLSTM = WeatherLSTM
            _module_registered = True

        if _device_cache is None:
            try:
                import torch_directml
                _device_cache = torch_directml.device()
            except ImportError:
                _device_cache = torch.device('cpu')
        device = _device_cache

        # 加载 skorch 模型 + 缩放器（pickle 格式）
        if _model_cache is None:
            with open(MODEL_PATH, 'rb') as f:
                saved = pickle.load(f)
            if isinstance(saved, dict):
                _model_cache = saved['model']
                _scaler_X_cache = saved['scaler_X']
                _scaler_Y_cache = saved['scaler_y']
            else:
                # 兼容旧格式（仅保存了模型，未打包缩放器）
                _model_cache = saved
                _scaler_X_cache = joblib.load(SCALER_X_PATH)
                _scaler_Y_cache = joblib.load(SCALER_Y_PATH)
            _model_cache.module_.to(device)
            _model_cache.module_.eval()
        model = _model_cache

        # 历史数据 + 特征工程
        df = _load_data().copy()
        df['doy_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)
        df['pressure_change_1d'] = df['surface_pressure'].diff(1).fillna(0)
        df = df.dropna()

        scaler_X = _scaler_X_cache
        scaler_y = _scaler_Y_cache

        # 最后 90 天输入
        seq_len = 30
        recent = df[FEATURE_COLS].iloc[-seq_len:]
        X = scaler_X.transform(recent)
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)

        # 推理 - skorch 模型用 module_ 访问底层 PyTorch 模块
        with torch.no_grad():
            y_scaled = model.module_(X_tensor).cpu().numpy()[0]

        # 反标准化
        y_pred = np.zeros((7, 6))
        for i, col in enumerate(TARGET_COLS):
            y_pred[:, i] = scaler_y[col].inverse_transform(y_scaled[:, i].reshape(-1, 1)).flatten()

        # 格式化
        # 预测起始日 = 数据截止日 + 1 天，不是硬编码从 "明天" 开始
        last_data_date = df['date'].iloc[-1]
        start_date = last_data_date + timedelta(days=1)
        lines = [f"🔮 南阳未来 {days} 天天气预测 (LSTM)", ""]
        lines.append(f"  {'日期':<8} {'最高℃':>6} {'最低℃':>6} {'均温℃':>6} {'湿度%':>6} {'风速':>7} {'降水mm':>7}")
        lines.append("  " + "-" * 52)

        for d in range(days):
            date = start_date + timedelta(days=d)
            date_s = date.strftime('%m/%d')
            tmax, tmin, tmean = y_pred[d, 0], y_pred[d, 1], y_pred[d, 2]
            hum = max(0, min(100, y_pred[d, 3]))
            wind = max(0, y_pred[d, 4])
            precip = max(0, y_pred[d, 5])

            icon = "🌧" if precip > 5 else ("💧" if precip > 0.5 else "☀")
            lines.append(
                f"  {date_s:<8} {tmax:>6.1f} {tmin:>6.1f} {tmean:>6.1f} "
                f"{hum:>6.0f} {wind:>6.1f} {icon} {precip:>5.1f}"
            )

        # MAE
        try:
            with open(PROJECT_ROOT / "data" / "json" / "pending_model.json", 'r') as f:
                pm = json.load(f)
            per = pm['comparison']['best']
            lines.append("")
            lines.append(
                f" 预测精度参考：气温 ±{per['temperature_max']:.1f}/"
                f"{per['temperature_min']:.1f}/{per['temperature_mean']:.1f}℃ | "
                f"湿度 ±{per['humidity_mean']:.1f}% | "
                f"风速 ±{per['wind_speed_max']:.1f} | "
                f"降水 ±{per['precipitation']:.1f}mm"
            )
        except:
            lines.append("")
            lines.append("  ℹ 预测精度：暂无数据")

        return {"success": True, "result": "\n".join(lines), "error": None}

    except FileNotFoundError as e:
        return {"success": False, "result": "", "error": f"模型文件缺失: {e}"}
    except Exception as e:
        return {"success": False, "result": "", "error": f"预测失败: {e}"}


# ============================================================
# ⑤ 模型信息
# ============================================================
def get_model_info(**kwargs) -> dict:
    """返回当前预测模型的基本信息"""
    try:
        df = _load_data()
        info = [
            "🤖 天气预测模型",
            "",
            f"  架构: LSTM(64→32→16split) × 3路独立输出头",
            f"  输入: 15 特征 × 90 天序列",
            f"  输出: 6 指标 × 7 天预测",
            f"  数据: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')} ({len(df)}条)",
            f"  位置: 南阳市 (32.99°N, 112.53°E)",
        ]

        try:
            with open(PROJECT_ROOT / "data" / "json" / "pending_model.json", 'r') as f:
                pm = json.load(f)
            info.append(
                f"  最佳 MAE: {pm['best_avg_mae']}（气温±{pm['comparison']['best']['temperature_mean']:.1f}℃ | 降水±{pm['comparison']['best']['precipitation']:.1f}mm）")
        except:
            pass

        return {"success": True, "result": "\n".join(info), "error": None}

    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


# ====== 辅助 ======

def _format_weather_row(df_row) -> str:
    r = df_row.iloc[0] if len(df_row) > 0 else df_row
    date_s = r['date'].strftime('%Y年%m月%d日')
    return "\n".join([
        f"📍 南阳 {date_s} 天气",
        "",
        f"  🌡 温度  {r['temperature_max']:.1f}℃ / {r['temperature_min']:.1f}℃  (均{r['temperature_mean']:.1f}℃)",
        f"  💧 降水  {r['precipitation']:.1f} mm",
        f"  💨 风速  最大 {r['wind_speed_max']:.1f} km/h",
        f"  🌫 湿度  {r['humidity_mean']:.0f}%",
        f"  📊 气压  {r['surface_pressure']:.1f} hPa",
    ])


# ============================================================
# ⑥ 执行项目脚本
# ============================================================
_ALLOWED_SCRIPTS = {
    "获取数据.py": "从 Open-Meteo API 获取南阳天气数据",
    "train-model.py": "训练 LSTM 模型",
    "spearman_corr.py": "spearman秩相关系数分析"
}

import subprocess


def run_script(script: str, args: str = "", timeout: int = 300, **kwargs) -> dict:
    """
    执行项目中的 Python 脚本（仅限白名单内的脚本）。

    Args:
        script: 脚本文件名，如 '获取数据.py'、'train-model.py'
        args: 命令行参数，如 '--force'；获取数据时留空即可
        timeout: 最大等待秒数，默认 300
    """
    if script not in _ALLOWED_SCRIPTS:
        return {
            "success": False,
            "result": "",
            "error": f"不允许执行的脚本: {script}。可执行列表: {', '.join(_ALLOWED_SCRIPTS)}",
        }
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        return {"success": False, "result": "", "error": f"脚本不存在: {script}"}

    try:
        # 优先使用 venv Python，否则依赖包找不到
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        python_exe = str(venv_python) if venv_python.exists() else sys.executable
        cmd = [python_exe, str(script_path)]
        if args:
            cmd.extend(args.split())
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        stdout = (result.stdout or "").strip() or "（无输出）"
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            detail = stderr or stdout
            return {
                "success": False,
                "result": f"脚本执行失败 (exit code {result.returncode})\n\n{detail[:2000]}\n\n（不要再重试此脚本，直接告诉用户失败原因）",
                "error": None,
            }
        return {
            "success": True,
            "result": f"✅ {script} 执行完成\n\n{stdout[:3000]}",
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "result": "", "error": f"脚本执行超时 ({timeout}秒)"}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


# ============================================================
# ⑦ 知识库
# ============================================================
def memory_search(query: str, **kwargs) -> dict:
    try:
        if not MEMORY_DIR.exists():
            return {"success": True, "result": "知识库为空。", "error": None}
        query_lower = query.lower()
        query_words = query_lower.split()
        matches = []
        for mf in sorted(MEMORY_DIR.glob("*.md")):
            content = mf.read_text(encoding="utf-8")
            first_line = content.strip().split("\n")[0]
            score = sum(first_line.lower().count(w) * 3 + content.lower().count(w) for w in query_words)
            if score > 0:
                matches.append((score, mf, content))
        if not matches:
            return {"success": True, "result": f"未找到与 '{query}' 相关的记忆。", "error": None}
        matches.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, mf, content in matches[:3]:
            title = content.strip().split("\n")[0].lstrip("# ").strip()
            results.append(f"### {title}\n{content[:1200]}")
        return {"success": True, "result": f"找到 {len(matches)} 条:\n\n" + "\n\n---\n\n".join(results), "error": None}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def memory_save(title: str, content: str, **kwargs) -> dict:
    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)[:40]
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        filepath = MEMORY_DIR / f"{ts}_{safe}.md"
        full = f"# {title}\n\n> 记录: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}"
        filepath.write_text(full, encoding="utf-8")
        return {"success": True, "result": f"已保存: {filepath.name}", "error": None}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def memory_update(title: str, content: str, **kwargs) -> dict:
    """按标题更新知识库条目。找到标题匹配的文件后覆盖其内容。"""
    try:
        if not MEMORY_DIR.exists():
            return {"success": False, "result": "", "error": "知识库为空，没有可更新的条目"}
        for mf in MEMORY_DIR.glob("*.md"):
            text = mf.read_text(encoding="utf-8")
            first_line = text.strip().split("\n")[0].lstrip("# ").strip()
            if first_line == title:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_text = f"# {title}\n\n> 更新: {ts}\n\n{content}"
                mf.write_text(new_text, encoding="utf-8")
                return {"success": True, "result": f"已更新: {mf.name}", "error": None}
        return {"success": False, "result": "", "error": f"未找到标题为 '{title}' 的知识库条目。试试 memory_list 查看现有条目。"}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def memory_list(**kwargs) -> dict:
    try:
        if not MEMORY_DIR.exists() or not list(MEMORY_DIR.glob("*.md")):
            return {"success": True, "result": "知识库为空。", "error": None}
        items = []
        for mf in sorted(MEMORY_DIR.glob("*.md")):
            first = mf.read_text(encoding="utf-8").strip().split("\n")[0].lstrip("# ").strip()
            items.append(f"- {mf.stem}: {first}")
        return {"success": True, "result": "知识库:\n" + "\n".join(items), "error": None}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


# ============================================================
# 注册表
# ============================================================
TOOL_HANDLERS = {
    "read_file": read_file,
    "get_date": get_date,
    "get_weather": get_weather,
    "get_weather_range": get_weather_range,
    "predict_weather": predict_weather,
    "get_model_info": get_model_info,
    "run_script": run_script,
    "memory_search": memory_search,
    "memory_save": memory_save,
    "memory_update": memory_update,
    "memory_list": memory_list,
}

TOOL_DEFINITIONS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取项目文件内容。用于验证代码中的事实：特征维度、参数值、配置等。不用于修改代码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径，如 'train-model.py'、'data/json/best_mae.json'"},
                    "offset": {"type": "integer", "description": "起始行号"},
                    "limit": {"type": "integer", "description": "最大行数，默认 100"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "获取今天的日期。当用户问'今天'、'现在'等时间相关问题时首先调用此工具。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询南阳市某一天的历史天气数据，返回温度、降水、风速、湿度、气压。用于查'昨天天气'、'某天天气'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {"type": "string", "description": "日期，YYYY-MM-DD 格式，如 '2026-07-05'。留空按今天处理。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_range",
            "description": "查询一段日期范围的天气统计概览和逐日详情。适合'这周天气怎么样'、'最近一周天气'类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "起始日期 YYYY-MM-DD"},
                    "end": {"type": "string", "description": "结束日期 YYYY-MM-DD，留空则为 start 后 7 天"},
                },
                "required": ["start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_weather",
            "description": "使用 LSTM 深度学习模型预测南阳市未来 1-7 天天气。返回预测的温度、湿度、风速、降水及模型精度。用于'明天天气'、'未来一周天气'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "预测天数，1-7，默认 7"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "获取当前天气预测模型的基本信息：架构、数据范围、精度(MAE)。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": "执行项目中的Python脚本。仅限白名单脚本: 获取数据.py（从Open-Meteo获取天气数据）、train-model.py（训练LSTM模型）。用于更新数据或重新训练模型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "脚本文件名，如 '获取数据.py'、'train-model.py'"},
                    "args": {"type": "string", "description": "命令行参数，获取数据时留空即可"},
                    "timeout": {"type": "integer", "description": "最大等待秒数，默认300"},
                },
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "搜索项目知识库。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "按标题更新知识库条目内容。找到匹配标题的文件后覆盖写入新内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "要更新的条目标题（精确匹配）"},
                    "content": {"type": "string", "description": "新的Markdown内容"},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "保存信息到知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "标题"},
                    "content": {"type": "string", "description": "内容(Markdown)"},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "列出知识库所有条目。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
