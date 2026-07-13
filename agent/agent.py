"""
Weather Project Agent — 主程序

基于 DeepSeek API + 工具调用的自主编程 Agent。
通过 Web 界面（app.py）提供聊天服务。
"""

import sys
import os
import json
import io
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Windows GBK 编码修复
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 加载 .env 文件
def _load_dotenv():
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():#将字符串按换行划分返回各行文本组成的列表
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")#使用限制分隔符=分隔元素，分为三维元祖
                if key.strip() not in os.environ:#os.environ代表的是环境变量
                    os.environ[key.strip()] = val.strip()

_load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-reasoner"

# 本地 Ollama 模型
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"

# 可用模型注册表
AVAILABLE_MODELS = {
    "deepseek": {"base_url": DEEPSEEK_BASE_URL, "model": DEEPSEEK_MODEL, "label": "DeepSeek API (云端)"},
    "ollama":   {"base_url": OLLAMA_BASE_URL, "model": OLLAMA_MODEL, "label": "Ollama 本地 (qwen2.5:7b)"},
}

# ============================================================
# 初始化 DeepSeek 客户端
# ============================================================
def get_client(model_key: str = None):
    """创建 OpenAI 兼容客户端，支持 DeepSeek API 和本地 Ollama。

    model_key: "deepseek" (默认) | "ollama"
    默认模型可通过环境变量 AGENT_MODEL 设置。
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("未安装 openai SDK，请运行: pip install openai")
        sys.exit(1)

    if model_key is None:
        model_key = os.environ.get("AGENT_MODEL", "deepseek")

    config = AVAILABLE_MODELS.get(model_key)
    if config is None:
        print(f"未知模型: {model_key}，可用: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    if model_key == "ollama":
        # Ollama 本地模型不需要 API key
        return OpenAI(api_key="ollama", base_url=config["base_url"])

    # DeepSeek 需要 API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY，请在项目根目录 .env 文件中设置\n"
            "格式: DEEPSEEK_API_KEY=sk-..."
        )

    return OpenAI(api_key=api_key, base_url=config["base_url"])


def get_model_name(model_key: str = None) -> str:
    """获取当前使用的模型名称。"""
    if model_key is None:
        model_key = os.environ.get("AGENT_MODEL", "deepseek")
    return AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["deepseek"])["model"]


# ============================================================
# 初始化知识库
# ============================================================
def init_memory():
    """创建项目初始知识库"""
    from pathlib import Path
    memory_dir = Path(__file__).resolve().parent / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    memories = {
        "01_project_overview.md": """# 项目概述

## 基本信息
- **项目名称**: 南阳天气预测 (Nanyang Weather Prediction)
- **位置**: `d:/pythonproject/whether preview/`
- **目标**: 使用 LSTM 深度学习模型预测南阳市未来 7 天的 6 项天气指标
- **数据源**: Open-Meteo Archive API
- **GPU**: NVIDIA RTX 3060 (通过 DirectML)

## 核心文件
| 文件 | 功能 |
|------|------|
| `获取数据.py` | 从 Open-Meteo API 获取南阳天气数据并预处理 |
| `train-model.py` | LSTM 模型训练，含超参数搜索和评估 |
| `feature_importance.py` | 排列重要性分析，识别有用/噪音特征 |
| `app.py` | Flask Web 应用，提供模型评估和预测可视化 |

## 运行方式
```bash
source .venv/Scripts/activate
python 获取数据.py          # 获取数据
python train-model.py       # 训练模型
python feature_importance.py # 特征重要性分析
python app.py               # 启动 Web 应用
```
""",

 "02_model_architecture.md": """# 模型架构

## WeatherLSTM
```
7天历史 → LSTM1(64) → Dropout → LSTM2(32) → Dropout → LSTM3(16)
  → 温度分支（自回归）：
      第1步：全零输入 → Decoder LSTM → head → 第1天预测
      第2-7步：上一步预测(断梯度) → Decoder LSTM → head → 逐日预测
  → 其他分支（直接多步）：末隐层 → Linear → 7×3
  → 拼接 → (batch, 7, 6)
```
## 关键参数
- **序列长度**: 7 天
- **预测步长**: 7 天
- **输入特征**: 14维（原始 7 + 工程特征 6）
- **输出**: 6 维（温度 3 + 湿度 1 + 风速 1 + 降水 1）
- **损失函数**: HuberLoss (delta=1.5)
- **优化器**: Adam (lr=1.5e-4)
""",

        "03_data_schema.md": """# 数据 Schema

## 特征列 (FEATURE_COLS — 14 维)
'temperature_max', 'temperature_min', 'temperature_mean',
    'precipitation', 'wind_speed_max', 'humidity_mean',
    'surface_pressure', 'shortwave_radiation',
    'vapour_pressure_deficit', 'dew_point', 'wind_gusts_max',
     'pressure_change_1d','doy_sin', 'doy_cos'

## 目标列 (TARGET_COLS — 6 维)
temperature_max, temperature_min, temperature_mean,
humidity_mean, wind_speed_max, precipitation

## 标准化
- X: StandardScaler
- y: precipitation用MinMaxScaler，其余StandardScaler
""",

        "04_decisions.md": """# 技术决策记录

## GPU 选择
使用 DirectML 而非 CUDA（Windows + RTX 3060 兼容性考虑）。
## 测试集
固定 2023 年全年作为测试集。
## 超参数搜索
RandomizedSearchCV (n_iter=6) + TimeSeriesSplit (3折)
""",
    }

    for filename, content in memories.items():
        filepath = memory_dir / filename
        if filepath.exists():
            print(f"  · skip {filename}")
        else:
            filepath.write_text(content, encoding="utf-8")
            print(f"  + create {filename}")

    print(f"\nmemory initialized: {memory_dir}")


# ============================================================
# 上下文管理（摘要压缩）
# ============================================================
MAX_CONTEXT_TOKENS = 32000   # 超过此阈值触发压缩
KEEP_RECENT = 8              # 压缩时保留最近 N 条消息


def _close_old_user_messages(messages: list, keep_last: int = 0) -> list:
    """将所有旧 user 消息截断为陈述式摘要，防止模型重答旧问题。

    keep_last: 保留最新 N 条用户消息不被标记（默认 0，全部标记）。
    在 chat() 中，新消息尚未追加，所有 user 都是旧的 → keep_last=0。
    在 run_agent() 中，新消息已追加，保留最后 1 条 → keep_last=1。
    """
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_last:
        return messages
    for idx in user_indices[: -keep_last] if keep_last > 0 else user_indices:
        msg = messages[idx]
        original = msg.get("content", "")
        if original.startswith("[历史]"):
            continue
        topic = original[:60].replace("\n", " ").rstrip("？?。.！!")
        if len(original) > 60:
            topic += "…"
        messages[idx] = {
            **msg,
            "content": f"[历史] {topic}"
        }
    return messages


def _estimate_tokens(messages: list) -> int:
    """粗略估算 token 数。中文 ~1.5 字符/token，英文/JSON ~4 字符/token。"""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        if isinstance(content, str):
            chinese = sum(1 for c in content if '一' <= c <= '鿿')#如果是汉字返回1得到汉字总数
            total += chinese / 1.5 + (len(content) - chinese) / 4
        for tc in msg.get("tool_calls", []) or []:
            total += len(json.dumps(tc, ensure_ascii=False, default=str)) / 4
    return int(total)


def _summarize_messages(client, model, msgs: list) -> str:
    """调用 LLM 将一段消息历史压缩为结构化摘要。"""
    prompt = (
        "Summarize the conversation above concisely, in Chinese. Keep: "
        "1) user requests & intents, "
        "2) key decisions, "
        "3) files created/modified and why, "
        "4) important findings / results, "
        "5) errors and resolutions, "
        "6) code patterns or conventions used, "
        "7) ALL tool calls made by the assistant (which tool, on what file/parameter, result). "
        "This is critical — if you omit a tool call, the assistant will forget it happened. "
        "Output compact summary only. Do NOT continue or re-execute any task."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=list(msgs) + [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def _compress_context(messages: list, client, model) -> list:
    """若消息总 token 超阈值，将旧消息压缩为摘要，保留系统提示 + 最近消息。"""
    # 先清理 tool 消息和 tool_calls，防止压缩边界切断 tool_calls/tool_results 配对
    if len(messages) <= KEEP_RECENT + 4:
        return messages

    estimated = _estimate_tokens(messages)
    if estimated < MAX_CONTEXT_TOKENS:
        return messages

    system_msg = messages[0] if messages[0]["role"] == "system" else None
    start = 1 if system_msg else 0
    recent = messages[-KEEP_RECENT:]
    old = messages[start:-KEEP_RECENT]

    if len(old) < 4:
        return messages

    print(f"\n[上下文压缩] {estimated:,} tokens → ", end="", flush=True)
    summary = _summarize_messages(client, model, old)

    new_msgs = []
    if system_msg:
        new_msgs.append(system_msg)
    new_msgs.append({"role": "user", "content": f"[对话历史摘要]\n{summary}\n[/摘要]"})
    new_msgs.extend(recent)

    print(f"{_estimate_tokens(new_msgs):,} tokens")
    return new_msgs


# ============================================================
# Agent 核心循环
# ============================================================

# ============================================================
# Web Chat 接口
# ============================================================
def chat(user_message: str, model: str = None, messages: list = None, model_key: str = None) -> dict:
    """供 Web 调用的聊天接口。返回 {reply, tool_calls, messages, tokens}"""
    from agent.tools import TOOL_DEFINITIONS_OPENAI, TOOL_HANDLERS
    try:
        from agent.prompts import build_system_prompt
    except ImportError:
        from agent.prompts_example import build_system_prompt

    if model is None:
        model = get_model_name(model_key)

    client = get_client(model_key)
    system_prompt = build_system_prompt(str(PROJECT_ROOT))

    if messages is None:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        # 每轮刷新 system prompt，避免旧 session 锁死过时版本
        messages[0] = {"role": "system", "content": system_prompt}

    # 追加前剥离旧轮的工具消息 — raw data 污染语义 + orphaned tool_calls 报错
    messages = [m for m in messages if m["role"] != "tool"]
    messages = [{k: v for k, v in m.items() if k != "tool_calls"} for m in messages]
    _close_old_user_messages(messages)
    messages.append({"role": "user", "content": user_message})
    messages = _compress_context(messages, client, model)

    tool_calls_made = []
    total_tokens = 0
    max_turns = 20

    for turn in range(1, max_turns + 1):
        messages = _compress_context(messages, client, model)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS_OPENAI,
            temperature=0.1,
            max_tokens=4096,
        )

        choice = response.choices[0]
        msg = choice.message
        if response.usage:
            total_tokens += response.usage.total_tokens

        if choice.finish_reason == "stop":
            # 必须把 assistant 回复追加到 messages，否则下轮模型只看到 user 消息
            # 看不到任何回答记录，会认为旧问题都没被处理而重复回答
            messages.append({"role": "assistant", "content": msg.content or ""})
            return {
                "reply": msg.content or "",
                "reasoning": getattr(msg, 'reasoning_content', '') or '',
                "tool_calls": tool_calls_made,
                "messages": messages,
                "tokens": total_tokens,
            }

        elif choice.finish_reason == "tool_calls":
            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                detail = tool_input.get('date_str', '') or tool_input.get('start', '') or tool_input.get('query', '') or ''
                handler = TOOL_HANDLERS.get(tool_name)
                if handler:
                    try:
                        result = handler(**tool_input)
                    except TypeError:
                        result = {"success": False, "result": "", "error": "参数错误"}
                else:
                    result = {"success": False, "result": "", "error": f"unknown: {tool_name}"}

                tool_calls_made.append({
                    "name": tool_name,
                    "detail": str(detail)[:60],
                    "success": result["success"],
                })

                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        elif choice.finish_reason == "length":
            messages.append({"role": "assistant", "content": msg.content})
            messages.append({"role": "user", "content": "continue."})

        else:
            break

    return {
        "reply": "抱歉，处理超时了。请换个简单点的问题试试。",
        "reasoning": "",
        "tool_calls": tool_calls_made,
        "messages": messages,
        "tokens": total_tokens,
    }


# ============================================================
# CLI 入口
# ============================================================

