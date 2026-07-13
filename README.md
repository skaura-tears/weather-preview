# 南阳天气预测系统

基于 **LSTM 杂交架构** 的 7 天天气预报系统，集成 Web 可视化面板 + AI天气助手（可调用工具）。

## 架构概览

数据采集 → 清洗 → LSTM 训练 → 模型文件 → Flask Web 服务 → 前端可视化 + AI 助手

## 模型性能

| 指标 | MAE | MSE | RMSE | R² | 评价 |
|------|------|------|------|------|:-----|
| 最高温度 | 2.90 | 14.24 | 3.77 | 0.83 | 优秀 |
| 最低温度 | 2.13 | 7.27 | 2.70 | 0.91 | 优秀 |
| 平均温度 | 2.19 | 8.01 | 2.83 | 0.90 | 优秀 |
| 湿度 | 9.15 | 147.59 | 12.15 | 0.19 | 较弱 |
| 风速 | 4.47 | 35.95 | 6.00 | 0.00 | 较弱 |
| 降水量 | 5.76 | 77.07 | 8.78 | -0.14 | 无效 |

> 杂交架构：温度自回归解码 + 其他特征直接多步输出。训练数据：南阳市 2008-2023 年逐日观测。

## 快速开始

### 1. 安装依赖
```bash
# Windows（推荐 DirectML GPU 加速）
pip install -r requirements.txt
pip install torch-directml

# Linux / macOS（CUDA GPU）
pip install -r requirements.txt
```

### 2.获取数据 
```bash
python 获取数据.py 
```

### 3. 训练模型

```bash
python train_model.py
```

第一次训练完成后 `data/models/` 下会生成：
- `saved_weather_model.pkl` — 模型权重，特征标签缩放器

### 4. 启动 Web 服务

```bash
python app.py
```

浏览器打开 http://localhost:5000，四个页面：

| 页面 | 功能 |
|------|------|
| 模型评估 | 损失曲线、测试集指标、模型版本替换 |
| 天气预测 | 未来 7 天逐日预报 + 表格 |
| 历史数据 | 15 年数据可视化，多特征叠加对比 |
| AI 助手 | DeepSeek 驱动的天气咨询聊天机器人 |

## 项目结构

```
whether preview/
├── app.py                  # Flask Web 服务入口
├── train_model.py          # 模型训练（WeatherLSTM 类定义）
├── 获取数据.py              # Open-Meteo API 数据采集
├── spearman_corr.py        # Spearman 秩相关系数分析
├── requirements.txt        # Python 依赖
│
├── agent/                  # AI 聊天模块
│   ├── agent.py            # DeepSeek/Ollama 模型调用
│   ├── tools.py            # 工具函数（预测、查询、知识库）
│   ├── prompts.py          # 系统提示词
│   └── memory/             # 知识库 Markdown 文件
│
├── templates/              # 前端页面
│   ├── model_evaluation.html
│   ├── weather_prediction.html
│   ├── historical_analysis.html
│   └── chat.html
│
├── static/js/echarts.min.js # ECharts 本地副本
│
├── data/
│   ├── csv/                # 原始+清洗数据、预测结果
│   ├── json/               # 训练历史、模型对比
│   └── models/             # *.pkl 模型文件（gitignore）
│
└── .env.example            # API Key 模板
```

## 数据来源

- [Open-Meteo](https://open-meteo.com/) — 免费、无需 API Key
- 南阳市 (32.9907°N, 112.5283°E)，2008–至今逐日数据
- 15 个特征：温度、湿度、降水、风速、气压、短波辐射、露点等

## 技术栈

| 层 | 技术 |
|----|------|
| 模型 | PyTorch LSTM (encoder-decoder 杂交) |
| 训练 | skorch + RandomizedSearchCV |
| 后端 | Flask + pandas |
| 前端 | ECharts 5.4 + 原生 JS |
| AI | DeepSeek API（支持 Ollama 本地模型）|
| GPU | Windows: DirectML / Linux: CUDA |

## Known Issues

- 风速不可预测（单站缺气压梯度信息，Spearman 时滞相关性仅 0.30）
- 湿度 / 降水 R² 偏低（数据天花板，需多站点和更多的数据）
- 单 LSTM 模型难以预测突变点（门控机制倾向于平滑输出，对降水、风速等突发性特征捕捉能力不足）

