# 海上风电多算法智能体系统

基于机器学习与强化学习的海上风电场功率预测与智能调度系统。集成 HistGradientBoosting / LightGBM 预测模型与 Q-Learning / DQN / 遗传算法 / 启发式搜索等多种调度策略，提供 Vue 3 可视化驾驶舱。

## 项目背景

针对福建海域 5 个风电场，基于 2022–2023 年 15 分钟分辨率气象与功率数据，构建「预测 + 调度」一体化智能体系统：

- **功率预测**：融合 38+ 维时序特征（风向分解、湍流强度、风功率密度、滞后功率等），对比 HistGBR 与 LightGBM 模型精度
- **风险评分**：6 维加权风险评估（风速突变、降水、高风速、密度异常、云量、历史误差）
- **调度决策**：4 种动作（积极并网 / 平衡调度 / 保守预留 / 风险巡检），通过 Q-Learning、DQN、遗传算法、启发式搜索对比寻优

## 系统架构

```
┌─────────────────────────────────────────────────┐
│                  Vue 3 Frontend                  │
│   Overview │ Prediction │ Sites │ Experiments    │
│                     │ Agent                      │
└──────────────────────┬──────────────────────────┘
                       │ REST API (/api/*)
┌──────────────────────▼──────────────────────────┐
│               Flask API Server                   │
│                  (app.py)                        │
├─────────────────────────────────────────────────┤
│         OffshoreWindAgentSystem                  │
│            (wind_agent/system.py)                │
├────────────┬──────────────┬─────────────────────┤
│ forecasting│  rl_control  │   ollama_client     │
│ (HistGBR  │ (Q-Learning  │   (DeepSeek/Llama   │
│  LightGBM) │  DQN/GA/HS)  │    问答智能体)       │
├────────────┴──────────────┴─────────────────────┤
│              data_utils (特征工程)                │
└─────────────────────────────────────────────────┘
```

## 项目结构

```
offshore_wind_agent/
├── app.py                    # Flask API 入口 (端口 5000)
├── train_system.py           # 训练入口脚本
├── eval_model.py             # 模型精度评估实验 (v1)
├── eval_model_v2.py          # 模型精度评估实验 (v2, 增加滞后特征 + XGBoost)
├── requirements.txt          # Python 依赖
│
├── wind_agent/               # 核心 Python 包
│   ├── __init__.py
│   ├── system.py             # 系统编排 (OffshoreWindAgentSystem)
│   ├── data_utils.py         # 数据加载 / 清洗 / 特征工程
│   ├── forecasting.py        # 功率预测模型 (HistGBR / LightGBM)
│   ├── rl_control.py         # RL 调度 (Q-Learning / DQN / GA / HS)
│   └── ollama_client.py      # LLM 智能问答 (Ollama)
│
├── frontend/                 # Vue 3 + Vite 前端
│   ├── src/
│   │   ├── views/
│   │   │   ├── OverviewPage.vue      # 总览大屏
│   │   │   ├── PredictionPage.vue    # 预测分析
│   │   │   ├── SitesPage.vue         # 站点驾驶舱
│   │   │   ├── ExperimentsPage.vue   # 算法实验
│   │   │   └── AgentPage.vue         # 智能体中心
│   │   ├── components/               # SVG 图表 / 地图组件
│   │   ├── services/agentApi.js      # API 客户端
│   │   └── router.js                 # Vue Router
│   ├── vite.config.js
│   └── package.json
│
├── data/
│   ├── original/              # 原始数据 (中文 CSV)
│   └── raw/                   # 预处理后 CSV
│
└── artifacts/
    ├── best_run.txt           # 指向最优训练运行
    └── runs/                  # 训练产物 (system_bundle.joblib)
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.13, Flask 3.x |
| 预测模型 | scikit-learn HistGradientBoostingRegressor, LightGBM 4.x |
| 强化学习 | PyTorch 2.x (DQN), 表格型 Q-Learning |
| 特征工程 | pandas 2.x, numpy |
| 前端 | Vue 3 (Composition API), Vue Router 4, Vite 5 |
| 可视化 | 内联 SVG (无第三方图表库) |
| LLM 集成 | Ollama (DeepSeek-R1 / Llama3) |
| 序列化 | joblib |
| 训练监控 | TensorBoard |

## 快速开始

### 环境要求

- Python ≥ 3.13
- Node.js ≥ 18
- (可选) NVIDIA GPU + CUDA 12.1/11.8 用于 DQN 训练加速
- (可选) [Ollama](https://ollama.com/) 用于智能问答

### 1. 安装 Python 依赖

```bash
cd offshore_wind_agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt

# 如使用 GPU 训练 DQN, 请按 CUDA 版本安装 PyTorch:
# 例如 CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. 训练模型

首次使用需先训练（数据预处理 + 预测模型 + RL 策略），产物保存至 `artifacts/runs/`：

```bash
python train_system.py
```

训练完成后自动更新 `artifacts/best_run.txt` 指向最优结果。

### 3. 启动后端服务

```bash
python app.py
```

Flask API 运行于 `http://127.0.0.1:5000`，加载最优训练产物。

### 4. 启动前端 (开发模式)

```bash
cd frontend
npm install
npm run dev
```

Vite 开发服务器运行于 `http://127.0.0.1:5173`，自动代理 `/api` 到 Flask 后端。

### 5. (可选) 启动 Ollama 问答智能体

```bash
ollama serve                    # 启动 Ollama 服务
ollama pull deepseek-r1:7b      # 拉取模型 (或用 llama3)
```

前端「智能体中心」页面可进行自然语言问答。

## API 端点

### 仪表盘与数据

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/dashboard` | 总览大屏全量数据 |
| GET | `/api/site/<site_id>` | 单站点详情与调度表 |
| GET | `/api/comparison` | 模型/算法对比数据 |

### 智能问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ask` | 自然语言问答 (LLM / 规则回退) |
| POST | `/api/agent/refresh` | 重新检测 Ollama 连接 |

### 上传与预测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload/predict` | 上传 CSV 获取预测结果 |
| POST | `/api/upload/predict/download` | 上传 CSV 下载预测 CSV |
| POST | `/api/upload/save` | 持久化预测结果 |
| GET | `/api/uploads` | 列出历史上传记录 |
| GET | `/api/uploads/<id>` | 上传记录详情 |
| DELETE | `/api/uploads/<id>` | 删除上传记录 |
| GET | `/api/download/template` | 下载 CSV 模板 |

### 导出

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/export/forecast.csv` | 导出验证集预测对比 CSV |
| GET | `/api/export/rl_report.json` | 导出 RL 报告 JSON |

### 调试

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/debug/info` | 系统诊断信息 |

## 调度策略

| 动作 | 系数 | 说明 |
|------|------|------|
| 积极并网 | 1.04 | 高可信度下最大化出力 |
| 平衡调度 | 0.96 | 默认策略，平衡收益与风险 |
| 保守预留 | 0.86 | 中高风险下预留备用容量 |
| 风险巡检 | 0.78 | 高风险窗口主动降出力巡检 |

## 风险评分维度

| 维度 | 权重 | 触发条件 |
|------|------|----------|
| 风速突变 | 28% | 风速短期大幅波动 |
| 降水量 | 18% | 强降水影响风机运行 |
| 高风速 | 14% | 接近/超过切出风速 |
| 空气密度异常 | 12% | 偏离标准工况密度 |
| 云量 | 8% | 强对流天气预报 |
| 历史误差 | 12% | 该站点该时段历史预测不准 |
| 高出力压力 | 8% | 预测值接近装机容量 |

## 许可证

MIT License
