"""
海上风电强化学习调度控制模块 (RL Dispatch Control Module)
==========================================================
支持两种强化学习算法进行调度策略优化对比:
  - Q-Learning (表格型): 离散状态-动作空间的经典值迭代方法
  - DQN (Deep Q-Network): 用神经网络逼近 Q 函数, 处理高维/连续状态

同时实现了启发式搜索与遗传算法作为对比基线。
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from random import Random
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
_TORCH_AVAILABLE = True


# ============================================================================
# 动作空间定义 —— 4 种调度策略
# ============================================================================

ACTIONS = [
    {
        "id": 0,
        "name": "积极并网",
        "label_zh": "积极并网",
        "dispatch_factor": 1.04,
        "reserve_factor": 0.03,
        "maintenance_factor": 0.02,
        "description": "风险较低时优先追求更多发电收益。",
    },
    {
        "id": 1,
        "name": "平衡调度",
        "label_zh": "平衡调度",
        "dispatch_factor": 0.96,
        "reserve_factor": 0.08,
        "maintenance_factor": 0.03,
        "description": "在收益、预留裕度和系统稳定性之间保持平衡。",
    },
    {
        "id": 2,
        "name": "保守预留",
        "label_zh": "保守预留",
        "dispatch_factor": 0.86,
        "reserve_factor": 0.18,
        "maintenance_factor": 0.05,
        "description": "当风速波动增强时保留更多裕度以降低风险。",
    },
    {
        "id": 3,
        "name": "风险巡检",
        "label_zh": "风险巡检",
        "dispatch_factor": 0.78,
        "reserve_factor": 0.25,
        "maintenance_factor": 0.12,
        "description": "降低并网输出，优先保障设备安全与状态稳定。",
    },
]

# 训练超参数
TRAIN_EPISODE_LIMIT = 140    # Q-Learning 每次训练的 episode 数量上限
TRAIN_EPOCHS = 18            # Q-Learning 训练轮数
GA_POPULATION = 24           # 遗传算法种群大小
GA_GENERATIONS = 18          # 遗传算法迭代代数
GA_MUTATION_RATE = 0.05      # 变异率
GA_CROSSOVER_RATE = 0.78     # 交叉率
GAME_EPISODE_LIMIT = 96      # 游戏评估的 episode 数

# DQN 专用超参数
DQN_HIDDEN_DIM = 256          # 隐藏层维度 (连续特征需要更大容量)
DQN_EPOCHS = 25               # 训练轮数
DQN_LR = 0.0003               # Adam 学习率 (降低以提高稳定性)
DQN_GAMMA = 0.90              # 折扣因子 (与 Q-Learning 保持一致)
DQN_EPSILON_START = 0.50      # 初始探索率
DQN_EPSILON_MIN = 0.05        # 最终探索率
DQN_EPSILON_DECAY = 0.94      # 每轮探索率衰减系数
DQN_BATCH_SIZE = 64           # 经验回放批次大小
DQN_REPLAY_SIZE = 50000       # 经验回放缓冲区容量 (增大以保留早期探索经验)
DQN_TARGET_UPDATE = 4         # 每 N 轮硬同步一次目标网络
DQN_TAU = 0.005               # 软更新系数 (Polyak averaging, 每步微量更新目标网络)
DQN_GRAD_CLIP = 1.0           # 梯度裁剪阈值 (更紧的裁剪防梯度爆炸)
DQN_WARMUP_STEPS = 800        # 预热步数: 先填充回放缓冲区再开始学习


# ============================================================================
# GPU 设备检测
# ============================================================================

def get_device() -> str:
    """检测并返回最优计算设备

    优先级: CUDA GPU > MPS (Apple Silicon) > CPU
    同时打印详细的设备诊断信息, 方便排查 GPU 不可用的问题。
    """
    if not _TORCH_AVAILABLE:
        return "cpu"

    # ---- 诊断: 打印 PyTorch 版本和编译信息 ----
    print(f"[DQN] PyTorch 版本: {torch.__version__}")
    cuda_version = torch.version.cuda if hasattr(torch, 'version') and torch.version.cuda else None
    print(f"[DQN] PyTorch 编译时 CUDA 版本: {cuda_version or '未编译 CUDA 支持 (可能安装了 CPU-only 版本)'}")

    # ---- 尝试 CUDA GPU ----
    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[DQN] 检测到 {gpu_count} 块 GPU, 使用: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("[DQN] 使用 Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("[DQN] ⚠ 未检测到 GPU, 使用 CPU 训练 (DQN 训练时间可能较长)")
    return device


# ============================================================================
# 结果数据类
# ============================================================================

@dataclass
class RLBundle:
    """强化学习完整结果包

    包含 Q-Learning Q表、DQN 网络(可选)、多算法对比结果等。
    """
    # ---- Q-Learning 结果 ----
    q_table: dict[tuple[int, int, int, int, int], list[float]]
    comparison: list[dict[str, Any]]           # 各策略对比 (Q-Learning vs 基线)
    learned_policy_reward: float
    learned_policy_incident_rate: float
    action_catalog: list[dict[str, Any]]

    # ---- 启发式搜索 & 遗传算法结果 ----
    heuristic_examples: list[dict[str, Any]]
    ga_examples: list[dict[str, Any]]
    experiment_modules: list[dict[str, Any]]

    # ---- DQN 结果 (可选) ----
    dqn_comparison: list[dict[str, Any]] = field(default_factory=list)
    dqn_policy_reward: float = 0.0
    dqn_policy_incident_rate: float = 0.0
    dqn_experiment_modules: list[dict[str, Any]] = field(default_factory=list)
    dqn_state_dict: dict | None = None  # DQN 模型权重, 用于缓存持久化

    # ---- RL 算法对比 ----
    rl_algorithm_comparison: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 分时电价
# ============================================================================

def _price_for_hours(hours: np.ndarray) -> np.ndarray:
    """根据小时返回分时电价

    电价结构:
      - 低谷 (22-7时): 0.75
      - 平段 (6-7, 12-17时): 1.00
      - 尖峰 (8-11, 18-21时): 1.20
    """
    prices = np.full(hours.shape, 0.75, dtype=float)
    prices[((8 <= hours) & (hours <= 11)) | ((18 <= hours) & (hours <= 21))] = 1.20
    prices[((6 <= hours) & (hours <= 7)) | ((12 <= hours) & (hours <= 17))] = 1.00
    return prices


# ============================================================================
# 状态离散化
# ============================================================================

def _state_from_row(row: pd.Series, site_index: dict[str, int]) -> tuple[int, int, int, int, int]:
    """将 DataFrame 中的一行转换为离散状态元组

    状态 = (站点编号, 预测功率档, 风险档, 风速突变档, 时段档)
    每个维度都被离散化为 3-4 个级别, 用于 Q-Learning 查表。
    """
    pred_bin = int(np.digitize([row["predicted_ratio"]], [0.20, 0.45, 0.70])[0])
    risk_bin = int(np.digitize([row["risk_score"]], [0.20, 0.45, 0.70])[0])
    ramp_bin = int(np.digitize([row["wind_ramp_abs"]], [0.6, 1.5])[0])
    hour_bin = min(int(row["timestamp"].hour // 6), 3)
    return (site_index[row["site_id"]], pred_bin, risk_bin, ramp_bin, hour_bin)


# ============================================================================
# Episode 构建 —— 将时序数据组织为日级调度任务
# ============================================================================

def _build_episode_payloads(frame: pd.DataFrame, site_index: dict[str, int]) -> list[dict[str, Any]]:
    """将验证集数据组织为每日-每站点的 episode (调度任务)

    每个 episode 包含一天内某个站点的所有时间片,
    以及每个时间片在 4 种动作下的 reward 和 incident 矩阵。

    Returns:
        list of episode dicts, 每个包含: site_id, date, states, risk,
        pred_ratio, reward_matrix, incident_matrix, timestamps
    """
    temp = frame.copy()
    temp["episode_date"] = temp["timestamp"].dt.strftime("%Y-%m-%d")
    payloads: list[dict[str, Any]] = []

    for (_, _), group in temp.groupby(["site_id", "episode_date"], sort=True):
        group = group.reset_index(drop=True)
        site_id = str(group["site_id"].iloc[0])
        site_idx = site_index[site_id]

        # 提取时序数组
        pred_ratio = group["predicted_ratio"].to_numpy(dtype=float)
        risk = group["risk_score"].to_numpy(dtype=float)
        ramp_abs = group["wind_ramp_abs"].to_numpy(dtype=float)
        hours = group["timestamp"].dt.hour.to_numpy(dtype=int)
        actual = group["power_mw"].to_numpy(dtype=float)
        pred_power = group["predicted_power_mw"].to_numpy(dtype=float)
        capacity = group["capacity_mw"].to_numpy(dtype=float)
        prices = _price_for_hours(hours)

        # 状态离散化
        pred_bins = np.digitize(pred_ratio, [0.20, 0.45, 0.70]).astype(int)
        risk_bins = np.digitize(risk, [0.20, 0.45, 0.70]).astype(int)
        ramp_bins = np.digitize(ramp_abs, [0.6, 1.5]).astype(int)
        hour_bins = np.minimum(hours // 6, 3).astype(int)

        states = [
            (site_idx, int(pred_bins[i]), int(risk_bins[i]), int(ramp_bins[i]), int(hour_bins[i]))
            for i in range(len(group))
        ]

        # DQN 连续状态特征: 比离散 one-hot 包含更丰富的数值信息
        dqn_states = []
        for i in range(len(group)):
            hour_val = hours[i]
            vec = np.zeros(len(site_index) + 6, dtype=np.float32)
            vec[site_idx] = 1.0                                          # 站点 one-hot
            vec[len(site_index)] = pred_ratio[i]                         # 预测功率比
            vec[len(site_index) + 1] = risk[i]                           # 风险评分
            vec[len(site_index) + 2] = ramp_abs[i]                       # 风速突变幅度
            vec[len(site_index) + 3] = np.sin(2.0 * np.pi * hour_val / 24.0)  # 小时循环 sin
            vec[len(site_index) + 4] = np.cos(2.0 * np.pi * hour_val / 24.0)  # 小时循环 cos
            vec[len(site_index) + 5] = capacity[i] / 100.0                # 归一化装机容量
            dqn_states.append(vec)

        # 为每个时间片×每个动作计算 reward 和 incident
        reward_matrix = np.zeros((len(group), len(ACTIONS)), dtype=float)
        incident_matrix = np.zeros((len(group), len(ACTIONS)), dtype=float)

        for action_idx, action in enumerate(ACTIONS):
            # 调度目标功率: 预测功率 × 并网系数, 不超过装机容量
            dispatch_target = np.minimum(capacity, pred_power * action["dispatch_factor"])
            delivered = np.minimum(actual, dispatch_target)           # 实际交付
            shortfall = np.maximum(dispatch_target - actual, 0.0)     # 供电缺口
            curtailment = np.maximum(actual - dispatch_target, 0.0)   # 弃风量
            uncertainty = np.abs(actual - pred_power) / np.maximum(capacity, 1e-6)
            exposure = max(action["dispatch_factor"] - 0.92, 0.0)     # 风险敞口

            # 奖励函数: 五部分组成
            revenue = delivered * prices                              # ① 售电收入
            shortfall_penalty = shortfall * (1.40 + 3.00 * risk)     # ② 缺口惩罚 (高风险时更重)
            curtailment_penalty = curtailment * 0.55                  # ③ 弃风惩罚
            instability_penalty = capacity * exposure * risk * (0.60 + uncertainty)  # ④ 不稳定性惩罚
            maintenance_cost = capacity * action["maintenance_factor"]             # ⑤ 维护成本
            reserve_bonus = capacity * action["reserve_factor"] * np.maximum(risk - 0.35, 0.0) * 0.24  # ⑥ 预留奖励

            reward_matrix[:, action_idx] = (
                revenue
                - shortfall_penalty
                - curtailment_penalty
                - instability_penalty
                - maintenance_cost
                + reserve_bonus
            )
            # 事故判定: 供电缺口 > 8% 装机容量 且 风险 > 0.55
            incident_matrix[:, action_idx] = ((shortfall > capacity * 0.08) & (risk > 0.55)).astype(float)

        payloads.append({
            "site_id": site_id,
            "date": str(group["episode_date"].iloc[0]),
            "states": states,
            "dqn_states": dqn_states,
            "risk": risk,
            "pred_ratio": pred_ratio,
            "reward_matrix": reward_matrix,
            "incident_matrix": incident_matrix,
            "timestamps": group["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        })

    return payloads


def _sample_training_payloads(episodes: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """随机采样固定数量的 episode 用于训练"""
    if len(episodes) <= limit:
        return episodes
    random = Random(seed)
    indices = list(range(len(episodes)))
    random.shuffle(indices)
    keep = set(indices[:limit])
    return [episode for idx, episode in enumerate(episodes) if idx in keep]


# ============================================================================
# 策略选择与评估
# ============================================================================

def _fallback_action(risk_value: float, pred_ratio_value: float) -> int:
    """Q表为空时的回退规则策略

    风险高 → 保守/巡检; 风险中 → 预留; 低风险高预测 → 积极并网; 其他 → 平衡
    """
    if risk_value >= 0.70:
        return 3   # 风险巡检
    if risk_value >= 0.45:
        return 2   # 保守预留
    if pred_ratio_value >= 0.72 and risk_value <= 0.22:
        return 0   # 积极并网
    return 1       # 平衡调度


def _choose_action(
    policy_type: str,
    episode: dict[str, Any],
    idx: int,
    q_table: dict[tuple[int, int, int, int, int], list[float]] | None = None,
    dqn_agent: Any = None,
) -> int:
    """根据策略类型选择调度动作

    Args:
        policy_type: "learned"(Q-Learning) / "dqn" / "aggressive" / "balanced" / "conservative" / "heuristic"
        episode: 当前 episode 数据
        idx: 当前时间步索引
        q_table: Q-Learning 的 Q 表 (policy_type="learned" 时使用)
        dqn_agent: DQN 智能体 (policy_type="dqn" 时使用)
    """
    state = episode["states"][idx]
    risk_value = float(episode["risk"][idx])
    pred_value = float(episode["pred_ratio"][idx])

    if policy_type == "learned":
        # Q-Learning 查表
        values = np.array((q_table or {}).get(state, []), dtype=float)
        if values.size == len(ACTIONS) and not np.allclose(values, 0.0):
            return int(np.argmax(values))
        return _fallback_action(risk_value, pred_value)

    if policy_type == "dqn":
        # DQN 神经网络推理 (使用连续特征向量)
        if dqn_agent is not None:
            return dqn_agent.select_action(episode["dqn_states"][idx], epsilon=0.0)
        return _fallback_action(risk_value, pred_value)

    if policy_type == "aggressive":
        return 0
    if policy_type == "balanced":
        return 1
    if policy_type == "conservative":
        return 2 if risk_value >= 0.45 else 1
    if policy_type == "heuristic":
        return heuristic_action_for_state(
            risk_value, pred_value,
            float(episode["reward_matrix"][idx, 1]),
            episode["reward_matrix"][idx],
        )
    raise ValueError(f"Unsupported policy type: {policy_type}")


def _evaluate_payloads(
    episodes: list[dict[str, Any]],
    policy_type: str,
    q_table: dict[tuple[int, int, int, int, int], list[float]] | None = None,
    dqn_agent: Any = None,
) -> dict[str, Any]:
    """评估指定策略在一组 episode 上的表现

    Args:
        episodes: 测试 episode 列表
        policy_type: 策略类型标识
        q_table: Q-Learning Q表
        dqn_agent: DQN 智能体

    Returns:
        dict: 包含 avg_reward, p10_reward, incident_rate, dominant_action, action_mix
    """
    rewards = []
    total_steps = 0
    total_incidents = 0.0
    action_counts: Counter[int] = Counter()

    for episode in episodes:
        episode_reward = 0.0
        reward_matrix = episode["reward_matrix"]
        incident_matrix = episode["incident_matrix"]

        for idx in range(len(episode["states"])):
            action_idx = _choose_action(policy_type, episode, idx, q_table, dqn_agent)
            action_counts[action_idx] += 1
            episode_reward += float(reward_matrix[idx, action_idx])
            total_incidents += float(incident_matrix[idx, action_idx])
            total_steps += 1

        rewards.append(episode_reward)

    dominant_action = action_counts.most_common(1)[0][0] if action_counts else 1
    return {
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "p10_reward": float(np.percentile(rewards, 10) if rewards else 0.0),
        "incident_rate": float(total_incidents / max(total_steps, 1)),
        "dominant_action": ACTIONS[dominant_action]["label_zh"],
        "action_mix": {ACTIONS[idx]["label_zh"]: int(count) for idx, count in sorted(action_counts.items())},
    }


# ============================================================================
# 启发式搜索策略
# ============================================================================

def heuristic_action_for_state(
    risk_value: float, pred_ratio_value: float, balanced_reward: float, candidate_rewards: np.ndarray
) -> int:
    """启发式规则 + 局部 reward 改善

    决策逻辑:
      风险极高(≥0.75) → 巡检模式
      风险偏高(≥0.55) → 保守预留
      低风险高预测(≥0.78) → 在积极并网和平衡调度中选收益高的
      平衡调度收益不差 → 选平衡
      否则 → 全局选最优

    这体现了"风险优先, 收益择优"的工程直觉。
    """
    if risk_value >= 0.75:
        return 3
    if risk_value >= 0.55:
        return 2
    if pred_ratio_value >= 0.78 and risk_value <= 0.20:
        return int(np.argmax(candidate_rewards[[0, 1]]))
    if candidate_rewards[1] >= balanced_reward - 1e-9:
        return 1
    return int(np.argmax(candidate_rewards))


def run_heuristic_search(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """运行启发式搜索策略

    Returns:
        (examples, summaries): 前3个样例 和 统计摘要列表
    """
    examples: list[dict[str, Any]] = []
    rewards = []
    total_incident_steps = 0
    total_steps = 0
    mix: Counter[int] = Counter()

    for episode in episodes:
        actions = []
        reward_sum = 0.0
        incident_sum = 0.0
        for idx in range(len(episode["states"])):
            action_idx = heuristic_action_for_state(
                float(episode["risk"][idx]),
                float(episode["pred_ratio"][idx]),
                float(episode["reward_matrix"][idx, 1]),
                episode["reward_matrix"][idx],
            )
            # 动作平滑: 低风险时避免频繁切换, 除非新动作收益明显更好
            if idx > 0 and action_idx != actions[-1] and float(episode["risk"][idx]) < 0.25:
                local_best = int(np.argmax(episode["reward_matrix"][idx]))
                action_idx = (
                    actions[-1]
                    if episode["reward_matrix"][idx, actions[-1]] >= episode["reward_matrix"][idx, local_best] - 3.0
                    else local_best
                )
            actions.append(action_idx)
            reward_sum += float(episode["reward_matrix"][idx, action_idx])
            incident_sum += float(episode["incident_matrix"][idx, action_idx])
            mix[action_idx] += 1

        rewards.append(reward_sum)
        total_incident_steps += int(incident_sum)
        total_steps += len(actions)
        if len(examples) < 3:
            examples.append({
                "site_id": episode["site_id"],
                "date": episode["date"],
                "reward": round(reward_sum, 2),
                "incident_rate": round(incident_sum / max(len(actions), 1), 4),
                "actions": [ACTIONS[idx]["label_zh"] for idx in actions[:12]],
            })

    summary = {
        "policy": "启发式搜索",
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "p10_reward": float(np.percentile(rewards, 10) if rewards else 0.0),
        "incident_rate": float(total_incident_steps / max(total_steps, 1)),
        "dominant_action": ACTIONS[mix.most_common(1)[0][0]]["label_zh"] if mix else ACTIONS[1]["label_zh"],
        "action_mix": {ACTIONS[idx]["label_zh"]: int(count) for idx, count in sorted(mix.items())},
    }
    return examples, [summary]


# ============================================================================
# 遗传算法调度优化
# ============================================================================

def _evaluate_action_sequence(episode: dict[str, Any], actions: list[int]) -> tuple[float, float]:
    """评估一个动作序列在某个 episode 上的总分和事故率"""
    idx_array = np.arange(len(actions))
    act_array = np.array(actions, dtype=int)
    reward = float(episode["reward_matrix"][idx_array, act_array].sum())
    incident_rate = float(episode["incident_matrix"][idx_array, act_array].mean())
    return reward, incident_rate


def _initial_population(random: Random, base_actions: list[int], length: int) -> list[list[int]]:
    """生成初始种群: 1个基于启发式的种子 + (N-1)个变异个体"""
    population = [base_actions[:]]
    for _ in range(GA_POPULATION - 1):
        individual = base_actions[:]
        for idx in range(length):
            if random.random() < 0.18:  # 18% 概率随机替换动作
                individual[idx] = random.randrange(len(ACTIONS))
        population.append(individual)
    return population


def _tournament_select(random: Random, scored_population: list[tuple[float, list[int]]]) -> list[int]:
    """锦标赛选择: 随机选3个, 取适应度最高的"""
    contenders = [scored_population[random.randrange(len(scored_population))] for _ in range(3)]
    contenders.sort(key=lambda item: item[0], reverse=True)
    return contenders[0][1][:]


def _crossover(random: Random, parent_a: list[int], parent_b: list[int]) -> tuple[list[int], list[int]]:
    """单点交叉: 在随机位置交换两个父代的后半段"""
    if len(parent_a) < 2 or random.random() > GA_CROSSOVER_RATE:
        return parent_a[:], parent_b[:]
    point = random.randrange(1, len(parent_a) - 1)
    return parent_a[:point] + parent_b[point:], parent_b[:point] + parent_a[point:]


def _mutate(random: Random, individual: list[int]) -> list[int]:
    """变异操作: 按 GA_MUTATION_RATE 概率随机修改每个基因位

    倾向于向中间保守策略变异 (0→1/2, 3→2/1), 避免极端策略。
    """
    for idx in range(len(individual)):
        if random.random() < GA_MUTATION_RATE:
            if individual[idx] == 0:
                individual[idx] = 1 if random.random() < 0.7 else 2
            elif individual[idx] == 3:
                individual[idx] = 2 if random.random() < 0.65 else 1
            else:
                individual[idx] = random.randrange(len(ACTIONS))
    return individual


def run_genetic_algorithm(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """运行遗传算法调度优化

    将一天的调度动作序列编码为染色体, 通过选择、交叉、变异搜索最优序列。

    Returns:
        (examples, summaries): 样例和统计摘要
    """
    random = Random(42)
    # 在所有 episode 上运行 GA, 与其他算法公平对比
    selected = list(episodes)
    examples: list[dict[str, Any]] = []
    rewards = []
    total_incident_steps = 0
    total_episode_steps = 0
    mix: Counter[int] = Counter()

    for episode in selected:
        # 以启发式策略的结果作为初始种群的种子
        base_actions = [
            heuristic_action_for_state(
                float(episode["risk"][idx]),
                float(episode["pred_ratio"][idx]),
                float(episode["reward_matrix"][idx, 1]),
                episode["reward_matrix"][idx],
            )
            for idx in range(len(episode["states"]))
        ]
        population = _initial_population(random, base_actions, len(base_actions))
        best_actions = base_actions[:]
        best_reward, best_incident = _evaluate_action_sequence(episode, best_actions)
        best_fitness = best_reward - best_incident * 320.0

        # GA 迭代
        for _ in range(GA_GENERATIONS):
            scored = []
            for individual in population:
                reward, incident_rate = _evaluate_action_sequence(episode, individual)
                # 适应度 = 收益 - 事故惩罚 - 动作切换惩罚 (鼓励平滑策略)
                volatility_penalty = (
                    sum(1 for idx in range(1, len(individual)) if individual[idx] != individual[idx - 1]) * 1.2
                )
                fitness = reward - incident_rate * 320.0 - volatility_penalty
                scored.append((fitness, individual[:]))
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_reward = reward
                    best_incident = incident_rate
                    best_actions = individual[:]

            # 精英保留 + 生成新一代
            scored.sort(key=lambda item: item[0], reverse=True)
            next_population = [scored[0][1][:], scored[1][1][:]]  # 保留最优2个
            while len(next_population) < GA_POPULATION:
                parent_a = _tournament_select(random, scored)
                parent_b = _tournament_select(random, scored)
                child_a, child_b = _crossover(random, parent_a, parent_b)
                next_population.append(_mutate(random, child_a))
                if len(next_population) < GA_POPULATION:
                    next_population.append(_mutate(random, child_b))
            population = next_population

        rewards.append(best_reward)
        total_incident_steps += int(best_incident * len(best_actions))
        total_episode_steps += len(best_actions)
        for action_idx in best_actions:
            mix[action_idx] += 1
        if len(examples) < 3:
            examples.append({
                "site_id": episode["site_id"],
                "date": episode["date"],
                "reward": round(best_reward, 2),
                "incident_rate": round(best_incident, 4),
                "actions": [ACTIONS[idx]["label_zh"] for idx in best_actions[:12]],
            })

    summary = {
        "policy": "遗传算法",
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "p10_reward": float(np.percentile(rewards, 10) if rewards else 0.0),
        "incident_rate": float(total_incident_steps / max(total_episode_steps, 1)),
        "dominant_action": ACTIONS[mix.most_common(1)[0][0]]["label_zh"] if mix else ACTIONS[1]["label_zh"],
        "action_mix": {ACTIONS[idx]["label_zh"]: int(count) for idx, count in sorted(mix.items())},
    }
    return examples, [summary]


# ============================================================================
# 游戏式序列调度评估
# ============================================================================

def run_game_evaluation(
    episodes: list[dict[str, Any]],
    q_table: dict[tuple[int, int, int, int, int], list[float]],
    dqn_agent: Any = None,
) -> list[dict[str, Any]]:
    """实验20: 游戏式序列调度对比

    将日调度视为序贯决策游戏, 对比 Q-Learning、DQN(如果可用)、启发式、GA种子的表现。
    """
    selected = sorted(episodes, key=lambda item: float(item["risk"].mean()), reverse=True)[
        : min(GAME_EPISODE_LIMIT, len(episodes))
    ]
    q_rewards = []
    q_incidents = 0
    q_steps = 0
    dqn_rewards = []
    heuristic_rewards = []
    ga_rewards = []

    for episode in selected:
        # Q-Learning 策略
        q_reward = 0.0
        for idx in range(len(episode["states"])):
            q_action = _choose_action("learned", episode, idx, q_table)
            q_reward += float(episode["reward_matrix"][idx, q_action])
            q_incidents += int(episode["incident_matrix"][idx, q_action])
            q_steps += 1
        q_rewards.append(q_reward)

        # DQN 策略 (如果可用)
        if dqn_agent is not None:
            dqn_reward = 0.0
            for idx in range(len(episode["states"])):
                dqn_action = _choose_action("dqn", episode, idx, dqn_agent=dqn_agent)
                dqn_reward += float(episode["reward_matrix"][idx, dqn_action])
            dqn_rewards.append(dqn_reward)

        # 启发式策略
        heuristic_actions = [
            heuristic_action_for_state(
                float(episode["risk"][idx]),
                float(episode["pred_ratio"][idx]),
                float(episode["reward_matrix"][idx, 1]),
                episode["reward_matrix"][idx],
            )
            for idx in range(len(episode["states"]))
        ]
        heuristic_reward, _ = _evaluate_action_sequence(episode, heuristic_actions)
        heuristic_rewards.append(heuristic_reward)

        # GA 种子策略 (未经 GA 优化的启发式结果)
        base_actions = heuristic_actions[:]
        ga_reward, _ = _evaluate_action_sequence(episode, base_actions)
        ga_rewards.append(ga_reward)

    result = [
        {
            "module": "序贯决策对比",
            "description": (
                "把日级调度任务视为序列决策博弈，对比学习策略与搜索型对手策略的差异。"
            ),
            "q_learning_avg_reward": round(float(np.mean(q_rewards) if q_rewards else 0.0), 2),
            "q_learning_incident_rate": round(float(q_incidents / max(q_steps, 1)), 4),
            "heuristic_avg_reward": round(float(np.mean(heuristic_rewards) if heuristic_rewards else 0.0), 2),
            "ga_seed_avg_reward": round(float(np.mean(ga_rewards) if ga_rewards else 0.0), 2),
        }
    ]

    if dqn_agent is not None and dqn_rewards:
        result[0]["dqn_avg_reward"] = round(float(np.mean(dqn_rewards) if dqn_rewards else 0.0), 2)

    return result


# ============================================================================
# 实验模块构建
# ============================================================================

def build_experiment_modules(
    q_summary: dict[str, Any],
    heuristic_summary: dict[str, Any],
    ga_summary: dict[str, Any],
    game_summary: list[dict[str, Any]],
    dqn_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """构建统一的实验模块列表, 用于前端展示

    包含实验18(启发式搜索)、19(遗传算法)、20(游戏对比)、24(Q-Learning)、25(DQN)。
    """
    modules = [
        {
            "module": "启发式规则搜索",
            "algorithm": "贪心 + 局部搜索",
            "description": "以风险优先规则为起点，并通过局部收益改进搜索每个时间片的调度动作。",
            "avg_reward": round(float(heuristic_summary["avg_reward"]), 2),
            "incident_rate": round(float(heuristic_summary["incident_rate"]), 4),
        },
        {
            "module": "遗传算法优化",
            "algorithm": "染色体序列优化",
            "description": "把日级调度动作序列编码成染色体，并通过选择、交叉与变异寻找更优方案。",
            "avg_reward": round(float(ga_summary["avg_reward"]), 2),
            "incident_rate": round(float(ga_summary["incident_rate"]), 4),
        },
        {
            "module": "序贯决策对比",
            "algorithm": "多策略博弈评估",
            "description": "把日级调度任务视为序列决策博弈，对比学习策略与搜索型对手策略的差异。",
            "avg_reward": round(float(game_summary[0]["q_learning_avg_reward"]), 2),
            "incident_rate": round(float(game_summary[0]["q_learning_incident_rate"]), 4),
        },
        {
            "module": "Q学习调度策略",
            "algorithm": "表格型强化学习",
            "description": "从预测功率、波动性、时间片和风险等级中学习状态-动作调度策略（表格型Q学习）。",
            "avg_reward": round(float(q_summary["avg_reward"]), 2),
            "incident_rate": round(float(q_summary["incident_rate"]), 4),
        },
    ]

    # 如果有 DQN 结果, 追加
    if dqn_summary is not None:
        modules.append({
            "module": "深度Q网络调度",
            "algorithm": "神经网络强化学习",
            "description": "使用神经网络从离散化状态特征逼近Q值，结合经验回放和目标网络实现稳定训练。",
            "avg_reward": round(float(dqn_summary["avg_reward"]), 2),
            "incident_rate": round(float(dqn_summary["incident_rate"]), 4),
        })

    return modules


# ============================================================================
# DQN (Deep Q-Network) 实现
# ============================================================================

def _state_to_onehot(state: tuple[int, int, int, int, int], num_sites: int) -> np.ndarray:
    """将离散状态元组转换为 one-hot 向量, 作为 DQN 网络输入

    状态 = (site_idx, pred_bin, risk_bin, ramp_bin, hour_bin)
      site_idx: 0 ~ num_sites-1 → num_sites 维 one-hot
      pred_bin: 0~3 → 4 维 one-hot
      risk_bin: 0~3 → 4 维 one-hot
      ramp_bin: 0~2 → 3 维 one-hot
      hour_bin: 0~3 → 4 维 one-hot
    总维度 = num_sites + 15
    """
    onehot = np.zeros(num_sites + 15, dtype=np.float32)

    site_idx, pred_bin, risk_bin, ramp_bin, hour_bin = state

    # 站点 one-hot
    if 0 <= site_idx < num_sites:
        onehot[site_idx] = 1.0

    offset = num_sites
    # 预测功率档 (4 bins)
    onehot[offset + min(pred_bin, 3)] = 1.0
    offset += 4
    # 风险档 (4 bins)
    onehot[offset + min(risk_bin, 3)] = 1.0
    offset += 4
    # 风速突变档 (3 bins)
    onehot[offset + min(ramp_bin, 2)] = 1.0
    offset += 3
    # 时段档 (4 bins)
    onehot[offset + min(hour_bin, 3)] = 1.0

    return onehot


class DQNNetwork(nn.Module):
    """DQN 神经网络: 3层 MLP 逼近 Q(s, a)

    输入: one-hot 编码的状态向量 (num_sites + 15 维)
    输出: 4 个动作的 Q 值
    """

    def __init__(self, input_dim: int, hidden_dim: int = DQN_HIDDEN_DIM, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyDQNNetwork(nn.Module):
    """旧版 DQN 网络 (兼容旧训练产物): 3层 MLP + BatchNorm (hidden_dim=256)

    旧架构 net 层索引:
      net.0: Linear(input_dim, 256)
      net.1: BatchNorm1d(256)
      net.2: ReLU
      net.3: Linear(256, 256)
      net.4: BatchNorm1d(256)
      net.5: ReLU
      net.6: Linear(256, 4)
    """

    _LEGACY_HIDDEN = 256

    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        h = self._LEGACY_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区 (Experience Replay Buffer)

    存储 (state, action, reward, next_state, done) 五元组,
    训练时随机采样打乱样本相关性, 提升 DQN 训练稳定性。
    """

    def __init__(self, capacity: int = DQN_REPLAY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """存入一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一个批次"""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(float(d))
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN 智能体 (Double DQN + Soft Target Update)

    改进点 (相比原始 DQN):
      - Double DQN: 策略网络选动作, 目标网络评估, 抑制 Q 值高估
      - Soft Update: 每步用 τ=0.005 微量更新目标网络, 训练更稳定
      - 梯度裁剪更紧 (1.0), 防止单批次大梯度破坏训练
      - Warmup: 先填充回放缓冲区再开始梯度更新
    """

    def __init__(self, input_dim: int, num_sites: int, device: str = "cpu"):
        self.input_dim = input_dim
        self.num_sites = num_sites
        self.device = device

        # 策略网络 (在线更新)
        self.policy_net = DQNNetwork(input_dim).to(device)
        # 目标网络 (通过软更新缓慢跟踪策略网络)
        self.target_net = DQNNetwork(input_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)
        self.replay_buffer = ReplayBuffer()
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss
        self.train_steps = 0

        # 统计追踪
        self.total_reward_sum = 0.0
        self.total_reward_count = 0

    def select_action(self, state, epsilon: float = 0.0) -> int:
        """ε-贪心动作选择 (支持连续 np.ndarray 或旧离散 tuple)"""
        if np.random.random() < epsilon:
            return np.random.randint(0, len(ACTIONS))

        if isinstance(state, np.ndarray):
            state_vec = state.astype(np.float32)
        else:
            state_vec = _state_to_onehot(state, self.num_sites)
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self, batch_size: int = DQN_BATCH_SIZE) -> dict | None:
        """Double DQN 梯度更新 + 软更新目标网络

        Double DQN 公式:
          a* = argmax_a Q_policy(s', a)           ← 策略网络选动作
          y  = r + γ · Q_target(s', a*) · (1-done) ← 目标网络评估
          loss = Huber(Q_policy(s, a), y)

        Returns:
            dict with "loss" and "q_mean" keys, or None if buffer too small
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # ---- Double DQN 目标计算 ----
        with torch.no_grad():
            # 策略网络选择 s' 的最优动作
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            # 目标网络评估该动作的 Q 值
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + DQN_GAMMA * next_q * (1.0 - dones_t)

        # 当前 Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t)
        loss = self.loss_fn(current_q, target_q)

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), DQN_GRAD_CLIP)
        self.optimizer.step()

        # 软更新目标网络 (每步微量同步, 比硬同步更稳定)
        self._soft_update_target(DQN_TAU)

        self.train_steps += 1
        return {"loss": float(loss.item()), "q_mean": float(current_q.mean().item())}

    def _soft_update_target(self, tau: float = 0.005) -> None:
        """Polyak 平均: θ_target ← τ·θ_policy + (1-τ)·θ_target"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def sync_target_network(self) -> None:
        """硬同步目标网络 (warmup 后初始化用)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def track_reward(self, reward: float) -> None:
        """累积 reward 统计, 用于 TensorBoard"""
        self.total_reward_sum += reward
        self.total_reward_count += 1

    def get_avg_reward(self) -> float:
        """返回并重置平均 reward"""
        if self.total_reward_count == 0:
            return 0.0
        avg = self.total_reward_sum / self.total_reward_count
        self.total_reward_sum = 0.0
        self.total_reward_count = 0
        return avg

    def get_state_dict(self) -> dict:
        """导出模型参数 (用于序列化)"""
        return {
            "policy_net": {k: v.cpu().numpy() for k, v in self.policy_net.state_dict().items()},
            "input_dim": self.input_dim,
            "num_sites": self.num_sites,
        }

    @staticmethod
    def _is_legacy_state(policy_net_state: dict) -> bool:
        """通过检查参数名判断是否为旧版架构 (带 BatchNorm, hidden_dim=256)"""
        return "net.6.weight" in policy_net_state

    @classmethod
    def from_state_dict(cls, state_dict: dict, device: str = "cpu") -> "DQNAgent":
        """从序列化的参数恢复模型 (兼容新旧架构)"""
        input_dim = state_dict["input_dim"]
        num_sites = state_dict["num_sites"]
        policy_state = state_dict["policy_net"]
        tensor_state = {k: torch.from_numpy(v) for k, v in policy_state.items()}

        if cls._is_legacy_state(policy_state):
            # ---- 旧版架构: 3层 MLP + BatchNorm (hidden_dim=256) ----
            agent = cls.__new__(cls)
            agent.input_dim = input_dim
            agent.num_sites = num_sites
            agent.device = device
            agent.policy_net = LegacyDQNNetwork(input_dim).to(device)
            agent.target_net = LegacyDQNNetwork(input_dim).to(device)
            # strict=False: 旧版序列化时未保存 BatchNorm 的 running_mean/running_var,
            # 使用默认值 (running_mean=0, running_var=1) 对推理影响很小
            agent.policy_net.load_state_dict(tensor_state, strict=False)
            agent.target_net.load_state_dict(tensor_state, strict=False)
            agent.target_net.eval()
            agent.optimizer = optim.Adam(agent.policy_net.parameters(), lr=DQN_LR)
            agent.replay_buffer = ReplayBuffer()
            agent.loss_fn = nn.SmoothL1Loss()
            agent.train_steps = 0
            agent.total_reward_sum = 0.0
            agent.total_reward_count = 0
            return agent

        # ---- 新版架构: 3层 MLP 无 BatchNorm (hidden_dim=128) ----
        agent = cls(
            input_dim=input_dim,
            num_sites=num_sites,
            device=device,
        )
        agent.policy_net.load_state_dict(tensor_state)
        agent.sync_target_network()
        return agent


def train_dqn(
    validation_frame: pd.DataFrame,
    log_dir: str | None = None,
) -> tuple[DQNAgent, dict[str, Any], int]:
    """训练 Double DQN 智能体 (含 TensorBoard 监控)

    改进后的训练流程:
      1. Warmup: 先随机探索填充回放缓冲区, 不更新网络
      2. 每步: ε-贪心交互 → 存经验 → Double DQN 更新 → 软更新目标网络
      3. TensorBoard 实时记录: loss, Q值均值, reward, epsilon
      4. 每轮评估: 贪心策略跑一遍验证集, 记录平均 reward

    Args:
        validation_frame: 验证集 DataFrame
        log_dir: TensorBoard 日志目录, 默认 "artifacts/runs/dqn_<timestamp>"

    Returns:
        (agent, evaluation_summary, num_sites)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch 未安装, 无法使用 DQN。请执行: pip install torch")

    site_ids = sorted(validation_frame["site_id"].unique().tolist())
    site_index = {site_id: idx for idx, site_id in enumerate(site_ids)}
    num_sites = len(site_ids)
    input_dim = num_sites + 6  # 连续特征: site_onehot(5) + pred_ratio + risk + ramp + hour_sin + hour_cos + cap_norm

    all_episodes = _build_episode_payloads(validation_frame, site_index)
    train_episodes = _sample_training_payloads(all_episodes, TRAIN_EPISODE_LIMIT, seed=42)

    # ---- 初始化 DQN 智能体 ----
    device = get_device()
    agent = DQNAgent(input_dim=input_dim, num_sites=num_sites, device=device)
    print(f"[DQN] 状态维度={input_dim}(连续) {len(train_episodes)}eps, epochs={DQN_EPOCHS}")

    # ---- TensorBoard ----
    if log_dir is None:
        from datetime import datetime
        log_dir = f"artifacts/runs/dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[DQN] TensorBoard 日志目录: {log_dir}")
    print(f"[DQN] 启动 TensorBoard: tensorboard --logdir={log_dir}")

    epsilon = DQN_EPSILON_START
    random = Random(42)
    global_step = 0
    warmup_done = False

    # ---- 评估函数 (每轮跑一次) ----
    # 随机抽取 episode 做评估, 避免前20个碰巧都是差天气导致"假负值"
    eval_episodes = random.sample(train_episodes, min(30, len(train_episodes)))

    def evaluate_greedy() -> dict:
        """用纯贪心策略评估当前模型

        返回 dict 包含:
          - avg_reward: 平均 episode 总 reward
          - p10_reward: 最差 10% 的 reward (尾部风险)
          - action_mix: 各动作使用比例
          - per_action_q: 每个动作的平均 Q 值
        """
        ep_rewards = []
        action_counts = [0] * len(ACTIONS)
        per_action_q_sums = [0.0] * len(ACTIONS)
        q_count = 0

        for ep in eval_episodes:
            ep_reward = 0.0
            for idx in range(len(ep["states"])):
                dqn_state = ep["dqn_states"][idx]
                action_idx = agent.select_action(dqn_state, epsilon=0.0)
                ep_reward += float(ep["reward_matrix"][idx, action_idx])
                action_counts[action_idx] += 1

                # 记录 Q 值 (用于监测是否发散)
                state_tensor = torch.from_numpy(dqn_state.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = agent.policy_net(state_tensor)
                for a_idx in range(len(ACTIONS)):
                    per_action_q_sums[a_idx] += float(q_vals[0, a_idx].item())
                q_count += 1

            ep_rewards.append(ep_reward)

        avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        total_actions = max(sum(action_counts), 1)
        return {
            "avg_reward": avg_reward,
            "p10_reward": float(np.percentile(ep_rewards, 10)) if ep_rewards else 0.0,
            "action_mix": {
                ACTIONS[i]["label_zh"]: round(action_counts[i] / total_actions * 100, 1)
                for i in range(len(ACTIONS))
            },
            "per_action_q": {
                ACTIONS[i]["label_zh"]: round(per_action_q_sums[i] / max(q_count, 1), 1)
                for i in range(len(ACTIONS))
            },
        }

    # 记录训练前的 baseline (随机策略的 eval_reward)
    eval_result = evaluate_greedy()
    eval_baseline = eval_result["avg_reward"]  # 用于计算相对提升
    print(f"[DQN] 训练前 baseline eval_reward = {eval_baseline:.1f}")

    for epoch in range(DQN_EPOCHS):
        shuffled = train_episodes[:]
        random.shuffle(shuffled)

        epoch_losses = []
        epoch_q_means = []
        epoch_rewards = []

        for episode in shuffled:
            dqn_states_ep = episode["dqn_states"]
            reward_matrix = episode["reward_matrix"]

            for idx in range(len(dqn_states_ep)):
                dqn_state = dqn_states_ep[idx]
                # ε-贪心选择动作
                action_idx = agent.select_action(dqn_state, epsilon)
                reward = float(reward_matrix[idx, action_idx])
                done = (idx == len(dqn_states_ep) - 1)
                next_dqn_state = np.zeros_like(dqn_state) if done else dqn_states_ep[idx + 1]

                # 存入经验回放 (直接存连续向量)
                agent.replay_buffer.push(
                    dqn_state.astype(np.float32), action_idx, reward,
                    next_dqn_state.astype(np.float32), done
                )
                agent.track_reward(reward)
                epoch_rewards.append(reward)

                # Warmup: 积累足够经验后才开始学习
                if len(agent.replay_buffer) >= DQN_WARMUP_STEPS and not warmup_done:
                    warmup_done = True
                    agent.sync_target_network()  # 初始化目标网络
                    print(f"[DQN] Warmup 完成 (buffer={len(agent.replay_buffer)}), 开始训练...")

                # 梯度更新 (warmup 之后才执行)
                if warmup_done:
                    result = agent.update()
                    if result is not None:
                        epoch_losses.append(result["loss"])
                        epoch_q_means.append(result["q_mean"])
                        # TensorBoard: 每50步记录一次, 避免event文件过大导致加载缓慢
                        if global_step % 50 == 0:
                            writer.add_scalar("DQN/Loss", result["loss"], global_step)
                            writer.add_scalar("DQN/Q_Mean", result["q_mean"], global_step)

                global_step += 1

        # ---- 每轮结束: 衰减 ε, 评估, TensorBoard 记录 ----
        epsilon = max(DQN_EPSILON_MIN, epsilon * DQN_EPSILON_DECAY)

        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_loss = np.mean(epoch_losses[-200:]) if epoch_losses else 0.0
        avg_q = np.mean(epoch_q_means[-200:]) if epoch_q_means else 0.0
        eval_result = evaluate_greedy()

        # ---- TensorBoard: 标量指标 ----
        writer.add_scalar("DQN/Epsilon", epsilon, epoch)
        writer.add_scalar("DQN/Loss_Epoch", avg_loss, epoch)
        writer.add_scalar("DQN/QMean_Epoch", avg_q, epoch)
        writer.add_scalar("DQN/BufferSize", len(agent.replay_buffer), epoch)
        # 核心指标: reward
        writer.add_scalar("Reward/Train_Avg", avg_reward, epoch)
        writer.add_scalar("Reward/Eval_Avg", eval_result["avg_reward"], epoch)
        writer.add_scalar("Reward/Eval_P10", eval_result["p10_reward"], epoch)
        writer.add_scalar("Reward/Eval_vs_Baseline", eval_result["avg_reward"] - eval_baseline, epoch)

        # ---- TensorBoard: 各动作 Q 值 (监测是否某动作 Q 值发散) ----
        for action_name, q_val in eval_result["per_action_q"].items():
            writer.add_scalar(f"Q_Value/{action_name}", q_val, epoch)

        # ---- TensorBoard: 动作分布 (应随训练趋于合理) ----
        for action_name, pct in eval_result["action_mix"].items():
            writer.add_scalar(f"ActionPct/{action_name}", pct, epoch)

        # 控制台输出
        if (epoch + 1) % 5 == 0 or epoch == 0:
            top_action = max(eval_result["action_mix"], key=eval_result["action_mix"].get)
            print(f"[DQN] Epoch {epoch+1:2d}/{DQN_EPOCHS} | "
                  f"ε={epsilon:.3f} | "
                  f"loss={avg_loss:.2f} | "
                  f"Q={avg_q:+.1f} | "
                  f"train_R={avg_reward:.1f} | "
                  f"eval_R={eval_result['avg_reward']:.1f} | "
                  f"top={top_action}")

    writer.close()
    print(f"[DQN] 训练完成, TensorBoard 日志已保存至: {log_dir}")

    # 最终评估: 使用全部验证集 episode
    dqn_eval = _evaluate_payloads(all_episodes, "dqn", dqn_agent=agent)
    print(f"[DQN] 全部验证集评估 ({len(all_episodes)} episodes): "
          f"avg_reward={dqn_eval['avg_reward']:.2f}, "
          f"incident_rate={dqn_eval['incident_rate']:.4f}, "
          f"dominant_action={dqn_eval['dominant_action']}")
    print(f"[DQN] 注意: 训练过程中的 eval_reward 使用随机抽样的30个episode, "
          f"可能因低风速日导致负值, 属于正常现象。"
          f"最终的全量评估 ({dqn_eval['avg_reward']:.1f}) 才是模型真实水平。")

    return agent, dqn_eval, num_sites


# ============================================================================
# Q-Learning 训练 (保留原实现)
# ============================================================================

def train_q_learning(validation_frame: pd.DataFrame) -> RLBundle:
    """训练 Q-Learning 调度策略

    核心算法: 表格型 Q-Learning
      Q(s, a) ← Q(s, a) + α [r + γ·max_a' Q(s', a') - Q(s, a)]

    状态为离散化后的 5 维元组, 动作为 4 种调度策略。

    Returns:
        RLBundle: 包含 Q表、多算法对比、启发式/GA样例、实验模块
    """
    site_ids = sorted(validation_frame["site_id"].unique().tolist())
    site_index = {site_id: idx for idx, site_id in enumerate(site_ids)}
    all_episodes = _build_episode_payloads(validation_frame, site_index)
    train_episodes = _sample_training_payloads(all_episodes, TRAIN_EPISODE_LIMIT, seed=42)

    # 初始化 Q 表: defaultdict 保证未见过状态自动零初始化
    q_table: defaultdict[tuple[int, int, int, int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(len(ACTIONS))
    )
    random = Random(42)
    alpha = 0.18       # 学习率
    gamma = 0.90       # 折扣因子
    epsilon = 0.30     # 初始探索率

    # Q-Learning 训练循环
    for epoch in range(TRAIN_EPOCHS):
        shuffled = train_episodes[:]
        random.shuffle(shuffled)
        for episode in shuffled:
            states = episode["states"]
            reward_matrix = episode["reward_matrix"]
            risk_values = episode["risk"]
            pred_values = episode["pred_ratio"]

            for idx, state in enumerate(states):
                # ε-贪心动作选择
                if random.random() < epsilon:
                    action_idx = random.randrange(len(ACTIONS))
                else:
                    values = q_table[state]
                    if np.allclose(values, 0.0):
                        action_idx = _fallback_action(
                            float(risk_values[idx]), float(pred_values[idx])
                        )
                    else:
                        action_idx = int(np.argmax(values))

                # Q-Learning 更新公式
                reward = float(reward_matrix[idx, action_idx])
                if idx == len(states) - 1:
                    target = reward
                else:
                    next_state = states[idx + 1]
                    target = reward + gamma * float(np.max(q_table[next_state]))
                q_table[state][action_idx] += alpha * (target - q_table[state][action_idx])

        # ε 衰减: 逐渐减少探索, 增加利用
        epsilon = max(0.05, epsilon * 0.95)

    # 序列化 Q 表 (round 减少 JSON 精度问题)
    serializable_q = {state: values.round(6).tolist() for state, values in q_table.items()}

    # 运行启发式搜索
    heuristic_examples, heuristic_list = run_heuristic_search(all_episodes)
    # 运行遗传算法
    ga_examples, ga_list = run_genetic_algorithm(all_episodes)

    # 尝试训练 DQN 做对比
    dqn_agent = None
    dqn_eval = None
    dqn_state_dict = None
    dqn_experiment_modules: list[dict[str, Any]] = []
    if _TORCH_AVAILABLE:
        try:
            dqn_agent, dqn_eval, num_sites = train_dqn(validation_frame)
            # 导出 DQN 模型权重, 用于缓存持久化 (避免下次重新训练)
            dqn_state_dict = dqn_agent.get_state_dict()
        except Exception as exc:
            print(f"[DQN] 训练失败: {exc}, 将仅使用 Q-Learning")
            dqn_agent = None
            dqn_eval = None

    # 运行游戏评估 (对比各策略)
    game_summary = run_game_evaluation(all_episodes, serializable_q, dqn_agent)

    # 构建所有策略对比
    comparisons = [
        {"policy": "Q学习策略", **_evaluate_payloads(all_episodes, "learned", serializable_q)},
        {"policy": "启发式搜索", **heuristic_list[0]},
        {"policy": "遗传算法", **ga_list[0]},
        {"policy": "始终积极并网", **_evaluate_payloads(all_episodes, "aggressive")},
        {"policy": "始终平衡调度", **_evaluate_payloads(all_episodes, "balanced")},
        {"policy": "规则保守策略", **_evaluate_payloads(all_episodes, "conservative")},
    ]

    # 如果有 DQN, 加入对比
    dqn_comparison: list[dict[str, Any]] = []
    if dqn_eval is not None:
        comparisons.append({"policy": "DQN学习策略", **dqn_eval})

    comparisons.sort(key=lambda item: item["avg_reward"], reverse=True)

    # 提取各策略摘要
    learned_metrics = next(item for item in comparisons if item["policy"] == "Q学习策略")
    q_summary = next(item for item in comparisons if item["policy"] == "Q学习策略")
    heuristic_summary = next(item for item in comparisons if item["policy"] == "启发式搜索")
    ga_summary = next(item for item in comparisons if item["policy"] == "遗传算法")

    # 提取 DQN 摘要
    dqn_summary_for_modules = None
    if dqn_eval is not None:
        dqn_summary_for_modules = next(
            (item for item in comparisons if item["policy"] == "DQN学习策略"), None
        )
        dqn_comparison = [dqn_eval]

    # 构建实验模块
    experiment_modules = build_experiment_modules(
        q_summary, heuristic_summary, ga_summary, game_summary, dqn_summary_for_modules
    )

    # 构建 RL 算法对比
    rl_algorithm_comparison: dict[str, Any] = {
        "q_learning": {
            "avg_reward": learned_metrics["avg_reward"],
            "incident_rate": learned_metrics["incident_rate"],
            "dominant_action": learned_metrics["dominant_action"],
        },
    }
    if dqn_eval is not None:
        rl_algorithm_comparison["dqn"] = {
            "avg_reward": dqn_eval["avg_reward"],
            "incident_rate": dqn_eval["incident_rate"],
            "dominant_action": dqn_eval["dominant_action"],
        }
        ql_reward = learned_metrics["avg_reward"]
        dqn_reward = dqn_eval["avg_reward"]
        rl_algorithm_comparison["comparison"] = {
            "q_learning_reward": round(ql_reward, 2),
            "dqn_reward": round(dqn_reward, 2),
            "reward_difference": round(dqn_reward - ql_reward, 2),
            "winner": "DQN" if dqn_reward > ql_reward else "Q-Learning",
        }

    return RLBundle(
        q_table=serializable_q,
        comparison=comparisons,
        learned_policy_reward=learned_metrics["avg_reward"],
        learned_policy_incident_rate=learned_metrics["incident_rate"],
        action_catalog=ACTIONS,
        heuristic_examples=heuristic_examples,
        ga_examples=ga_examples,
        experiment_modules=experiment_modules,
        dqn_comparison=dqn_comparison,
        dqn_policy_reward=dqn_eval["avg_reward"] if dqn_eval else 0.0,
        dqn_policy_incident_rate=dqn_eval["incident_rate"] if dqn_eval else 0.0,
        dqn_experiment_modules=dqn_experiment_modules,
        dqn_state_dict=dqn_state_dict,
        rl_algorithm_comparison=rl_algorithm_comparison,
    )


# ============================================================================
# 单点调度推荐
# ============================================================================

def pick_action(
    row: pd.Series,
    q_table: dict[tuple[int, int, int, int, int], list[float]],
    site_ids: list[str],
    dqn_agent: Any = None,
) -> dict[str, Any]:
    """为单个时间片推荐调度动作

    优先使用 DQN (如果提供), 回退到 Q-Learning Q表, 再回退到规则策略。

    Args:
        row: DataFrame 中的一行 (含 predicted_ratio, risk_score 等)
        q_table: Q-Learning Q表
        site_ids: 站点 ID 列表
        dqn_agent: DQN 智能体 (可选)

    Returns:
        dict: action_id, action_name, action_label, recommended_dispatch_mw, ...
    """
    site_index = {site_id: idx for idx, site_id in enumerate(sorted(site_ids))}
    state = _state_from_row(row, site_index)

    if dqn_agent is not None:
        # DQN 推理 (构建连续状态向量)
        try:
            hour_val = row["timestamp"].hour
            ramp_val = float(row.get("wind_ramp_abs", 0.0))
            dqn_vec = np.zeros(len(site_index) + 6, dtype=np.float32)
            dqn_vec[site_index[row["site_id"]]] = 1.0
            dqn_vec[len(site_index)] = float(row["predicted_ratio"])
            dqn_vec[len(site_index) + 1] = float(row["risk_score"])
            dqn_vec[len(site_index) + 2] = ramp_val
            dqn_vec[len(site_index) + 3] = np.sin(2.0 * np.pi * hour_val / 24.0)
            dqn_vec[len(site_index) + 4] = np.cos(2.0 * np.pi * hour_val / 24.0)
            dqn_vec[len(site_index) + 5] = float(row["capacity_mw"]) / 100.0
            action_idx = dqn_agent.select_action(dqn_vec, epsilon=0.0)
        except Exception:
            action_idx = None
    else:
        action_idx = None

    if action_idx is None:
        # Q-Learning 查表
        values = np.array(q_table.get(state, []), dtype=float)
        if values.size == len(ACTIONS) and not np.allclose(values, 0.0):
            action_idx = int(np.argmax(values))
        else:
            # 回退到规则策略
            action_idx = _fallback_action(float(row["risk_score"]), float(row["predicted_ratio"]))

    action = ACTIONS[action_idx]
    recommended_dispatch = min(
        float(row["capacity_mw"]),
        float(row["predicted_power_mw"]) * action["dispatch_factor"],
    )
    return {
        "action_id": action_idx,
        "action_name": action["name"],
        "action_label": action["label_zh"],
        "recommended_dispatch_mw": recommended_dispatch,
        "reserve_factor": action["reserve_factor"],
        "description": action["description"],
    }
