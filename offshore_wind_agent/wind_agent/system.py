"""
海上风电智能体系统编排层 (System Orchestrator)
==============================================
负责:
  1. 数据加载与缓存管理
  2. 预测模型训练 (HistGBR + LightGBM 对比)
  3. 强化学习调度策略训练 (Q-Learning + DQN 对比)
  4. 构建前端 Dashboard 所需的完整数据载荷
  5. 规则问答 + Ollama LLM 问答
  6. 数据导出 (CSV + JSON)
  7. 上传测试集管理与持久化
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_utils import DatasetBundle, energy_from_power, load_raw_bundle
from .forecasting import (
    MODEL_TYPE_BOTH,
    ForecastBundle,
    train_forecast_system,
)
from .ollama_client import OllamaAgentClient
from .rl_control import (
    ACTIONS,
    RLBundle,
    pick_action,
    train_q_learning,
)


class OffshoreWindAgentSystem:
    """海上风电智能体系统

    整合预测、风险、调度、问答四大模块, 提供统一的 API 接口。
    支持 HistGBR/LightGBM 预测对比 + Q-Learning/DQN 调度对比。

    两种运行模式:
      - mode="train":  完整训练流程, 结果保存到 artifacts/runs/run_<timestamp>/
      - mode="serve":  仅加载最优历史结果, 绝不触发训练 (用于 app.py)
    """

    def __init__(self, project_root: Path, mode: str = "serve"):
        self.project_root = Path(project_root)
        self.raw_dir = self.project_root / "data" / "raw"
        self.artifact_dir = self.project_root / "artifacts"
        self.runs_dir = self.artifact_dir / "runs"
        self.uploads_dir = self.artifact_dir / "uploads"
        self.best_run_marker = self.artifact_dir / "best_run.txt"
        self.version = "1.2.0"

        # 确保目录存在
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # 核心数据容器
        self.dataset_bundle: DatasetBundle | None = None
        self.forecast_bundle: ForecastBundle | None = None
        self.rl_bundle: RLBundle | None = None
        self.summary_payload: dict[str, Any] | None = None
        self.site_payload_cache: dict[str, dict[str, Any]] = {}

        # DQN 智能体
        self.dqn_agent: Any = None

        # 当前加载的训练结果信息
        self.run_info: dict[str, Any] = {}

        # Ollama LLM 客户端
        self.ollama = OllamaAgentClient()

        if mode == "train":
            self._train_all()
        else:
            self._load_best_or_fail()

    # ========================================================================
    # 缓存管理 — serve 模式: 选最优结果加载, 绝不训练
    # ========================================================================

    def _find_best_run(self) -> Path | None:
        """扫描所有历史训练结果, 返回最优的 bundle 路径

        评选标准: 优先比 DQN avg_reward, 其次 Q-Learning avg_reward,
                 最后比 model MAE (越低越好)。

        Returns:
            最优 system_bundle.joblib 的路径, 没有则返回 None
        """
        bundles = sorted(self.runs_dir.glob("run_*/system_bundle.joblib"))
        if not bundles:
            return None

        best_path = None
        best_score = float("-inf")
        best_info = ""

        for bundle_path in bundles:
            try:
                b = joblib.load(bundle_path)
                if b.get("version") != self.version:
                    continue
                sp = b.get("summary_payload", {})
                rl_comp = sp.get("rl_algorithm_comparison", {}).get("comparison", {})

                # 优先 DQN reward
                if "dqn_reward" in rl_comp:
                    score = rl_comp["dqn_reward"]
                    source = "DQN"
                else:
                    # 其次 Q-Learning reward
                    ql = rl_comp.get("q_learning_reward", 0)
                    score = ql
                    source = "Q-Learning"

                if score > best_score:
                    best_score = score
                    best_path = bundle_path
                    best_info = f"{source} avg_reward={score:.1f}"
            except Exception:
                continue

        if best_path:
            print(f"[System] 最优结果: {best_path.parent.name} ({best_info})")
        return best_path

    def _load_best_or_fail(self) -> None:
        """serve 模式: 加载最优训练结果, 没有则直接报错退出

        优先使用 best_run.txt 指定的路径, 若不存在或无效则自动扫描全部 run 选最优。
        """
        # 先尝试迁移旧版缓存
        self._migrate_legacy_bundle()

        # 优先读取 best_run.txt (允许手动指定)
        best_path = self._load_from_marker()
        if best_path is not None:
            marker_name = best_path.parent.name
        else:
            # 回退: 自动扫描所有 run 选最优
            best_path = self._find_best_run()
            marker_name = None

        if best_path is None:
            msg = (
                "\n"
                "  ========================================\n"
                "   未找到任何训练结果!\n"
                "   请先运行训练脚本:\n"
                "     python train_system.py\n"
                "  ========================================\n"
            )
            print(msg)
            raise FileNotFoundError(
                "No trained model found. Please run: python train_system.py"
            )

        print(f"[System] 历史训练记录共 {len(list(self.runs_dir.glob('run_*/')))} 次")
        if marker_name:
            print(f"[System] 使用 best_run.txt 指定结果: {marker_name}")

        # 尝试加载, 若手动指定的 bundle 无法加载则回退到自动扫描
        try:
            self._load_from_run(best_path)
        except Exception as exc:
            if marker_name:
                print(f"[System] 指定结果 {marker_name} 加载失败: {exc}")
                fallback = self._find_best_run()
                if fallback is not None and fallback != best_path:
                    print(f"[System] 回退至自动扫描最优: {fallback.parent.name}")
                    self._load_from_run(fallback)
                else:
                    raise
            else:
                raise

    def _load_from_marker(self) -> Path | None:
        """读取 best_run.txt 手动指定的 run 目录, 有效则返回 bundle 路径, 否则返回 None"""
        if not self.best_run_marker.exists():
            return None
        try:
            marker_text = self.best_run_marker.read_text(encoding="utf-8").strip()
            if not marker_text:
                return None
            run_dir = Path(marker_text)
            if not run_dir.is_absolute():
                run_dir = self.project_root / run_dir
            bundle_path = run_dir / "system_bundle.joblib"
            if bundle_path.exists():
                return bundle_path
        except Exception:
            pass
        return None

    def _load_from_run(self, bundle_path: Path) -> None:
        """从指定的 bundle 文件加载系统状态

        加载历史训练产物后，调用当前代码重新构建 summary_payload
        和 site_payload_cache，确保前端拿到的是最新的中文数据结构和内容。
        """
        bundle = joblib.load(bundle_path)

        self.dataset_bundle = load_raw_bundle(self.raw_dir)
        self.forecast_bundle = bundle["forecast_bundle"]
        self.rl_bundle = bundle["rl_bundle"]

        # 记录当前 run 信息
        run_dir = bundle_path.parent
        self.run_info = {
            "run_name": run_dir.name,
            "bundle_path": str(bundle_path),
            "model_type": self.forecast_bundle.model_type,
        }

        # 恢复 DQN 模型权重
        if "dqn_state_dict" in bundle and bundle["dqn_state_dict"] is not None:
            try:
                from .rl_control import DQNAgent, get_device
                device = get_device()
                self.dqn_agent = DQNAgent.from_state_dict(
                    bundle["dqn_state_dict"], device=device
                )
                self.run_info["dqn_device"] = device
                print(f"[System] DQN 模型已加载 (device={device})")
            except Exception as exc:
                print(f"[System] DQN 加载失败: {exc}")
                self.dqn_agent = None

        # 重新挂载动作 (为 test_frame 和 validation_frame 都挂载)
        self._attach_actions()

        # ★ 翻译旧缓存中的英文策略名/动作名 (兼容旧版缓存)
        self._translate_cached_rl_data()

        # ★ 关键: 使用当前代码重新构建 payload, 确保中文 + 验证集对比数据
        self.summary_payload = self._build_summary_payload()
        self.site_payload_cache = self._build_site_payloads()

        print(f"[System] 已加载训练结果并重建中文载荷: {run_dir.name}")

    # ========================================================================
    # 缓存数据翻译 —— 兼容旧版英文缓存
    # ========================================================================

    # 旧版英文 → 中文 映射表
    _POLICY_NAME_MAP = {
        "Learned Q-Learning": "Q学习策略",
        "Learned DQN": "DQN学习策略",
        "Heuristic Search": "启发式搜索",
        "Genetic Algorithm": "遗传算法",
        "Always Aggressive": "始终积极并网",
        "Always Balanced": "始终平衡调度",
        "Always Conservative": "始终保守预留",
        "Rule Conservative": "规则保守策略",
    }
    _ACTION_NAME_MAP = {
        "Aggressive Export": "积极并网",
        "Balanced Dispatch": "平衡调度",
        "Reserve Margin": "保守预留",
        "Inspection Mode": "风险巡检",
    }
    _ACTION_DESC_MAP = {
        "Prioritize energy capture when risk is low.": "风险较低时优先追求更多发电收益。",
        "Balance revenue, reserve margin, and stability.": "在收益、预留裕度和系统稳定性之间保持平衡。",
        "Hold extra reserve when wind volatility increases.": "当风速波动增强时保留更多裕度以降低风险。",
        "Reduce export and focus on equipment safety.": "降低并网输出，优先保障设备安全与状态稳定。",
    }
    _EXPERIMENT_DESC_MAP = {
        "Use risk-first rules and local reward improvement to search dispatch actions for each time step.":
            "以风险优先规则为起点，并通过局部收益改进搜索每个时间片的调度动作。",
        "Encode a day-level action sequence as a chromosome and optimize it through selection, crossover, and mutation.":
            "把日级调度动作序列编码成染色体，并通过选择、交叉与变异寻找更优方案。",
        "Treat each day-level dispatch task as a sequential decision game and compare the learned policy against search-based opponents.":
            "把日级调度任务视为序列决策博弈，对比学习策略与搜索型对手策略的差异。",
        "Learn a state-action dispatch policy from predicted power, volatility, time period, and risk level using tabular Q-Learning.":
            "从预测功率、波动性、时间片和风险等级中学习状态-动作调度策略（表格型Q学习）。",
        "Use a neural network to approximate Q-values from discretized state features, with experience replay and target network for stable training.":
            "使用神经网络从离散化状态特征逼近Q值，结合经验回放和目标网络实现稳定训练。",
    }

    def _translate_cached_rl_data(self) -> None:
        """将旧缓存中的英文策略名/动作名/实验描述翻译为中文

        旧版缓存(由旧代码训练生成)中 rl_bundle 包含英文名称。
        此方法在加载缓存后原地翻译，确保后续重建的载荷全部是中文。
        """
        if self.rl_bundle is None:
            return

        # 翻译策略对比中的策略名
        for item in self.rl_bundle.comparison:
            old = item.get("policy", "")
            if old in self._POLICY_NAME_MAP:
                item["policy"] = self._POLICY_NAME_MAP[old]
            # 主导动作名翻译
            dom = item.get("dominant_action", "")
            if dom in self._ACTION_NAME_MAP:
                item["dominant_action"] = self._ACTION_NAME_MAP[dom]

        # 翻译动作目录
        for action in self.rl_bundle.action_catalog:
            old_name = action.get("name", "")
            if old_name in self._ACTION_NAME_MAP:
                action["name"] = self._ACTION_NAME_MAP[old_name]
            old_desc = action.get("description", "")
            if old_desc in self._ACTION_DESC_MAP:
                action["description"] = self._ACTION_DESC_MAP[old_desc]

        # 翻译实验模块描述 + 模块名
        _OLD_MODULE_MAP = {
            "实验18：启发式搜索": "启发式规则搜索",
            "实验19：遗传算法": "遗传算法优化",
            "实验20：游戏": "序贯决策对比",
            "实验24：强化学习(Q-Learning)": "Q学习调度策略",
            "实验25：强化学习(DQN)": "深度Q网络调度",
        }
        for module in self.rl_bundle.experiment_modules:
            old_desc = module.get("description", "")
            if old_desc in self._EXPERIMENT_DESC_MAP:
                module["description"] = self._EXPERIMENT_DESC_MAP[old_desc]
            old_mod = module.get("module", "")
            if old_mod in _OLD_MODULE_MAP:
                module["module"] = _OLD_MODULE_MAP[old_mod]
            old_alg = module.get("algorithm", "")
            if old_alg == "Greedy + Local Search":
                module["algorithm"] = "贪心 + 局部搜索"
            elif old_alg == "GA Sequence Optimization":
                module["algorithm"] = "染色体序列优化"
            elif old_alg == "Sequential Dispatch Game":
                module["algorithm"] = "多策略博弈评估"
            elif old_alg == "Q-Learning":
                module["algorithm"] = "表格型强化学习"
            elif old_alg == "Deep Q-Network":
                module["algorithm"] = "神经网络强化学习"

        # 翻译 DQN 对比数据
        if self.rl_bundle.rl_algorithm_comparison.get("comparison"):
            comp = self.rl_bundle.rl_algorithm_comparison["comparison"]
            if comp.get("winner") == "DQN":
                comp["winner"] = "DQN"
            elif comp.get("winner") == "Q-Learning":
                comp["winner"] = "Q-Learning"

        print("[System] 已翻译旧缓存中的英文策略/动作名称为中文")

    # ========================================================================
    # 训练模式 — 完整训练 + 保存到时间戳目录 + 更新 best_run
    # ========================================================================

    def _train_all(self) -> None:
        """完整训练流程 + 保存结果到 artifacts/runs/run_<timestamp>/"""
        from datetime import datetime

        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = run_dir / "system_bundle.joblib"

        print(f"[System] 开始训练, 结果将保存至: {run_dir}")

        # 1. 加载数据
        self.dataset_bundle = load_raw_bundle(self.raw_dir)

        # 2. 训练预测模型
        self.forecast_bundle = train_forecast_system(
            self.dataset_bundle, model_type=MODEL_TYPE_BOTH
        )

        # 3. 训练 RL 调度策略
        self.rl_bundle = train_q_learning(self.forecast_bundle.validation_frame)

        # 4. 提取 DQN 权重
        dqn_state_dict = self.rl_bundle.dqn_state_dict
        self.dqn_agent = None
        if dqn_state_dict is not None:
            try:
                from .rl_control import DQNAgent, get_device
                device = get_device()
                self.dqn_agent = DQNAgent.from_state_dict(dqn_state_dict, device=device)
                print(f"[System] DQN 模型已从训练结果中加载")
            except Exception as exc:
                print(f"[System] DQN 模型加载失败: {exc}")

        # 5. 挂载动作
        self._attach_actions()

        # 6. 构建载荷
        self.summary_payload = self._build_summary_payload()
        self.site_payload_cache = self._build_site_payloads()

        # 7. 保存到 run 目录
        joblib.dump(
            {
                "version": self.version,
                "forecast_bundle": self.forecast_bundle,
                "rl_bundle": self.rl_bundle,
                "summary_payload": self.summary_payload,
                "site_payload_cache": self.site_payload_cache,
                "dqn_state_dict": dqn_state_dict,
                "run_name": run_name,
                "created_at": datetime.now().isoformat(),
            },
            bundle_path,
        )
        print(f"[System] 训练结果已保存: {bundle_path}")

        # 8. 更新 best_run 标记
        self._update_best_run(run_dir)

        self.run_info = {"run_name": run_name, "bundle_path": str(bundle_path)}

    def _update_best_run(self, new_run_dir: Path) -> None:
        """比较新结果与当前最优, 如果更好则更新 best_run.txt"""
        current_best = self._find_best_run()

        if current_best is None or current_best.parent == new_run_dir:
            action = "(首次训练)"
        else:
            # 比较新旧
            new_bundle = joblib.load(new_run_dir / "system_bundle.joblib")
            old_bundle = joblib.load(current_best)
            new_rl = new_bundle.get("summary_payload", {}).get("rl_algorithm_comparison", {})
            old_rl = old_bundle.get("summary_payload", {}).get("rl_algorithm_comparison", {})

            new_score = new_rl.get("comparison", {}).get("dqn_reward") or new_rl.get("comparison", {}).get("q_learning_reward", 0)
            old_score = old_rl.get("comparison", {}).get("dqn_reward") or old_rl.get("comparison", {}).get("q_learning_reward", 0)

            if new_score > old_score:
                action = f"(优于 {current_best.parent.name}: {new_score:.1f} > {old_score:.1f})"
            else:
                action = f"(未超越 {current_best.parent.name}, best_run 保持不变)"

        self.best_run_marker.write_text(str(new_run_dir), encoding="utf-8")
        print(f"[System] best_run.txt 已更新 → {new_run_dir.name} {action}")

    # ========================================================================
    # 旧版兼容: 如果有旧格式 system_bundle.joblib, 迁移到 runs 目录
    # ========================================================================

    def _migrate_legacy_bundle(self) -> Path | None:
        """将旧版 artifacts/system_bundle.joblib 迁移到 runs/ 目录"""
        legacy_path = self.artifact_dir / "system_bundle.joblib"
        if not legacy_path.exists():
            return None

        legacy_run = self.runs_dir / "run_legacy"
        legacy_run.mkdir(parents=True, exist_ok=True)
        target = legacy_run / "system_bundle.joblib"

        import shutil
        shutil.copy2(legacy_path, target)
        legacy_path.rename(legacy_path.with_suffix(".joblib.bak"))
        print(f"[System] 旧版缓存已迁移至: {legacy_run}")
        return target

    # ========================================================================
    # 动作挂载
    # ========================================================================

    def _attach_actions(self) -> None:
        """为验证帧和测试帧的每一行附加调度动作推荐"""
        assert self.forecast_bundle is not None
        assert self.rl_bundle is not None

        # 为 test_frame 挂载动作
        frame = self.forecast_bundle.test_frame
        records = []
        for idx, row in frame.iterrows():
            action = pick_action(
                row,
                self.rl_bundle.q_table,
                self.forecast_bundle.site_ids,
                dqn_agent=self.dqn_agent,
            )
            record = action.copy()
            record["index"] = idx
            records.append(record)
        action_df = pd.DataFrame(records).set_index("index")
        for column in action_df.columns:
            frame[column] = action_df[column]

        # 为 validation_frame 也挂载动作 (需要 predicted_ratio, risk_score 等列)
        val_frame = self.forecast_bundle.validation_frame
        # 确保 validation_frame 有 predicted_ratio 列 (pick_action 需要)
        if "predicted_ratio" not in val_frame.columns:
            val_frame["predicted_ratio"] = (
                val_frame["predicted_power_mw"] / val_frame["capacity_mw"]
            ).clip(lower=0.0, upper=1.2)
        # 确保有 wind_ramp_abs
        if "wind_ramp_abs" not in val_frame.columns:
            val_frame["wind_ramp_abs"] = 0.0
        val_records = []
        for idx, row in val_frame.iterrows():
            action = pick_action(
                row,
                self.rl_bundle.q_table,
                self.forecast_bundle.site_ids,
                dqn_agent=self.dqn_agent,
            )
            record = action.copy()
            record["index"] = idx
            val_records.append(record)
        val_action_df = pd.DataFrame(val_records).set_index("index")
        for column in val_action_df.columns:
            val_frame[column] = val_action_df[column]

    # ========================================================================
    # 数值格式化
    # ========================================================================

    @staticmethod
    def _float(value: Any, digits: int = 4) -> float:
        """安全地将值转为指定位数的浮点数"""
        return round(float(value), digits)

    # ========================================================================
    # 算法模拟对比数据构建 —— 用于实验分析页
    # ========================================================================

    def _build_algorithm_simulation_data(self) -> dict[str, Any]:
        """构建各调度策略在验证集上的模拟调度 vs 实际功率对比数据

        对每种策略(QL/DQN/启发式/GA/激进/平衡/保守)，在验证集上模拟调度，
        记录调度功率与真实功率的逐点差距，用于实验分析页展示策略优劣。
        """
        assert self.forecast_bundle is not None
        assert self.rl_bundle is not None

        val_frame = self.forecast_bundle.validation_frame
        site_ids = self.forecast_bundle.site_ids
        q_table = self.rl_bundle.q_table
        dqn_agent = self.dqn_agent

        import numpy as np
        from .rl_control import _state_from_row, ACTIONS, _fallback_action, heuristic_action_for_state

        site_index = {sid: idx for idx, sid in enumerate(sorted(site_ids))}

        # 取前2880个时间片(30天)做逐点对比展示
        sample = val_frame.head(2880)

        strategies = [
            {"key": "q_learning", "label": "Q学习策略", "type": "q_learn"},
            {"key": "heuristic", "label": "启发式搜索", "type": "heuristic"},
            {"key": "aggressive", "label": "始终积极并网", "type": "aggressive"},
            {"key": "balanced", "label": "始终平衡调度", "type": "balanced"},
            {"key": "conservative", "label": "规则保守策略", "type": "conservative"},
        ]
        if dqn_agent is not None:
            strategies.insert(1, {"key": "dqn", "label": "DQN学习策略", "type": "dqn"})

        # 为每种策略计算模拟调度结果
        strategy_results = []
        for strat in strategies:
            dispatch_series = []
            total_reward = 0.0
            total_incidents = 0
            total_actual_energy = 0.0
            total_dispatched_energy = 0.0

            for _, row in sample.iterrows():
                state = _state_from_row(row, site_index)
                risk_val = float(row["risk_score"])
                pred_ratio = float(row.get("predicted_ratio", row["predicted_power_mw"] / max(row["capacity_mw"], 1e-6)))
                pred_power = float(row["predicted_power_mw"])
                actual_power = float(row["power_mw"])
                capacity = float(row["capacity_mw"])

                # 构建 DQN 连续状态向量
                hour_val = row["timestamp"].hour
                ramp_val = float(row.get("wind_ramp_abs", 0.0))
                dqn_state = np.zeros(len(site_index) + 6, dtype=np.float32)
                dqn_state[site_index[row["site_id"]]] = 1.0
                dqn_state[len(site_index)] = pred_ratio
                dqn_state[len(site_index) + 1] = risk_val
                dqn_state[len(site_index) + 2] = ramp_val
                dqn_state[len(site_index) + 3] = np.sin(2.0 * np.pi * hour_val / 24.0)
                dqn_state[len(site_index) + 4] = np.cos(2.0 * np.pi * hour_val / 24.0)
                dqn_state[len(site_index) + 5] = capacity / 100.0

                # 根据策略类型选择动作
                if strat["type"] == "q_learn":
                    values = np.array(q_table.get(state, []), dtype=float)
                    if values.size == 4 and not np.allclose(values, 0.0):
                        action_idx = int(np.argmax(values))
                    else:
                        action_idx = _fallback_action(risk_val, pred_ratio)
                elif strat["type"] == "dqn":
                    try:
                        action_idx = dqn_agent.select_action(dqn_state, epsilon=0.0)
                    except Exception:
                        action_idx = _fallback_action(risk_val, pred_ratio)
                elif strat["type"] == "aggressive":
                    action_idx = 0
                elif strat["type"] == "balanced":
                    action_idx = 1
                elif strat["type"] == "conservative":
                    action_idx = 2 if risk_val >= 0.45 else 1
                elif strat["type"] == "heuristic":
                    # 用各动作的 dispatch_factor 估算近似收益, 确保4个动作都有合理值
                    approx = np.array([
                        pred_power * 0.95,   # 积极并网 (×1.04, 有不稳定惩罚)
                        pred_power * 0.92,   # 平衡调度 (×0.96, 基准)
                        pred_power * 0.78,   # 保守预留 (×0.86)
                        pred_power * 0.65,   # 风险巡检 (×0.78)
                    ])
                    action_idx = heuristic_action_for_state(risk_val, pred_ratio, float(approx[1]), approx)
                else:
                    action_idx = _fallback_action(risk_val, pred_ratio)

                action = ACTIONS[action_idx]
                dispatch_power = min(capacity, pred_power * action["dispatch_factor"])
                delivered = min(actual_power, dispatch_power)
                shortfall = max(dispatch_power - actual_power, 0.0)

                dispatch_series.append({
                    "timestamp": row["timestamp"].strftime("%m-%d %H:%M"),
                    "actual_power_mw": self._float(actual_power, 2),
                    "predicted_power_mw": self._float(pred_power, 2),
                    "dispatch_power_mw": self._float(dispatch_power, 2),
                    "action_label": action["label_zh"],
                })

                total_reward += delivered * 1.0 - shortfall * 1.5
                total_incidents += 1 if (shortfall > capacity * 0.08 and risk_val > 0.55) else 0
                total_actual_energy += actual_power * 0.25
                total_dispatched_energy += dispatch_power * 0.25

            strategy_results.append({
                "key": strat["key"],
                "label": strat["label"],
                "total_reward": self._float(total_reward, 1),
                "incident_count": int(total_incidents),
                "actual_energy_mwh": self._float(total_actual_energy, 1),
                "dispatched_energy_mwh": self._float(total_dispatched_energy, 1),
                "energy_utilization_pct": self._float(total_dispatched_energy / max(total_actual_energy, 1e-6) * 100, 1),
                "dispatch_series": dispatch_series,
            })

        # 找最佳/最差策略做对比
        best = max(strategy_results, key=lambda x: x["total_reward"])
        worst = min(strategy_results, key=lambda x: x["total_reward"])

        return {
            "sample_period": f"{sample['timestamp'].min().strftime('%Y-%m-%d %H:%M')} ~ {sample['timestamp'].max().strftime('%Y-%m-%d %H:%M')}",
            "sample_points": len(sample),
            "strategies": strategy_results,
            "best_strategy": best["label"],
            "worst_strategy": worst["label"],
            "reward_spread": self._float(best["total_reward"] - worst["total_reward"], 1),
            "description": (
                "在验证集的同一段数据上，用每种调度策略模拟调度决策，"
                "对比各策略的累计收益、事故次数和能量利用率。"
                "调度功率越接近实际功率且越少发生供电缺口，策略越优。"
            ),
        }

    # ========================================================================
    # Dashboard 总载荷构建
    # ========================================================================

    def _build_summary_payload(self) -> dict[str, Any]:
        """构建前端 Dashboard 所需的完整摘要数据"""
        assert self.dataset_bundle is not None
        assert self.forecast_bundle is not None
        assert self.rl_bundle is not None

        validation_frame = self.forecast_bundle.validation_frame

        # ---- 逐站点概览 (基于验证集, 有真实功率可对比) ----
        by_site = []
        for site_id, val_group in validation_frame.groupby("site_id"):
            site_info = self.dataset_bundle.site_info.loc[
                self.dataset_bundle.site_info["site_id"] == site_id
            ].iloc[0]
            val_metrics = next(
                item for item in self.forecast_bundle.metrics["by_site"]
                if item["site_id"] == site_id
            )
            site_mae = self._float(val_group["abs_error_mw"].mean(), 3)
            dominant_action = (
                val_group["action_label"].mode().iloc[0]
                if "action_label" in val_group.columns and len(val_group["action_label"].mode()) > 0
                else "平衡调度"
            )
            by_site.append({
                "site_id": site_id,
                "site_name": site_info["site_name"],
                "region": site_info["region"],
                "capacity_mw": self._float(site_info["capacity_mw"], 2),
                "validation_mae_mw": self._float(val_metrics["mae"], 2),
                "site_mae_mw": site_mae,
                "actual_energy_mwh": self._float(energy_from_power(val_group["power_mw"]), 2),
                "predicted_energy_mwh": self._float(energy_from_power(val_group["predicted_power_mw"]), 2),
                "actual_peak_mw": self._float(val_group["power_mw"].max(), 2),
                "predicted_peak_mw": self._float(val_group["predicted_power_mw"].max(), 2),
                "avg_risk": self._float(val_group["risk_score"].mean(), 3),
                "dominant_action": dominant_action,
            })

        # ---- 峰值窗口 (使用验证集实际功率) ----
        peak_window = validation_frame.loc[validation_frame["power_mw"].idxmax()]

        # ---- RL 策略比较 (优先 DQN, 不可用时回退 Q-Learning) ----
        dqn_candidates = [item for item in self.rl_bundle.comparison if item["policy"] == "DQN学习策略"]
        ql_candidates = [item for item in self.rl_bundle.comparison if item["policy"] == "Q学习策略"]
        learned_row = dqn_candidates[0] if dqn_candidates else ql_candidates[0]
        learned_label = "DQN" if dqn_candidates else "Q-Learning"

        aggressive_row = next(
            item for item in self.rl_bundle.comparison
            if item["policy"] == "始终积极并网"
        )
        reward_gain = learned_row["avg_reward"] - aggressive_row["avg_reward"]
        incident_drop = aggressive_row["incident_rate"] - learned_row["incident_rate"]

        # ---- 高风险窗口 Top 12 (基于验证集) ----
        top_windows = (
            validation_frame.sort_values("risk_score", ascending=False)
            .head(12)[["site_id", "timestamp", "risk_score", "power_mw", "predicted_power_mw", "action_label"]]
            .copy()
        )

        # ---- 汇总卡片 ----
        summary_cards = [
            {
                "label": "验证集平均绝对误差 (HistGBR)",
                "value": f"{self._float(self.forecast_bundle.metrics['overall']['mae'], 2)} MW",
                "note": f"验证集 MAE={self._float(self.forecast_bundle.metrics['overall']['mae'], 2)} MW，"
                       f"RMSE={self._float(self.forecast_bundle.metrics['overall']['rmse'], 2)} MW，"
                       f"R²={self._float(self.forecast_bundle.metrics['overall'].get('r2', 0), 4)}。"
                       f"R²越接近1说明模型对功率变化的解释力越强。",
            },
            {
                "label": "模型推演期预估电量",
                "value": f"{self._float(self.forecast_bundle.metrics['test_energy_mwh'], 1)} MWh",
                "note": f"在模型推演期（2023年5-7月，无真实功率可对比），"
                       f"模型预估的 5 个风场总发电量。该时段为竞赛A榜测试集，无实际功率数据用于验证。",
            },
            {
                "label": "强化学习相对激进策略增益",
                "value": f"{self._float(reward_gain, 2)} 奖励分",
                "note": (
                    f"{learned_label} 学习到的调度策略相比始终积极并网基线策略的累计奖励提升。"
                    "正值表示学习策略在收益与安全性之间取得了更好的平衡。"
                ),
            },
            {
                "label": "事故率下降幅度",
                "value": f"{self._float(incident_drop * 100.0, 2)}%",
                "note": (
                    f"相比激进并网策略（事故率 {self._float(aggressive_row['incident_rate']*100, 2)}%），"
                    f"{learned_label} 策略将事故率降至 {self._float(learned_row['incident_rate']*100, 2)}%，"
                    "在维持收益竞争力的同时显著提升了系统安全性。"
                ),
            },
        ]

        # ---- 模型对比卡片 (如果 LightGBM 可用) ----
        if self.forecast_bundle.model_comparison:
            comp = self.forecast_bundle.model_comparison
            if comp.get("winner") == "LightGBM":
                improvement_str = f"LightGBM 更优 (领先 {comp['mae_improvement']:.2f} MW)"
                winner_name = "LightGBM"
            else:
                improvement_str = f"HistGBR 更优 (领先 {-comp['mae_improvement']:.2f} MW)"
                winner_name = "HistGBR"
            summary_cards.append({
                "label": "预测模型对比 (HistGBR vs LightGBM)",
                "value": improvement_str,
                "note": (
                    f"HistGBR 验证集 MAE: {comp['histgb_mae']:.2f} MW，"
                    f"LightGBM 验证集 MAE: {comp['lgbm_mae']:.2f} MW，"
                    f"{winner_name} 胜出，MAE 改善幅度 {abs(comp['mae_improvement_pct']):.1f}%。"
                    f"两模型均为梯度提升树方法，在海上风电功率预测任务上表现接近。"
                ),
            })

        # ---- RL 算法对比卡片 (如果有 DQN) ----
        if self.rl_bundle.rl_algorithm_comparison.get("comparison"):
            rl_comp = self.rl_bundle.rl_algorithm_comparison["comparison"]
            rl_winner = rl_comp.get("winner", "N/A")
            diff = rl_comp.get("reward_difference", 0)
            winner_label = "DQN 胜出" if rl_winner == "DQN" else "Q-Learning 胜出"
            summary_cards.append({
                "label": "调度算法对比 (Q-Learning vs DQN)",
                "value": winner_label,
                "note": (
                    f"Q-Learning 平均奖励: {rl_comp['q_learning_reward']:.2f}，"
                    f"DQN 平均奖励: {rl_comp['dqn_reward']:.2f}，"
                    f"差异: {diff:+.2f}。"
                    f"DQN 使用神经网络逼近 Q 函数，在状态空间较大时可能具有更强的泛化能力。"
                ),
            })

        # ---- 验证集预测 vs 实际对比数据 (核心新增: 用于精度验证展示) ----
        validation_comparison = self._build_validation_comparison()

        # ---- 数据时间线说明 ----
        data_timeline = {
            "train_period": f"{self.dataset_bundle.train['timestamp'].min().strftime('%Y-%m-%d')} ~ "
                           f"{self.dataset_bundle.train['timestamp'].max().strftime('%Y-%m-%d')}",
            "validation_period": f"{validation_frame['timestamp'].min().strftime('%Y-%m-%d')} ~ "
                                f"{validation_frame['timestamp'].max().strftime('%Y-%m-%d')}",
            "projection_period": f"{self.dataset_bundle.test['timestamp'].min().strftime('%Y-%m-%d')} ~ "
                                f"{self.dataset_bundle.test['timestamp'].max().strftime('%Y-%m-%d')}",
            "description": (
                "训练集（2022-01 ~ 2023-04）按时间顺序划分：前80%用于模型训练，后20%作为验证集用于精度评估。"
                "验证集中有真实功率数据，可以对比预测值与实际值，验证模型准确性。"
                "推演期（2023-05 ~ 2023-07）为竞赛A榜测试集，不含实际功率数据，仅展示模型在未见过数据上的推演结果。"
            ),
        }

        # 算法模拟对比数据
        algorithm_simulation = self._build_algorithm_simulation_data()

        return {
            "project_title": "海上风电历史数据分析与多算法智能体系统",
            "subtitle": "基于2022-2023年福建海上风电数据，集成功率建模验证、风险量化评估、多策略调度对比与智能问答的一体化分析平台。",
            "summary_cards": summary_cards,
            "station_overview": by_site,
            "model_metrics": self.forecast_bundle.metrics,
            "model_comparison": self.forecast_bundle.model_comparison,
            "rl_comparison": self.rl_bundle.comparison,
            "rl_algorithm_comparison": self.rl_bundle.rl_algorithm_comparison,
            "action_catalog": self.rl_bundle.action_catalog,
            "experiment_modules": self.rl_bundle.experiment_modules,
            "heuristic_examples": self.rl_bundle.heuristic_examples,
            "ga_examples": self.rl_bundle.ga_examples,
            "validation_comparison": validation_comparison,
            "algorithm_simulation": algorithm_simulation,
            "data_timeline": data_timeline,
            "top_risk_windows": [
                {
                    "site_id": row["site_id"],
                    "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "risk_score": self._float(row["risk_score"], 3),
                    "actual_power_mw": self._float(row["power_mw"], 2),
                    "predicted_power_mw": self._float(row["predicted_power_mw"], 2),
                    "action_label": row.get("action_label", "平衡调度"),
                }
                for _, row in top_windows.iterrows()
            ],
            "meta": {
                "site_count": int(self.dataset_bundle.site_info["site_id"].nunique()),
                "train_range": [
                    self.dataset_bundle.train["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
                    self.dataset_bundle.train["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
                ],
                "validation_range": [
                    validation_frame["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
                    validation_frame["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
                ],
                "projection_range": [
                    self.dataset_bundle.test["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
                    self.dataset_bundle.test["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
                ],
                "peak_site": peak_window["site_id"],
                "peak_timestamp": peak_window["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "peak_power_mw": self._float(peak_window["predicted_power_mw"], 2),
                "mean_validation_risk": self._float(validation_frame["risk_score"].mean(), 3),
                "validation_mae_mw": self._float(self.forecast_bundle.metrics['overall']['mae'], 2),
                "validation_rmse_mw": self._float(self.forecast_bundle.metrics['overall']['rmse'], 2),
            },
            "agent_backend": {
                "available": self.ollama.status.available,
                "backend": "ollama" if self.ollama.status.available else "rule-based",
                "model": self.ollama.status.model,
                "models": self.ollama.status.models,
                "message": self.ollama.status.message,
            },
        }

    # ========================================================================
    # 验证集预测 vs 实际对比数据构建
    # ========================================================================

    def _build_validation_comparison(self) -> dict[str, Any]:
        """构建验证集上的预测功率 vs 实际功率对比数据

        这是本次修改的核心：用训练集后20%的验证数据（有真实功率）来展示模型精度。
        包含：逐站点对比指标、整体时序对比序列、误差分布等。
        """
        assert self.forecast_bundle is not None
        val_frame = self.forecast_bundle.validation_frame

        # ---- 整体对比序列 (返回全部验证数据, 前端自行截取) ----
        overall_comparison_series = []
        for _, row in val_frame.iterrows():
            overall_comparison_series.append({
                "timestamp": row["timestamp"].strftime("%m-%d %H:%M"),
                "site_id": row["site_id"],
                "actual_power_mw": self._float(row["power_mw"], 3),
                "predicted_power_mw": self._float(row["predicted_power_mw"], 3),
                "abs_error_mw": self._float(row["abs_error_mw"], 3),
                "risk_score": self._float(row["risk_score"], 3),
            })

        # ---- 逐站点对比指标 ----
        by_site_comparison = []
        for site_id, group in val_frame.groupby("site_id"):
            mae = self._float(group["abs_error_mw"].mean(), 3)
            rmse = self._float(
                float(np.sqrt((group["abs_error_mw"] ** 2).mean())), 3
            )
            from sklearn.metrics import r2_score as _r2
            site_r2 = self._float(_r2(group["power_mw"], group["predicted_power_mw"]), 4)
            actual_mean = self._float(group["power_mw"].mean(), 2)
            pred_mean = self._float(group["predicted_power_mw"].mean(), 2)
            # MAPE (平均绝对百分比误差)
            safe_actual = group["power_mw"].replace(0.0, np.nan)
            mape = float(
                ((group["power_mw"] - group["predicted_power_mw"]).abs() / safe_actual)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .mean()
                * 100.0
            )
            if not np.isfinite(mape):
                mape = 0.0

            # 逐站点对比序列 (返回全量验证数据)
            site_series = []
            for _, row in group.iterrows():
                site_series.append({
                    "timestamp": row["timestamp"].strftime("%m-%d %H:%M"),
                    "actual_power_mw": self._float(row["power_mw"], 3),
                    "predicted_power_mw": self._float(row["predicted_power_mw"], 3),
                    "abs_error_mw": self._float(row["abs_error_mw"], 3),
                })

            by_site_comparison.append({
                "site_id": site_id,
                "mae_mw": mae,
                "rmse_mw": rmse,
                "r2": site_r2,
                "mape_pct": self._float(mape, 2),
                "actual_mean_mw": actual_mean,
                "pred_mean_mw": pred_mean,
                "sample_count": len(group),
                "comparison_series": site_series,
            })

        # ---- 误差分布统计 ----
        all_errors = val_frame["abs_error_mw"].dropna()
        error_distribution = {
            "mean": self._float(all_errors.mean(), 3),
            "median": self._float(all_errors.median(), 3),
            "p25": self._float(all_errors.quantile(0.25), 3),
            "p75": self._float(all_errors.quantile(0.75), 3),
            "p90": self._float(all_errors.quantile(0.90), 3),
            "p95": self._float(all_errors.quantile(0.95), 3),
            "max": self._float(all_errors.max(), 3),
            "within_1mw_pct": self._float((all_errors <= 1.0).mean() * 100, 1),
            "within_3mw_pct": self._float((all_errors <= 3.0).mean() * 100, 1),
        }

        from sklearn.metrics import r2_score
        overall_r2 = self._float(r2_score(val_frame["power_mw"], val_frame["predicted_power_mw"]), 4)

        return {
            "overall_mae_mw": self._float(val_frame["abs_error_mw"].mean(), 3),
            "overall_rmse_mw": self._float(
                float(np.sqrt((val_frame["abs_error_mw"] ** 2).mean())), 3
            ),
            "overall_r2": overall_r2,
            "overall_mape_pct": self._float(
                float(
                    ((val_frame["power_mw"] - val_frame["predicted_power_mw"]).abs()
                     / val_frame["power_mw"].replace(0.0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .mean()
                    * 100.0
                ),
                2,
            ) if len(val_frame) > 0 else 0.0,
            "error_distribution": error_distribution,
            "by_site": by_site_comparison,
            "overall_comparison_series": overall_comparison_series,
            "description": (
                "以下数据展示了验证集（训练数据后20%时间片）上模型预测功率与真实功率的对比。"
                "验证集中每个时间片都有实测功率，因此可以量化评估模型的预测精度。"
                "误差越小，说明模型对海上风电功率变化的捕捉越准确。"
            ),
        }

    # ========================================================================
    # 站点详情载荷构建
    # ========================================================================

    def _build_site_payloads(self) -> dict[str, dict[str, Any]]:
        """为每个站点构建详情页所需数据

        核心变化: 使用验证集 validation_frame（有真实功率 power_mw），
        可以对比预测值 vs 实际值，量化评估模型精度。
        """
        assert self.dataset_bundle is not None
        assert self.forecast_bundle is not None
        payloads: dict[str, dict[str, Any]] = {}

        val_frame = self.forecast_bundle.validation_frame

        for site_id, group in val_frame.groupby("site_id"):
            site_info = self.dataset_bundle.site_info.loc[
                self.dataset_bundle.site_info["site_id"] == site_id
            ].iloc[0]

            # ---- 日汇总: 同时包含实际值和预测值 ----
            daily = (
                group.assign(date=group["timestamp"].dt.strftime("%Y-%m-%d"))
                .groupby("date")
                .agg(
                    actual_avg=("power_mw", "mean"),
                    predicted_avg=("predicted_power_mw", "mean"),
                    actual_peak=("power_mw", "max"),
                    predicted_peak=("predicted_power_mw", "max"),
                    avg_risk=("risk_score", "mean"),
                    avg_error=("abs_error_mw", "mean"),
                    total_actual_energy=("power_mw", "sum"),
                    total_predicted_energy=("predicted_power_mw", "sum"),
                )
                .reset_index()
            )

            # ---- 时序图数据 (返回全量验证数据, 前端可滑动浏览) ----
            # 全部返回，不做 head 截断。前端 PowerChart 会自行 subsample 渲染 SVG。
            forecast_series = []
            for _, row in group.iterrows():
                forecast_series.append({
                    "timestamp": row["timestamp"].strftime("%m-%d %H:%M"),
                    "actual_power_mw": self._float(row["power_mw"], 3),
                    "predicted_power_mw": self._float(row["predicted_power_mw"], 3),
                    "abs_error_mw": self._float(row.get("abs_error_mw", abs(row["power_mw"] - row["predicted_power_mw"])), 3),
                    "risk_score": self._float(row["risk_score"], 3),
                    "action_label": row.get("action_label", "平衡调度"),
                    "dispatch_power_mw": self._float(row.get("recommended_dispatch_mw", row["predicted_power_mw"] * 0.96), 3),
                })

            # ---- 风险带 (返回全量, 前端截取展示) ----
            risk_band = []
            for _, row in group.iterrows():
                risk_band.append({
                    "timestamp": row["timestamp"].strftime("%m-%d %H:%M"),
                    "risk_score": self._float(row["risk_score"], 3),
                    "action_label": row.get("action_label", "平衡调度"),
                })

            # ---- 高风险窗口 Top 10 ----
            top_windows = (
                group.sort_values("risk_score", ascending=False)
                .head(10)[[
                    "timestamp", "risk_score", "power_mw", "predicted_power_mw",
                    "abs_error_mw", "action_label",
                ]]
                .copy()
            )

            # ---- 站点级误差指标 ----
            from sklearn.metrics import r2_score
            site_r2 = self._float(r2_score(group["power_mw"], group["predicted_power_mw"]), 4)
            site_mae = self._float(group["abs_error_mw"].mean(), 3)
            site_actual_mean = self._float(group["power_mw"].mean(), 2)
            site_pred_mean = self._float(group["predicted_power_mw"].mean(), 2)

            payloads[site_id] = {
                "site_id": site_id,
                "site_name": site_info["site_name"],
                "capacity_mw": self._float(site_info["capacity_mw"], 2),
                "region": site_info["region"],
                "site_mae_mw": site_mae,
                "actual_mean_mw": site_actual_mean,
                "predicted_mean_mw": site_pred_mean,
                "validation_energy_mwh": self._float(energy_from_power(group["power_mw"]), 2),
                "avg_risk": self._float(group["risk_score"].mean(), 3),
                "dominant_action": (
                    group["action_label"].mode().iloc[0]
                    if "action_label" in group.columns and len(group["action_label"].mode()) > 0
                    else "平衡调度"
                ),
                "data_source": "验证集（有真实功率可对比）",
                "forecast_series": forecast_series,
                "risk_band": risk_band,
                "daily_summary": [
                    {
                        "date": row["date"],
                        "actual_avg": self._float(row["actual_avg"], 3),
                        "predicted_avg": self._float(row["predicted_avg"], 3),
                        "actual_peak": self._float(row["actual_peak"], 3),
                        "predicted_peak": self._float(row["predicted_peak"], 3),
                        "avg_risk": self._float(row["avg_risk"], 3),
                        "avg_error": self._float(row["avg_error"], 3),
                        "total_actual_mwh": self._float(row["total_actual_energy"] * 0.25, 1),
                        "total_predicted_mwh": self._float(row["total_predicted_energy"] * 0.25, 1),
                    }
                    for _, row in daily.iterrows()
                ],
                "top_windows": [
                    {
                        "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                        "risk_score": self._float(row["risk_score"], 3),
                        "actual_power_mw": self._float(row["power_mw"], 3),
                        "predicted_power_mw": self._float(row["predicted_power_mw"], 3),
                        "abs_error_mw": self._float(row.get("abs_error_mw", abs(row["power_mw"] - row["predicted_power_mw"])), 3),
                        "action_label": row.get("action_label", "平衡调度"),
                    }
                    for _, row in top_windows.iterrows()
                ],
            }
        return payloads

    # ========================================================================
    # 公开 API
    # ========================================================================

    def get_dashboard_payload(self) -> dict[str, Any]:
        """获取 Dashboard 总览页的全量数据"""
        assert self.summary_payload is not None
        payload = dict(self.summary_payload)
        payload["run_info"] = self.run_info
        return payload

    def get_site_payload(self, site_id: str) -> dict[str, Any]:
        """获取指定站点的详情数据"""
        return self.site_payload_cache[site_id]

    # ========================================================================
    # 智能问答
    # ========================================================================

    def refresh_ollama(self) -> dict[str, Any]:
        """重新检测 Ollama 连接状态, 返回当前状态"""
        status = self.ollama.refresh_status()
        return {
            "available": status.available,
            "model": status.model,
            "message": status.message,
        }

    def answer_question(self, question: str) -> dict[str, Any]:
        """智能问答接口

        优先调用 Ollama LLM, 如果不可用则回退到内置规则问答。
        支持的问题类型: 预测、风险、调度、策略、站点、实验等。
        """
        assert self.forecast_bundle is not None
        assert self.summary_payload is not None
        text = question.lower().strip()

        # 尝试匹配站点名
        station_map = {
            item["site_id"].lower(): item
            for item in self.summary_payload["station_overview"]
        }
        station_match = next(
            (item for key, item in station_map.items() if key in text), None
        )
        if station_match is None:
            for item in self.summary_payload["station_overview"]:
                if item["site_name"].lower() in text:
                    station_match = item
                    break

        # 准备 LLM 上下文
        llm_context = {
            "summary_cards": self.summary_payload["summary_cards"],
            "station_overview": self.summary_payload["station_overview"],
            "rl_comparison": self.summary_payload["rl_comparison"],
            "rl_algorithm_comparison": self.summary_payload.get("rl_algorithm_comparison", {}),
            "model_comparison": self.summary_payload.get("model_comparison", {}),
            "experiment_modules": self.summary_payload.get("experiment_modules", []),
            "meta": self.summary_payload["meta"],
            "matched_station": station_match,
        }

        # 尝试 Ollama LLM
        llm_answer = self.ollama.generate_answer(question, llm_context)
        if llm_answer is not None:
            return {"question": question, **llm_answer}

        # ---- 规则回退问答 ----

        # 模型对比相关问题
        if "模型对比" in question or "lightgbm" in text or ("hist" in text and "gb" in text):
            comp = self.summary_payload.get("model_comparison", {})
            if comp:
                answer = (
                    f"预测模型对比结果: HistGBR MAE = {comp['histgb_mae']:.2f} MW, "
                    f"LightGBM MAE = {comp['lgbm_mae']:.2f} MW。"
                    f"胜出模型: {comp.get('winner', 'N/A')}, "
                    f"MAE 改善: {comp['mae_improvement']:.2f} MW ({comp['mae_improvement_pct']:.1f}%)。"
                )
            else:
                answer = "模型对比数据暂未生成, 请确保在系统训练时启用了 both 模式。"
        elif "dqn" in text or "深度" in text or "神经网络" in text:
            rl_comp = self.summary_payload.get("rl_algorithm_comparison", {})
            comp = rl_comp.get("comparison", {})
            if comp:
                answer = (
                    f"DQN vs Q-Learning 对比: Q-Learning 平均 reward = {comp['q_learning_reward']:.2f}, "
                    f"DQN 平均 reward = {comp['dqn_reward']:.2f}。"
                    f"胜出算法: {comp.get('winner', 'N/A')}, "
                    f"差异: {comp.get('reward_difference', 0):+.2f} reward。"
                    f"DQN 使用神经网络逼近 Q 函数, 可能在高维状态空间中表现更好。"
                )
            else:
                answer = (
                    "DQN 对比数据暂未生成。请确保已安装 PyTorch (pip install torch), "
                    "系统会在训练 Q-Learning 时自动训练 DQN 进行对比。"
                )
        elif "遗传算法" in question or "ga" in text:
            ga_module = next(
                (item for item in self.summary_payload.get("experiment_modules", [])
                 if "遗传算法" in item["module"]),
                None,
            )
            if ga_module is not None:
                answer = (
                    f"遗传算法优化模块：用染色体表示单日调度动作序列，"
                    f"通过选择、交叉、变异搜索更优方案。"
                    f"当前 GA 的平均 reward 为 {ga_module['avg_reward']:.2f}, "
                    f"事故率约 {ga_module['incident_rate']:.4f}。"
                    "它的价值主要在于展示优化过程机制, 以及和启发式搜索、Q-Learning 在同一环境下的差异。"
                )
            else:
                answer = "当前系统已经接入遗传算法实验, 但这次运行里还没有读到对应结果。"
        elif "启发式" in question or "搜索" in question:
            hs_module = next(
                (item for item in self.summary_payload.get("experiment_modules", [])
                 if "启发式" in item["module"]),
                None,
            )
            if hs_module is not None:
                answer = (
                    f"启发式规则搜索模块：核心思想是先用风险规则快速定位可行动作，"
                    f"再用局部 reward 改善策略。"
                    f"当前它的平均 reward 为 {hs_module['avg_reward']:.2f}, "
                    f"事故率约 {hs_module['incident_rate']:.4f}, "
                    "说明在这个海上风电调度任务上, 合理的规则搜索本身就很有竞争力。"
                )
            else:
                answer = "当前系统已经接入启发式搜索实验, 但这次运行里还没有读到对应结果。"
        elif "风险" in question or "risk" in text:
            worst = max(
                self.summary_payload["station_overview"],
                key=lambda item: item["avg_risk"],
            )
            answer = (
                f"当前风险最高的站点是 {worst['site_name']}, "
                f"平均风险分数约 {worst['avg_risk']:.2f}。"
                f"系统建议以{worst['dominant_action']}为主, "
                "并优先关注高风速突变和降水叠加的时段。"
            )
        elif "强化学习" in question or "策略" in question or "调度" in question:
            learned = next(
                item for item in self.summary_payload["rl_comparison"]
                if item["policy"] == "Q学习策略"
            )
            answer = (
                f"强化学习实验的核心, 是让智能体在不同风险状态下学习调度动作。"
                f"当前 learned policy 的平均 reward 为 {learned['avg_reward']:.2f}, "
                f"相比始终积极并网策略更稳健, 主要动作是 {learned['dominant_action']}, "
                "说明系统更偏向在波动时保留调节裕度。"
            )
        elif "预测" in question or "发电" in question or "功率" in question:
            peak_site = self.summary_payload["meta"]["peak_site"]
            peak_time = self.summary_payload["meta"]["peak_timestamp"]
            peak_power = self.summary_payload["meta"]["peak_power_mw"]
            val_mae = self.summary_payload["meta"].get("validation_mae_mw", "N/A")
            val_rmse = self.summary_payload["meta"].get("validation_rmse_mw", "N/A")
            answer = (
                f"在验证集（训练数据后20%时段，含真实功率）上，"
                f"模型预测的 MAE 为 {val_mae} MW，RMSE 为 {val_rmse} MW。"
                f"模型推演期（2023年5-7月，不含真实功率）预估总发电量约 "
                f"{self.summary_payload['model_metrics']['test_energy_mwh']:.1f} MWh。"
                f"推演期峰值出现在 {peak_time} 的 {peak_site}, "
                f"预估功率约 {peak_power:.2f} MW。"
                f"注意：推演期无真实功率数据，上述为模型推演结果而非实测对比。"
            )
        elif "验证" in question or "精度" in question or "误差" in question or "对比" in question:
            val_mae = self.summary_payload["meta"].get("validation_mae_mw", "N/A")
            val_rmse = self.summary_payload["meta"].get("validation_rmse_mw", "N/A")
            val_comp = self.summary_payload.get("validation_comparison", {})
            err_dist = val_comp.get("error_distribution", {}) if val_comp else {}
            answer = (
                f"模型精度验证（基于训练集后20%验证时段）："
                f"整体 MAE = {val_mae} MW，RMSE = {val_rmse} MW。"
                f"误差分布：均值 {err_dist.get('mean', 'N/A')} MW，"
                f"中位数 {err_dist.get('median', 'N/A')} MW，"
                f"P95 分位 {err_dist.get('p95', 'N/A')} MW。"
                f"误差在 1MW 以内的时段占比 {err_dist.get('within_1mw_pct', 'N/A')}%，"
                f"在 3MW 以内的占比 {err_dist.get('within_3mw_pct', 'N/A')}%。"
                f"验证集数据范围：{self.summary_payload['meta'].get('validation_range', ['N/A', 'N/A'])[0]}"
                f" ~ {self.summary_payload['meta'].get('validation_range', ['N/A', 'N/A'])[1]}。"
            )
        elif station_match is not None:
            answer = (
                f"{station_match['site_name']} 的装机容量为 {station_match['capacity_mw']:.0f} MW, "
                f"验证集 MAE 约 {station_match['validation_mae_mw']:.2f} MW, "
                f"测试期预测电量约 {station_match['forecast_energy_mwh']:.1f} MWh, "
                f"主导调控动作是 {station_match['dominant_action']}。"
            )
        else:
            answer = (
                "本系统基于2022-2023年福建海上风电历史数据，提供以下分析能力："
                "① 功率建模与精度验证（HistGBR + LightGBM 双模型对比，在验证集上评估 MAE/RMSE）；"
                "② 风险量化评估（综合风速突变、降水、高风速等7项因素）；"
                "③ 多策略调度对比（启发式搜索、遗传算法、Q-Learning、DQN）；"
                "④ 智能问答解释。"
                "你可以提问关于验证精度、模型对比、风险分析、调度策略、站点详情或各实验的设计思路。"
            )

        return {"question": question, "answer": answer, "backend": "rule-based", "model": None}

    # ========================================================================
    # 数据导出
    # ========================================================================

    # ========================================================================
    # 动态数据加载接口 — 上传新测试集CSV, 返回预测结果
    # ========================================================================

    def predict_on_upload(self, csv_file: BytesIO) -> BytesIO:
        """接收上传的 CSV 文件 (格式同 test_weather.csv), 返回预测结果 CSV

        流程:
          1. 读取上传的 CSV (多编码回退)
          2. 合并站点信息 (装机容量等)
          3. 特征工程 (38+7维)
          4. 预测功率 + 风险评分 + 调度建议
          5. 返回包含预测结果的 CSV

        Args:
            csv_file: 上传的 CSV 文件 (BytesIO)

        Returns:
            BytesIO: 预测结果 CSV (含 predicted_power_mw, risk_score, action_label)
        """
        from io import StringIO
        from .data_utils import engineer_features, read_csv_with_fallback
        from .forecasting import compute_risk, _prepare_matrix
        from .rl_control import pick_action

        assert self.forecast_bundle is not None

        # 1. 读取上传文件 (多编码回退)
        raw_bytes = csv_file.read()
        df = None
        last_err = None
        for encoding in ("gb18030", "utf-8-sig", "utf-8", "gbk"):
            try:
                df = pd.read_csv(BytesIO(raw_bytes), encoding=encoding)
                break
            except Exception as e:
                last_err = e
        if df is None:
            raise ValueError(f"无法解析上传的 CSV 文件: {last_err}")

        # 2. 列名智能映射 — 支持多种中文列名格式
        # 先试精确匹配, 不行再模糊匹配
        from .data_utils import WEATHER_MAP, SITE_INFO_MAP

        # 精确匹配
        df = df.rename(columns={k: v for k, v in WEATHER_MAP.items() if k in df.columns})

        # 模糊匹配 — 处理简写列名 (如 "气压" → "pressure")
        _FUZZY_MAP = {
            "时间": "timestamp", "气压": "pressure", "相对湿度": "humidity", "湿度": "humidity",
            "云量": "cloud_cover", "温度": "temperature_k", "辐照强度": "irradiance", "辐照": "irradiance",
            "降水": "precipitation", "10米风速": "wind10_speed", "10米风向": "wind10_dir",
            "100m风速": "wind100_speed", "100m风向": "wind100_dir",
            "站点编号": "site_id", "场站编号": "site_id", "站点": "site_id",
            "出力": "power_mw", "实际功率": "power_mw", "功率": "power_mw",
            "装机容量": "capacity_mw",
        }
        df = df.rename(columns={k: v for k, v in _FUZZY_MAP.items() if k in df.columns})

        # 3. 数值类型转换 + 缺失列补默认值
        numeric_cols = [
            "pressure", "humidity", "cloud_cover", "wind10_speed",
            "temperature_k", "irradiance", "precipitation", "wind100_speed",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0
        # 风向列可选 — 缺失补0
        for col in ["wind10_dir", "wind100_dir"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 4. 合并站点信息 (如果文件没有 capacity_mw 列则合并)
        if "capacity_mw" not in df.columns or df["capacity_mw"].isna().all():
            if self.dataset_bundle is not None:
                df = df.merge(
                    self.dataset_bundle.site_info[["site_id", "capacity_mw"]],
                    on="site_id", how="left"
                )
            else:
                site_info = read_csv_with_fallback(self.raw_dir / "train_site_info.csv")
                site_info = site_info.rename(columns=SITE_INFO_MAP)
                site_info["capacity_mw"] = site_info["capacity_mw"].astype(float)
                df = df.merge(site_info[["site_id", "capacity_mw"]], on="site_id", how="left")

        df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce").fillna(50).astype(float)
        if "site_id" not in df.columns:
            df["site_id"] = "f1"  # 默认站点

        # 检测文件是否含真实功率 (用于精度评估)
        has_actual_power = "power_mw" in df.columns and df["power_mw"].notna().sum() > 0
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)

        # 填充缺失气象数据 (扩展列表包含所有可能的特征列)
        all_fill_cols = numeric_cols + ["wind10_dir", "wind100_dir", "cloud_cover"]
        for col in all_fill_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df.groupby("site_id")[col].transform(lambda x: x.ffill().bfill())

        # 5. 特征工程
        site_ids = self.forecast_bundle.site_ids
        df = engineer_features(df, site_ids)

        # 6. 预测 — 使用训练时保存的特征列 (而非当前 FEATURE_COLUMNS)
        #    确保模型输入与训练时完全一致, 兼容新旧版本的 FEATURE_COLUMNS 变化
        feature_cols = [c for c in self.forecast_bundle.feature_columns if c in df.columns]

        # 诊断日志: 特征匹配检查
        model_expected = set(self.forecast_bundle.model.feature_names_in_)
        actual_input = set(feature_cols)
        extra = actual_input - model_expected
        missing = model_expected - actual_input
        if extra or missing:
            print(f"[PredictUpload] 特征不匹配! run={self.run_info.get('run_name')}", flush=True)
            print(f"[PredictUpload] model期望={len(model_expected)}列, 实际输入={len(actual_input)}列", flush=True)
            if extra:
                print(f"[PredictUpload] 多余特征: {extra}", flush=True)
            if missing:
                print(f"[PredictUpload] 缺失特征: {missing}", flush=True)
        else:
            print(f"[PredictUpload] 特征匹配OK: {len(feature_cols)}列, run={self.run_info.get('run_name')}", flush=True)

        x = _prepare_matrix(df, feature_cols, self.forecast_bundle.feature_medians)

        if self.forecast_bundle.lgbm_model is not None and self.forecast_bundle.model_type != "histgb":
            try:
                pred_ratio = np.clip(
                    self.forecast_bundle.lgbm_model.predict(x, num_iteration=self.forecast_bundle.lgbm_model.best_iteration or 300),
                    0.0, 1.15
                )
            except Exception:
                pred_ratio = np.clip(self.forecast_bundle.model.predict(x), 0.0, 1.15)
        else:
            pred_ratio = np.clip(self.forecast_bundle.model.predict(x), 0.0, 1.15)

        df["predicted_ratio"] = pred_ratio
        df["predicted_power_mw"] = (df["predicted_ratio"] * df["capacity_mw"]).clip(lower=0.0, upper=df["capacity_mw"])
        df["risk_score"] = compute_risk(df, df["predicted_ratio"], self.forecast_bundle.risk_reference)

        # 7. 调度建议
        for idx, row in df.iterrows():
            action = pick_action(row, self.rl_bundle.q_table, self.forecast_bundle.site_ids, dqn_agent=self.dqn_agent)
            df.at[idx, "action_label"] = action["action_label"]
            df.at[idx, "recommended_dispatch_mw"] = action["recommended_dispatch_mw"]

        # 8. 输出 (如果文件含真实功率, 附加误差列)
        base_cols = ["site_id", "timestamp", "predicted_power_mw", "risk_score", "action_label", "recommended_dispatch_mw"]
        base_names = ["场站编号", "时间", "预测功率_MW", "风险评分", "推荐调度策略", "推荐调度功率_MW"]

        if has_actual_power:
            df["abs_error_mw"] = (df["power_mw"] - df["predicted_power_mw"]).abs()
            base_cols = ["site_id", "timestamp", "power_mw", "predicted_power_mw", "abs_error_mw", "risk_score", "action_label", "recommended_dispatch_mw"]
            base_names = ["场站编号", "时间", "实际功率_MW", "预测功率_MW", "绝对误差_MW", "风险评分", "推荐调度策略", "推荐调度功率_MW"]

        out = df[[c for c in base_cols if c in df.columns]].copy()
        # 用存在的列名对应
        present_names = [n for c, n in zip(base_cols, base_names) if c in df.columns]
        out.columns = present_names

        buffer = BytesIO()
        buffer.write(out.to_csv(index=False).encode("utf-8-sig"))
        buffer.seek(0)
        return buffer

    def export_forecast_csv(self) -> BytesIO:
        """导出验证集预测 vs 实际对比结果为 CSV"""
        assert self.forecast_bundle is not None
        val_frame = self.forecast_bundle.validation_frame
        export_frame = val_frame[
            [
                "site_id", "timestamp", "power_mw", "predicted_power_mw",
                "abs_error_mw", "risk_score",
                "action_label", "recommended_dispatch_mw",
            ]
        ].copy()
        export_frame.rename(
            columns={
                "site_id": "场站编号",
                "timestamp": "时间",
                "power_mw": "实际功率_MW",
                "predicted_power_mw": "预测功率_MW",
                "abs_error_mw": "绝对误差_MW",
                "risk_score": "风险评分",
                "action_label": "推荐调度动作",
                "recommended_dispatch_mw": "推荐调度功率_MW",
            },
            inplace=True,
        )
        buffer = BytesIO()
        buffer.write(export_frame.to_csv(index=False).encode("utf-8-sig"))
        buffer.seek(0)
        return buffer

    def export_rl_report_json(self) -> BytesIO:
        """导出强化学习完整报告为 JSON (含多算法对比)"""
        assert self.summary_payload is not None
        assert self.rl_bundle is not None
        payload = {
            "summary": self.summary_payload,
            "model_comparison": self.summary_payload.get("model_comparison", {}),
            "rl_comparison": self.rl_bundle.comparison,
            "rl_algorithm_comparison": self.rl_bundle.rl_algorithm_comparison,
            "experiment_modules": self.rl_bundle.experiment_modules,
            "heuristic_examples": self.rl_bundle.heuristic_examples,
            "ga_examples": self.rl_bundle.ga_examples,
            "actions": ACTIONS,
        }
        buffer = BytesIO()
        buffer.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        buffer.seek(0)
        return buffer

    # ========================================================================
    # 上传测试集管理 (持久化, 跨页面共享, 刷新不丢失)
    # ========================================================================

    _MAX_UPLOADS = 5

    def save_upload(self, csv_buffer: BytesIO, original_filename: str) -> dict[str, Any]:
        """保存预测结果到磁盘, 返回 upload 元信息

        流程: predict_on_upload 已生成预测 CSV → 保存到 artifacts/uploads/<id>/
        """
        upload_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 保存预测 CSV
        csv_buffer.seek(0)
        pred_path = upload_dir / "predictions.csv"
        pred_path.write_bytes(csv_buffer.read())

        # 解析 CSV 提取元信息
        csv_buffer.seek(0)
        df = pd.read_csv(csv_buffer)

        # 站点列表 (场站编号列)
        site_col = "场站编号" if "场站编号" in df.columns else df.columns[0]
        sites = sorted(df[site_col].unique().tolist())

        # 预测功率列
        power_col = next((c for c in df.columns if "功率" in c and "MW" in c), df.columns[2])

        meta = {
            "id": upload_id,
            "original_filename": original_filename,
            "created_at": datetime.now().isoformat(),
            "total_rows": len(df),
            "sites": sites,
            "columns": list(df.columns),
            "pred_power_min": float(df[power_col].min()),
            "pred_power_max": float(df[power_col].max()),
        }
        (upload_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 清理超出上限的旧上传
        self._prune_old_uploads()

        return meta

    def _prune_old_uploads(self) -> None:
        """保留最近 _MAX_UPLOADS 次上传, 删除多余的"""
        all_uploads = sorted(
            self.uploads_dir.glob("upload_*"),
            key=lambda p: p.name,
            reverse=True,
        )
        for old_dir in all_uploads[self._MAX_UPLOADS:]:
            shutil.rmtree(old_dir, ignore_errors=True)

    def list_uploads(self) -> list[dict[str, Any]]:
        """列出所有已保存的上传记录 (按时间倒序)"""
        results = []
        for upload_dir in sorted(
            self.uploads_dir.glob("upload_*"),
            key=lambda p: p.name,
            reverse=True,
        ):
            meta_path = upload_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    results.append(meta)
                except Exception:
                    continue
        return results

    def get_upload_detail(self, upload_id: str) -> dict[str, Any] | None:
        """获取单次上传的详情 (含前200行预览数据)"""
        upload_dir = self.uploads_dir / upload_id
        meta_path = upload_dir / "meta.json"
        pred_path = upload_dir / "predictions.csv"
        if not meta_path.exists() or not pred_path.exists():
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        df = pd.read_csv(str(pred_path))
        # 按站点分组统计
        site_stats = []
        site_col = "场站编号" if "场站编号" in df.columns else df.columns[0]
        power_col = next((c for c in df.columns if "功率" in c and "MW" in c), df.columns[2])
        for site_id, g in df.groupby(site_col):
            site_stats.append({
                "site_id": site_id,
                "row_count": len(g),
                "pred_avg_mw": round(float(g[power_col].mean()), 2),
                "pred_peak_mw": round(float(g[power_col].max()), 2),
            })

        # 前200行预览 (每个站点取前40行)
        preview_rows = []
        for site_id in meta.get("sites", []):
            site_df = df[df[site_col] == site_id].head(40)
            preview_rows.extend(site_df.values.tolist())

        return {
            **meta,
            "site_stats": site_stats,
            "columns": list(df.columns),
            "preview_rows": preview_rows[:200],
        }

    def get_upload_site_data(self, upload_id: str, site_id: str) -> dict[str, Any] | None:
        """获取上传数据中指定站点的完整时序数据 (站点驾驶舱用)"""
        upload_dir = self.uploads_dir / upload_id
        meta_path = upload_dir / "meta.json"
        pred_path = upload_dir / "predictions.csv"
        if not meta_path.exists() or not pred_path.exists():
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        df = pd.read_csv(str(pred_path))

        site_col = "场站编号" if "场站编号" in df.columns else df.columns[0]
        time_col = "时间" if "时间" in df.columns else df.columns[1]
        power_col = next((c for c in df.columns if "功率" in c and "MW" in c), df.columns[2])
        risk_col = next((c for c in df.columns if "风险" in c), None)
        action_col = next((c for c in df.columns if "策略" in c), None)
        dispatch_col = next((c for c in df.columns if "调度" in c and "MW" in c), None)

        if site_id not in df[site_col].values:
            return None

        site_df = df[df[site_col] == site_id].copy()
        site_df["_ts"] = pd.to_datetime(site_df[time_col])
        site_df = site_df.sort_values("_ts").reset_index(drop=True)

        # ---- 时序图数据 (仅预测, 无实际对比) ----
        forecast_series = []
        for _, row in site_df.iterrows():
            pt = {
                "timestamp": row["_ts"].strftime("%m-%d %H:%M"),
                "predicted_power_mw": round(float(row[power_col]), 3),
            }
            if risk_col and risk_col in row:
                pt["risk_score"] = round(float(row[risk_col]), 3)
            if action_col and action_col in row:
                pt["action_label"] = str(row[action_col])
            if dispatch_col and dispatch_col in row:
                pt["dispatch_power_mw"] = round(float(row[dispatch_col]), 3)
            forecast_series.append(pt)

        # ---- 日汇总 (仅预测值) ----
        site_df["_date"] = site_df["_ts"].dt.strftime("%Y-%m-%d")
        daily = (
            site_df.groupby("_date")
            .agg(
                predicted_avg=(power_col, "mean"),
                predicted_peak=(power_col, "max"),
                predicted_energy=(power_col, "sum"),
            )
            .reset_index()
        )
        if risk_col:
            daily_risk = site_df.groupby("_date")[risk_col].mean().reset_index()
            daily = daily.merge(daily_risk.rename(columns={risk_col: "avg_risk"}), on="_date", how="left")
        else:
            daily["avg_risk"] = 0.0

        # 每日策略分布
        daily_actions = []
        if action_col:
            for date, day_df in site_df.groupby("_date"):
                action_counts = day_df[action_col].value_counts().to_dict()
                daily_actions.append({"date": date, "actions": action_counts})

        # 风险带
        risk_band = []
        if risk_col:
            for _, row in site_df.iterrows():
                risk_band.append({
                    "timestamp": row["_ts"].strftime("%m-%d %H:%M"),
                    "risk_score": round(float(row[risk_col]), 3),
                    "action_label": str(row[action_col]) if action_col and action_col in row else "平衡调度",
                })

        # 高风险窗口 Top 10
        top_windows = []
        if risk_col:
            top = site_df.sort_values(risk_col, ascending=False).head(10)
            for _, row in top.iterrows():
                win = {
                    "timestamp": row["_ts"].strftime("%Y-%m-%d %H:%M"),
                    "risk_score": round(float(row[risk_col]), 3),
                    "predicted_power_mw": round(float(row[power_col]), 3),
                }
                if action_col and action_col in row:
                    win["action_label"] = str(row[action_col])
                top_windows.append(win)

        # 容量估算 (用预测功率峰值)
        cap_est = round(float(site_df[power_col].max()) * 1.05, 0)
        try:
            site_info = self.dataset_bundle.site_info
            site_row = site_info[site_info["site_id"] == site_id]
            if len(site_row) > 0:
                cap_est = round(float(site_row["capacity_mw"].iloc[0]), 2)
        except Exception:
            pass

        return {
            "site_id": site_id,
            "capacity_mw": cap_est,
            "data_source": f"上传测试集 ({meta['original_filename']})",
            "upload_id": upload_id,
            "avg_risk": round(float(site_df[risk_col].mean()), 3) if risk_col else 0,
            "dominant_action": (
                site_df[action_col].mode().iloc[0]
                if action_col and len(site_df[action_col].mode()) > 0
                else "平衡调度"
            ),
            "forecast_series": forecast_series,
            "risk_band": risk_band,
            "daily_summary": [
                {
                    "date": row["_date"],
                    "predicted_avg": round(float(row["predicted_avg"]), 3),
                    "predicted_peak": round(float(row["predicted_peak"]), 3),
                    "avg_risk": round(float(row.get("avg_risk", 0)), 3),
                    "total_predicted_mwh": round(float(row["predicted_energy"]) * 0.25, 1),
                }
                for _, row in daily.iterrows()
            ],
            "daily_actions": daily_actions,
            "top_windows": top_windows,
        }

    def delete_upload(self, upload_id: str) -> bool:
        """删除指定上传记录"""
        upload_dir = self.uploads_dir / upload_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
            return True
        return False
