"""
海上风电功率预测模块 (Offshore Wind Power Forecasting Module)
==============================================================
支持两种梯度提升模型进行风电功率预测对比:
  - HistGradientBoostingRegressor (sklearn, 基于直方图的梯度提升)
  - LightGBM (微软开源的梯度提升框架, 训练更快、精度更高)

可通过 model_type 参数切换: "histgb" / "lgbm" / "both"(对比实验)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 尝试导入 LightGBM, 如果未安装则标记为不可用
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore[assignment]
    _LGBM_AVAILABLE = False

from .data_utils import DatasetBundle, engineer_features, energy_from_power, make_validation_mask


# ============================================================================
# 特征列定义 —— 38 维气象+时间+工程特征
# ============================================================================
FEATURE_COLUMNS = [
    # ---- 原始气象特征 ----
    "pressure",           # 气压 (Pa)
    "humidity",           # 相对湿度 (%)
    "cloud_cover",        # 云量
    "wind10_speed",       # 10米风速 (m/s)
    "wind10_dir",         # 10米风向 (°)
    "temperature_k",      # 温度 (K)
    "irradiance",         # 辐照强度 (J/m²)
    "precipitation",      # 降水 (m)
    "wind100_speed",      # 100米风速 (m/s) — 风机轮毂高度的主力风速
    "wind100_dir",        # 100米风向 (°)
    "capacity_mw",        # 装机容量 (MW)
    # ---- 时间特征 ----
    "hour",               # 小时 (0-23)
    "minute_slot",        # 15分钟槽位 (0-3)
    "weekday",            # 星期几 (0=周一)
    "month",              # 月份
    "dayofyear",          # 年中的第几天
    "hour_sin",           # 小时的 sin 循环编码
    "hour_cos",           # 小时的 cos 循环编码
    "doy_sin",            # 年天的 sin 循环编码
    "doy_cos",            # 年天的 cos 循环编码
    # ---- 风向量分解 (避免角度周期性跳变) ----
    "wind10_u",           # 10米风速的 U 分量 (东-西)
    "wind10_v",           # 10米风速的 V 分量 (北-南)
    "wind100_u",          # 100米风速的 U 分量
    "wind100_v",          # 100米风速的 V 分量
    "wind_shear",         # 风切变 = 100m风速 - 10m风速
    # ---- 物理计算特征 ----
    "temperature_c",      # 摄氏温度
    "air_density",        # 空气密度 (通过理想气体状态方程计算)
    # ---- 滚动统计特征 (捕捉风速变化趋势) ----
    "wind100_speed_roll4",   # 100m风速 4步 (1小时) 滚动均值
    "wind100_speed_roll12",  # 100m风速 12步 (3小时) 滚动均值
    "wind10_speed_roll4",    # 10m风速 4步滚动均值
    "humidity_roll4",        # 湿度 4步滚动均值
    "cloud_cover_roll4",     # 云量 4步滚动均值
    "precip_roll8",          # 降水 8步 (2小时) 滚动均值
    # ---- 差分特征 (一阶变化率, 捕捉突变) ----
    "wind_ramp",             # 风速一阶差分
    "humidity_ramp",         # 湿度一阶差分
    "wind_ramp_abs",         # 风速变化的绝对值
    # ---- 滞后功率特征 (核心精度提升点) ----
    "power_ratio_lag1",      # 15分钟前功率比
    "power_ratio_lag4",      # 1小时前功率比
    "power_ratio_lag8",      # 2小时前功率比
    "power_ratio_lag24",     # 6小时前功率比
    # ---- 物理扩展特征 ----
    "wind_power_density",    # 风功率密度 (0.5ρv³)
    "turbulence_4",          # 1小时湍流强度
    "turbulence_12",         # 3小时湍流强度
]

# 模型类型常量
MODEL_TYPE_HISTGB = "histgb"   # sklearn 直方图梯度提升
MODEL_TYPE_LGBM = "lgbm"       # LightGBM
MODEL_TYPE_BOTH = "both"       # 两者都跑, 用于对比实验

# 训练数据采样上限 (控制训练时间)
MAX_TRAIN_ROWS = 150000
MAX_FINAL_ROWS = 200000


@dataclass
class ForecastBundle:
    """预测系统完整结果包

    包含训练好的模型、预测结果、验证指标、风险参考等。
    当 model_type="both" 时, lgbm_metrics 和 model_comparison 会有值。
    """
    # ---- HistGBR 结果 (始终存在, 向后兼容) ----
    model: HistGradientBoostingRegressor
    feature_columns: list[str]
    feature_medians: dict[str, float]
    site_ids: list[str]
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_frame: pd.DataFrame
    metrics: dict[str, Any]               # 总体+逐站点的 HistGBR 指标
    risk_reference: dict[str, Any]

    # ---- LightGBM 结果 (model_type="lgbm" 或 "both" 时有值) ----
    lgbm_model: Any = None                # lightgbm.Booster 或 None
    lgbm_metrics: dict[str, Any] = field(default_factory=dict)

    # ---- 模型对比 (model_type="both" 时有值) ----
    model_comparison: dict[str, Any] = field(default_factory=dict)

    # ---- 元信息 ----
    model_type: str = MODEL_TYPE_HISTGB   # 本次训练使用的模型类型


# ============================================================================
# 特征矩阵准备
# ============================================================================

def _prepare_matrix(frame: pd.DataFrame, feature_columns: list[str], medians: dict[str, float]) -> pd.DataFrame:
    """准备特征矩阵, 用中位数填充缺失值"""
    available = [c for c in feature_columns if c in frame.columns]
    x = frame[available].copy()
    for column, value in medians.items():
        if column in x.columns:
            x[column] = x[column].fillna(value)
    return x


def _add_site_columns(frame: pd.DataFrame, site_ids: list[str]) -> list[str]:
    """将站点独热编码列名追加到特征列表，并去掉数据中不存在的列（如滞后特征在测试集中不存在）"""
    base = FEATURE_COLUMNS + [f"site_{site_id}" for site_id in site_ids]
    return [c for c in base if c in frame.columns]


# ============================================================================
# 模型拟合函数
# ============================================================================

def _fit_model_histgb(x_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingRegressor:
    """训练 HistGradientBoostingRegressor 模型

    sklearn 基于直方图的梯度提升树, 原生支持缺失值处理,
    不需要手动填充 NaN, 对大数据集速度优于传统 GBDT。
    """
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,      # 学习率
        max_depth=8,             # 树的最大深度
        max_iter=200,            # 提升迭代次数
        min_samples_leaf=20,     # 叶子节点最小样本数
        l2_regularization=0.01,  # L2 正则化系数
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def _fit_model_lgb(x_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """训练 LightGBM 模型

    LightGBM 使用基于梯度的单边采样(GOSS)和互斥特征捆绑(EFB),
    通常比 HistGBR 更快且精度更高。

    Returns:
        lightgbm.Booster: 训练好的 LightGBM 模型
    """
    if not _LGBM_AVAILABLE:
        raise ImportError("LightGBM 未安装, 请执行: pip install lightgbm")

    # 创建 LightGBM Dataset (内存效率更高)
    train_data = lgb.Dataset(x_train, label=y_train)

    # LightGBM 核心参数
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "max_depth": 10,
        "num_leaves": 128,
        "min_data_in_leaf": 15,
        "lambda_l2": 0.005,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "num_iterations": 300,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=None,                    # 不需要验证集 (与 HistGBR 保持一致)
    )
    return model


# ============================================================================
# 数据采样
# ============================================================================

def _sample_frame(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    """分层采样: 按站点均匀采样, 控制训练数据总量"""
    if len(frame) <= max_rows:
        return frame
    # 在每个站点内均匀采样
    sampled = (
        frame.groupby("site_id", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_rows // max(frame["site_id"].nunique(), 1)),
                random_state=random_state,
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return sampled


# ============================================================================
# 风险评分系统
# ============================================================================

def _build_risk_reference(validation_frame: pd.DataFrame, train_frame: pd.DataFrame) -> dict[str, Any]:
    """构建风险评分参考基准

    风险评分综合考虑:
      - 预测误差 (站点级历史偏差)
      - 风速突变 (ramp) 的极端值 (P95)
      - 降水极端值 (P95)
      - 100m 风速极端值 (P95)
      - 空气密度偏差 (偏离中位数的程度)
    """
    # 各站点的平均预测误差比
    site_error = validation_frame.groupby("site_id")["abs_error_ratio"].mean().to_dict()
    site_error_values = np.array(list(site_error.values()), dtype=float)
    site_min = float(site_error_values.min()) if len(site_error_values) else 0.0
    site_max = float(site_error_values.max()) if len(site_error_values) else 1.0
    scale = max(site_max - site_min, 1e-6)

    return {
        "site_error": site_error,
        "site_error_norm": {key: float((value - site_min) / scale) for key, value in site_error.items()},
        "wind_ramp_p95": float(train_frame["wind_ramp_abs"].quantile(0.95)),
        "precip_p95": float(train_frame["precipitation"].quantile(0.95)),
        "wind100_p95": float(train_frame["wind100_speed"].quantile(0.95)),
        "density_median": float(train_frame["air_density"].median()),
        "density_dev_p90": float((train_frame["air_density"] - train_frame["air_density"].median()).abs().quantile(0.90)),
    }


def compute_risk(frame: pd.DataFrame, predicted_ratio: pd.Series, risk_reference: dict[str, Any]) -> pd.Series:
    """计算每个时间片的综合风险评分 (0-1)

    风险评分 = 0.28×风速突变 + 0.18×降水 + 0.14×高风速 + 0.12×密度异常
             + 0.08×云量 + 0.12×站点历史误差 + 0.08×高出力压力

    分数越高 → 调度风险越大 → 建议保守策略
    """
    def clip_scale(series: pd.Series, high: float) -> pd.Series:
        """将序列缩放到 [0, 1], 以 P95 为上限"""
        denom = high if high > 1e-6 else 1.0
        return (series / denom).clip(lower=0.0, upper=1.0)

    # 各项风险分量归一化
    ramp_norm = clip_scale(frame["wind_ramp_abs"], risk_reference["wind_ramp_p95"])
    precip_norm = clip_scale(frame["precipitation"], risk_reference["precip_p95"])
    wind_norm = clip_scale(frame["wind100_speed"], risk_reference["wind100_p95"])
    density_dev = (frame["air_density"] - risk_reference["density_median"]).abs()
    density_norm = clip_scale(density_dev, risk_reference["density_dev_p90"])
    site_norm = frame["site_id"].map(risk_reference["site_error_norm"]).fillna(
        np.mean(list(risk_reference["site_error_norm"].values()) or [0.0])
    )
    # 高出力压力: 预测功率比超过 0.55 时的压力
    output_pressure = ((predicted_ratio - 0.55) / 0.45).clip(lower=0.0, upper=1.0)

    # 加权合成综合风险评分
    score = (
        0.28 * ramp_norm
        + 0.18 * precip_norm
        + 0.14 * wind_norm
        + 0.12 * density_norm
        + 0.08 * frame["cloud_cover"].clip(lower=0.0, upper=1.0)
        + 0.12 * site_norm
        + 0.08 * output_pressure
    )
    return score.clip(lower=0.0, upper=1.0)


# ============================================================================
# 评估指标
# ============================================================================

def _metric_dict(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """计算回归评估指标: MAE, RMSE, MAPE, R²"""
    from sklearn.metrics import r2_score
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    safe_actual = actual.replace(0.0, np.nan)
    mape = float((np.abs((actual - predicted) / safe_actual)).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100.0)
    r2 = float(r2_score(actual, predicted))
    return {"mae": mae, "rmse": rmse, "mape": mape if np.isfinite(mape) else 0.0, "r2": r2}


# ============================================================================
# 训练流程核心
# ============================================================================

def _train_single_model(
    bundle: DatasetBundle,
    model_type: str,
) -> ForecastBundle:
    """使用指定模型类型训练风电功率预测系统

    Args:
        bundle: 数据集 (训练集+测试集+站点信息)
        model_type: "histgb" 或 "lgbm"

    Returns:
        ForecastBundle: 包含模型、预测结果、指标和风险参考
    """
    if model_type == MODEL_TYPE_LGBM and not _LGBM_AVAILABLE:
        raise ImportError("LightGBM 未安装, 无法使用 lgbm 模型。请执行: pip install lightgbm")

    site_ids = bundle.site_info["site_id"].tolist()

    # ---- 1. 特征工程 ----
    train_frame = engineer_features(bundle.train, site_ids)
    test_frame = engineer_features(bundle.test, site_ids)

    feature_columns = _add_site_columns(train_frame, site_ids)

    # ---- 2. 时序验证集划分 (80%训练 / 20%验证, 按站点内时间顺序) ----
    val_mask = make_validation_mask(train_frame, val_fraction=0.2)
    train_split = train_frame.loc[~val_mask].copy()
    validation_frame = train_frame.loc[val_mask].copy()

    # ---- 3. 训练评估模型 (用小样本快速验证) ----
    medians = train_split[feature_columns].median().fillna(0.0).to_dict()
    sampled_train_split = _sample_frame(train_split, MAX_TRAIN_ROWS, random_state=42)
    x_train = _prepare_matrix(sampled_train_split, feature_columns, medians)
    x_val = _prepare_matrix(validation_frame, feature_columns, medians)
    y_train = sampled_train_split["power_ratio"]
    y_val = validation_frame["power_ratio"]

    if model_type == MODEL_TYPE_HISTGB:
        eval_model = _fit_model_histgb(x_train, y_train)
    else:
        eval_model = _fit_model_lgb(x_train, y_train)

    # 预测值裁剪到 [0, 1.15] (功率比理论上不超过 1.15)
    if model_type == MODEL_TYPE_HISTGB:
        val_pred_ratio = np.clip(eval_model.predict(x_val), 0.0, 1.15)
    else:
        # LightGBM predict 返回的是 numpy array, 直接裁剪
        val_pred_ratio = np.clip(eval_model.predict(x_val, num_iteration=eval_model.best_iteration or 300), 0.0, 1.15)

    validation_frame["predicted_ratio"] = val_pred_ratio
    validation_frame["predicted_power_mw"] = validation_frame["predicted_ratio"] * validation_frame["capacity_mw"]
    validation_frame["abs_error_ratio"] = (validation_frame["power_ratio"] - validation_frame["predicted_ratio"]).abs()
    validation_frame["abs_error_mw"] = (validation_frame["power_mw"] - validation_frame["predicted_power_mw"]).abs()

    # ---- 4. 训练最终模型 (用更多数据, 但只用测试集也有的特征) ----
    # 测试集没有 power_ratio 所以没有滞后特征, 最终模型只用测试集也有的特征
    test_feature_columns = [c for c in feature_columns if c in test_frame.columns]
    full_medians = train_frame[test_feature_columns].median().fillna(0.0).to_dict()
    sampled_full_train = _sample_frame(train_frame, MAX_FINAL_ROWS, random_state=43)

    if model_type == MODEL_TYPE_HISTGB:
        final_model = _fit_model_histgb(
            _prepare_matrix(sampled_full_train, test_feature_columns, full_medians),
            sampled_full_train["power_ratio"],
        )
        train_pred = np.clip(
            final_model.predict(_prepare_matrix(train_frame, test_feature_columns, full_medians)), 0.0, 1.15
        )
        test_pred = np.clip(
            final_model.predict(_prepare_matrix(test_frame, test_feature_columns, full_medians)), 0.0, 1.15
        )
    else:
        final_model = _fit_model_lgb(
            _prepare_matrix(sampled_full_train, test_feature_columns, full_medians),
            sampled_full_train["power_ratio"],
        )
        train_pred = np.clip(
            final_model.predict(_prepare_matrix(train_frame, test_feature_columns, full_medians),
                                num_iteration=final_model.best_iteration or 300), 0.0, 1.15
        )
        test_pred = np.clip(
            final_model.predict(_prepare_matrix(test_frame, test_feature_columns, full_medians),
                                num_iteration=final_model.best_iteration or 300), 0.0, 1.15
        )

    train_frame["predicted_ratio"] = train_pred
    train_frame["predicted_power_mw"] = train_frame["predicted_ratio"] * train_frame["capacity_mw"]

    # ---- 5. 构建风险参考基准 ----
    risk_reference = _build_risk_reference(validation_frame, train_frame)
    validation_frame["risk_score"] = compute_risk(validation_frame, validation_frame["predicted_ratio"], risk_reference)

    # ---- 6. 测试集预测 ----
    test_frame["predicted_ratio"] = test_pred
    test_frame["predicted_power_mw"] = (test_frame["predicted_ratio"] * test_frame["capacity_mw"]).clip(
        lower=0.0, upper=test_frame["capacity_mw"]
    )
    test_frame["risk_score"] = compute_risk(test_frame, test_frame["predicted_ratio"], risk_reference)

    # ---- 7. 计算评估指标 ----
    overall_metrics = _metric_dict(validation_frame["power_mw"], validation_frame["predicted_power_mw"])

    # 逐站点指标
    by_site: list[dict[str, Any]] = []
    for site_id, site_group in validation_frame.groupby("site_id"):
        metrics = _metric_dict(site_group["power_mw"], site_group["predicted_power_mw"])
        metrics["site_id"] = site_id
        metrics["capacity_mw"] = float(site_group["capacity_mw"].iloc[0])
        metrics["energy_actual_mwh"] = energy_from_power(site_group["power_mw"])
        metrics["energy_predicted_mwh"] = energy_from_power(site_group["predicted_power_mw"])
        by_site.append(metrics)

    metrics = {
        "overall": overall_metrics,
        "by_site": by_site,
        "test_energy_mwh": energy_from_power(test_frame["predicted_power_mw"]),
        "validation_energy_mwh": energy_from_power(validation_frame["power_mw"]),
    }

    return ForecastBundle(
        model=final_model if model_type == MODEL_TYPE_HISTGB else None,
        feature_columns=feature_columns,
        feature_medians=full_medians,
        site_ids=site_ids,
        validation_frame=validation_frame,
        test_frame=test_frame,
        train_frame=train_frame,
        metrics=metrics,
        risk_reference=risk_reference,
        lgbm_model=final_model if model_type == MODEL_TYPE_LGBM else None,
        model_type=model_type,
    )


def train_forecast_system(bundle: DatasetBundle, model_type: str = MODEL_TYPE_BOTH) -> ForecastBundle:
    """训练海上风电功率预测系统 (统一入口)

    支持两种模型, 可通过 model_type 参数选择:
      - "histgb": 仅使用 sklearn HistGradientBoostingRegressor (原方案)
      - "lgbm":   仅使用 LightGBM
      - "both":   两者都跑, 输出对比指标 (推荐用于对比实验)

    Args:
        bundle: 加载好的数据集
        model_type: 模型类型选择

    Returns:
        ForecastBundle: 完整预测结果包, model_type="both" 时包含对比数据
    """
    # ---- 模式1: 仅 HistGBR (向后兼容) ----
    if model_type == MODEL_TYPE_HISTGB:
        return _train_single_model(bundle, MODEL_TYPE_HISTGB)

    # ---- 模式2: 仅 LightGBM ----
    if model_type == MODEL_TYPE_LGBM:
        return _train_single_model(bundle, MODEL_TYPE_LGBM)

    # ---- 模式3: 两者对比 (对比实验模式) ----
    # 使用 HistGBR 作为主模型 (保持向后兼容)
    result = _train_single_model(bundle, MODEL_TYPE_HISTGB)

    # 如果 LightGBM 可用, 训练并对比
    if _LGBM_AVAILABLE:
        try:
            lgbm_result = _train_single_model(bundle, MODEL_TYPE_LGBM)
            result.lgbm_metrics = lgbm_result.metrics
            result.lgbm_model = lgbm_result.lgbm_model

            # 构建对比摘要
            histgb_mae = result.metrics["overall"]["mae"]
            lgbm_mae = lgbm_result.metrics["overall"]["mae"]
            result.model_comparison = {
                "histgb_mae": round(histgb_mae, 4),
                "lgbm_mae": round(lgbm_mae, 4),
                "mae_improvement": round(histgb_mae - lgbm_mae, 4),
                "mae_improvement_pct": round((histgb_mae - lgbm_mae) / max(histgb_mae, 1e-6) * 100, 2),
                "histgb_rmse": result.metrics["overall"]["rmse"],
                "lgbm_rmse": lgbm_result.metrics["overall"]["rmse"],
                "winner": "LightGBM" if lgbm_mae < histgb_mae else "HistGBR",
            }
            result.model_type = MODEL_TYPE_BOTH
        except Exception:
            # LightGBM 训练失败时回退到仅 HistGBR
            result.model_type = MODEL_TYPE_HISTGB
    else:
        result.model_type = MODEL_TYPE_HISTGB

    return result
