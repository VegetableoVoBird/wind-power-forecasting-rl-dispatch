"""
数据处理与特征工程模块 (Data Loading & Feature Engineering)
============================================================
负责:
  1. 多编码 CSV 文件读取 (gb18030 / utf-8-sig / utf-8 回退)
  2. 原始数据清洗、类型转换、缺失值填充
  3. 特征工程: 38 维特征构建 (时间编码、风向分解、滚动统计、差分等)
  4. 时序验证集划分 (按站点内时间顺序, 避免数据泄露)
  5. 发电量计算 (功率 × 时间间隔)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================================
# 列名映射 —— 中文列名 → 英文变量名
# ============================================================================

SITE_INFO_MAP = {
    "站点名称": "site_name",
    "站点编号": "site_id",
    "地区": "region",
    "装机容量(MW)": "capacity_mw",
}

WEATHER_MAP = {
    "站点编号": "site_id",
    "时间": "timestamp",
    "气压(Pa）": "pressure",
    "相对湿度（%）": "humidity",
    "云量": "cloud_cover",
    "10米风速（10m/s）": "wind10_speed",
    "10米风向（°)": "wind10_dir",
    "温度（K）": "temperature_k",
    "辐照强度（J/m2）": "irradiance",
    "降水（m）": "precipitation",
    "100m风速（100m/s）": "wind100_speed",
    "100m风向（°)": "wind100_dir",
    "出力(MW)": "power_mw",
}


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class DatasetBundle:
    """数据集容器

    train:  训练集 (含气象特征 + 实际功率)
    test:   测试集 (仅气象特征, 需要预测功率)
    site_info: 站点基本信息 (编号、名称、地区、装机容量)
    """
    train: pd.DataFrame
    test: pd.DataFrame
    site_info: pd.DataFrame


# ============================================================================
# CSV 读取 —— 多编码回退
# ============================================================================

def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """读取 CSV 文件, 自动尝试多种编码

    中文字符集常见的编码顺序: gb18030 > utf-8-sig > utf-8
    """
    last_error: Exception | None = None
    for encoding in ("gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, na_values=["<NULL>", "NULL", "null", ""])
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to read CSV: {path}") from last_error


# ============================================================================
# 数据清洗
# ============================================================================

def _coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """强制将指定列转为数值类型, 非法值变为 NaN"""
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_raw_bundle(raw_dir: Path) -> DatasetBundle:
    """加载原始 CSV 数据并完成清洗

    处理步骤:
      1. 读取三个 CSV 文件 (站点信息、训练气象+功率、测试气象)
      2. 统一列名为英文
      3. 数值类型强制转换
      4. 合并站点信息 (装机容量等)
      5. 按 site_id + timestamp 排序
      6. 删除训练集中功率缺失的行
      7. 按站点分组前向/后向填充缺失值
      8. 计算功率比 (power_ratio = power_mw / capacity_mw)

    Args:
        raw_dir: 原始数据目录, 期望包含:
          - train_site_info.csv
          - train_weather_power.csv
          - test_weather.csv

    Returns:
        DatasetBundle: 清洗后的完整数据集
    """
    # 读取原始 CSV
    site_info = read_csv_with_fallback(raw_dir / "train_site_info.csv").rename(columns=SITE_INFO_MAP)
    train = read_csv_with_fallback(raw_dir / "train_weather_power.csv").rename(columns=WEATHER_MAP)
    test = read_csv_with_fallback(raw_dir / "test_weather.csv").rename(columns=WEATHER_MAP)

    # 所有数值特征列
    numeric_columns = [
        "pressure", "humidity", "cloud_cover",
        "wind10_speed", "wind10_dir", "temperature_k",
        "irradiance", "precipitation",
        "wind100_speed", "wind100_dir", "power_mw",
    ]

    # 类型转换
    site_info["capacity_mw"] = site_info["capacity_mw"].astype(float)
    train = _coerce_numeric_columns(train, numeric_columns)
    test = _coerce_numeric_columns(test, numeric_columns)

    # 合并站点信息 (装机容量等)
    train = train.merge(site_info, on="site_id", how="left")
    test = test.merge(site_info, on="site_id", how="left")

    # 排序 + 索引重置
    for frame in (train, test):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["capacity_mw"] = frame["capacity_mw"].astype(float)
        frame.sort_values(["site_id", "timestamp"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    # 删除训练集中功率缺失的行 (无法用于训练)
    train.dropna(subset=["power_mw"], inplace=True)
    train.reset_index(drop=True, inplace=True)

    # 按站点分组进行前向填充 → 后向填充 (处理传感器短时缺失)
    fill_columns = [col for col in numeric_columns if col in train.columns or col in test.columns]
    for frame in (train, test):
        for column in fill_columns:
            if column in frame.columns:
                frame[column] = frame.groupby("site_id")[column].transform(
                    lambda series: series.ffill().bfill()
                )

    # 计算功率比 (相对装机容量的出力比例, 便于跨站点统一建模)
    train["power_ratio"] = (train["power_mw"] / train["capacity_mw"]).clip(lower=0.0, upper=1.2)
    return DatasetBundle(
        train=train,
        test=test,
        site_info=site_info.sort_values("site_id").reset_index(drop=True),
    )


# ============================================================================
# 特征工程
# ============================================================================

def _rolling_feature(
    frame: pd.DataFrame,
    source_col: str,
    window: int,
    new_col: str,
    min_periods: int = 1,
) -> None:
    """按站点分组的滚动均值特征

    Args:
        frame: DataFrame
        source_col: 源列名
        window: 滚动窗口大小 (15分钟为一个时间步)
        new_col: 新特征列名
        min_periods: 最小观测数
    """
    rolled = (
        frame.groupby("site_id")[source_col]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame[new_col] = rolled


def _diff_feature(frame: pd.DataFrame, source_col: str, new_col: str) -> None:
    """按站点分组的一阶差分特征 (捕捉变化趋势)"""
    frame[new_col] = frame.groupby("site_id")[source_col].diff().fillna(0.0)


def engineer_features(frame: pd.DataFrame, site_ids: Iterable[str]) -> pd.DataFrame:
    """从原始数据构建38维特征矩阵

    特征分为以下几类:
      1. 时间特征 (6维): hour, minute_slot, weekday, month, dayofyear + sin/cos循环编码
      2. 风向量分解 (4维): U/V分量 (避免角度0°/360°跳变)
      3. 物理计算特征 (3维): 风切变、摄氏温度、空气密度
      4. 滚动统计特征 (6维): 风速、湿度、云量、降水的滑动均值
      5. 差分特征 (3维): 风速/湿度的一阶差分及其绝对值
      6. 原始气象特征 (10维): 气压、湿度、云量、风速/风向、温度、辐照、降水
      7. 站点独热编码 (N维): 每个站点一列

    Args:
        frame: 合并了站点信息的 DataFrame
        site_ids: 所有站点 ID 列表

    Returns:
        包含全部特征和原始列的 DataFrame
    """
    df = frame.copy()
    timestamp = df["timestamp"]

    # ---- 1. 时间特征 ----
    hour_value = timestamp.dt.hour + timestamp.dt.minute / 60.0  # 连续小时 (0-23.75)
    df["hour"] = hour_value
    df["minute_slot"] = timestamp.dt.minute // 15                    # 15分钟槽位 (0-3)
    df["weekday"] = timestamp.dt.weekday                              # 星期 (0=周一)
    df["month"] = timestamp.dt.month                                  # 月份
    df["dayofyear"] = timestamp.dt.dayofyear                          # 年天

    # 循环编码: 用 sin/cos 保留周期性 (例如 23时 和 0时 应相邻)
    df["hour_sin"] = np.sin(2.0 * np.pi * hour_value / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour_value / 24.0)
    df["doy_sin"] = np.sin(2.0 * np.pi * df["dayofyear"] / 365.0)
    df["doy_cos"] = np.cos(2.0 * np.pi * df["dayofyear"] / 365.0)

    # ---- 2. 风向分解为 U/V 分量 (风向列缺失时跳过, 用0填充) ----
    if "wind10_dir" in df.columns:
        wind10_rad = np.deg2rad(df["wind10_dir"])
        df["wind10_u"] = df["wind10_speed"] * np.cos(wind10_rad)
        df["wind10_v"] = df["wind10_speed"] * np.sin(wind10_rad)
    else:
        df["wind10_u"] = 0.0
        df["wind10_v"] = 0.0
    if "wind100_dir" in df.columns:
        wind100_rad = np.deg2rad(df["wind100_dir"])
        df["wind100_u"] = df["wind100_speed"] * np.cos(wind100_rad)
        df["wind100_v"] = df["wind100_speed"] * np.sin(wind100_rad)
    else:
        df["wind100_u"] = 0.0
        df["wind100_v"] = 0.0

    # ---- 3. 物理计算特征 (缺失列时用默认值) ----
    if "wind100_speed" in df.columns and "wind10_speed" in df.columns:
        df["wind_shear"] = df["wind100_speed"] - df["wind10_speed"]
    else:
        df["wind_shear"] = 0.0
    df["temperature_c"] = df.get("temperature_k", 288) - 273.15
    # 空气密度: ρ = P / (R·T), R=287.05
    pressure = df.get("pressure", 101325)
    temp_k = df.get("temperature_k", 288)
    df["air_density"] = pressure / (287.05 * temp_k)

    # ---- 4. 滚动统计特征 (捕捉短期趋势) ----
    _rolling_feature(df, "wind100_speed", 4, "wind100_speed_roll4")   # 1小时均值
    _rolling_feature(df, "wind100_speed", 12, "wind100_speed_roll12")  # 3小时均值
    _rolling_feature(df, "wind10_speed", 4, "wind10_speed_roll4")
    _rolling_feature(df, "humidity", 4, "humidity_roll4")
    _rolling_feature(df, "cloud_cover", 4, "cloud_cover_roll4")
    _rolling_feature(df, "precipitation", 8, "precip_roll8")          # 2小时均值

    # ---- 5. 差分特征 (一阶变化率) ----
    _diff_feature(df, "wind100_speed", "wind_ramp")      # 风速变化
    _diff_feature(df, "humidity", "humidity_ramp")       # 湿度变化
    df["wind_ramp_abs"] = df["wind_ramp"].abs()           # 风速突变幅度 (方向无关)

    # ---- 6. 滞后功率特征 (核心精度提升点, 需power_ratio列) ----
    if "power_ratio" in df.columns:
        for lag in [1, 4, 8, 24]:
            col_name = f"power_ratio_lag{lag}"
            df[col_name] = df.groupby("site_id")["power_ratio"].shift(lag)
            df[col_name] = df[col_name].fillna(0.0)      # 边界处填0

    # ---- 7. 风功率密度与湍流强度 ----
    if "air_density" in df.columns:
        df["wind_power_density"] = 0.5 * df["air_density"] * (df["wind100_speed"] ** 3)
    for window in [4, 12]:
        roll_mean = df.groupby("site_id")["wind100_speed"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        roll_std = df.groupby("site_id")["wind100_speed"].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f"turbulence_{window}"] = roll_std / (roll_mean + 1e-6)

    # ---- 8. 站点独热编码 ----
    for site_id in site_ids:
        df[f"site_{site_id}"] = (df["site_id"] == site_id).astype(float)

    return df


# ============================================================================
# 验证集划分
# ============================================================================

def make_validation_mask(frame: pd.DataFrame, val_fraction: float = 0.2) -> pd.Series:
    """按站点内时间顺序划分验证集

    使用后 20% 的数据作为验证集 (时序数据不能用随机划分, 会泄露未来信息)。
    每个站点按时间顺序的前80%为训练, 后20%为验证。

    Args:
        frame: 按 site_id + timestamp 排序的 DataFrame
        val_fraction: 验证集比例 (默认 20%)

    Returns:
        bool Series: True 表示验证集, False 表示训练集
    """
    order = frame.groupby("site_id").cumcount()
    size = frame.groupby("site_id")["site_id"].transform("size")
    split_point = (size * (1.0 - val_fraction)).astype(int)
    return order >= split_point


# ============================================================================
# 发电量计算
# ============================================================================

def energy_from_power(series: pd.Series) -> float:
    """从功率序列计算总发电量 (MWh)

    时间步长为15分钟 = 0.25小时
    Energy = Σ(Power × Δt)
    """
    return float(series.sum() * 0.25)
