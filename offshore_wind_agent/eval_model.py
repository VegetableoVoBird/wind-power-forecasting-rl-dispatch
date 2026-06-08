"""
模型精度评估与参数调优脚本
"""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

raw = Path('data/raw')

# 加载数据
site_info = pd.read_csv(raw/'train_site_info.csv', encoding='gb18030')
site_info.columns = ['site_name','site_id','region','capacity_mw']
site_info['capacity_mw'] = pd.to_numeric(site_info['capacity_mw'], errors='coerce').astype(float)

train = pd.read_csv(raw/'train_weather_power.csv', encoding='gb18030')
col_map = {
    '站点编号':'site_id','时间':'timestamp','气压(Pa）':'pressure','相对湿度（%）':'humidity',
    '云量':'cloud_cover','10米风速（10m/s）':'wind10_speed','10米风向（°)':'wind10_dir',
    '温度（K）':'temperature_k','辐照强度（J/m2）':'irradiance','降水（m）':'precipitation',
    '100m风速（100m/s）':'wind100_speed','100m风向（°)':'wind100_dir','出力(MW)':'power_mw'
}
train = train.rename(columns=col_map)
train['timestamp'] = pd.to_datetime(train['timestamp'])

for c in ['pressure','humidity','cloud_cover','wind10_speed','wind10_dir','temperature_k',
           'irradiance','precipitation','wind100_speed','wind100_dir','power_mw']:
    if c in train.columns:
        train[c] = pd.to_numeric(train[c], errors='coerce')

train = train.merge(site_info, on='site_id')
train['capacity_mw'] = train['capacity_mw'].astype(float)
train = train.dropna(subset=['power_mw'])
train = train.sort_values(['site_id','timestamp']).reset_index(drop=True)
train['power_ratio'] = (train['power_mw'] / train['capacity_mw']).clip(0, 1.2)
for c in ['pressure','humidity','cloud_cover','wind10_speed','wind10_dir','temperature_k',
           'irradiance','precipitation','wind100_speed','wind100_dir']:
    train[c] = train.groupby('site_id')[c].transform(lambda x: x.ffill().bfill())

# 特征工程
df = train.copy()
ts = df['timestamp']
df['hour'] = ts.dt.hour + ts.dt.minute/60
df['minute_slot'] = ts.dt.minute // 15
df['weekday'] = ts.dt.weekday
df['month'] = ts.dt.month
df['dayofyear'] = ts.dt.dayofyear
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['doy_sin'] = np.sin(2*np.pi*df['dayofyear']/365)
df['doy_cos'] = np.cos(2*np.pi*df['dayofyear']/365)

w10r = np.deg2rad(df['wind10_dir'])
w100r = np.deg2rad(df['wind100_dir'])
df['wind10_u'] = df['wind10_speed'] * np.cos(w10r)
df['wind10_v'] = df['wind10_speed'] * np.sin(w10r)
df['wind100_u'] = df['wind100_speed'] * np.cos(w100r)
df['wind100_v'] = df['wind100_speed'] * np.sin(w100r)
df['wind_shear'] = df['wind100_speed'] - df['wind10_speed']
df['temperature_c'] = df['temperature_k'] - 273.15
df['air_density'] = df['pressure'] / (287.05 * df['temperature_k'])

for col in ['wind100_speed','wind10_speed','humidity','cloud_cover','precipitation']:
    for window in [4, 12]:
        df[f'{col}_roll{window}'] = df.groupby('site_id')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean())

df['wind_ramp'] = df.groupby('site_id')['wind100_speed'].diff().fillna(0)
df['wind_ramp_abs'] = df['wind_ramp'].abs()
df['humidity_ramp'] = df.groupby('site_id')['humidity'].diff().fillna(0)

for sid in sorted(df['site_id'].unique()):
    df[f'site_{sid}'] = (df['site_id'] == sid).astype(float)

features = [c for c in df.columns if c not in [
    'site_id','site_name','region','timestamp','power_mw','power_ratio',
    'wind10_dir','wind100_dir','wind_ramp'
]]
features = [c for c in features if df[c].dtype in ['float64','int64','float32','int32','bool']]

# 时序划分
df = df.sort_values(['site_id','timestamp'])
df['order'] = df.groupby('site_id').cumcount()
df['size'] = df.groupby('site_id')['site_id'].transform('size')
val_mask = df['order'] >= (df['size'] * 0.8).astype(int)

train_mask = ~val_mask
medians = df[train_mask][features].median().fillna(0).to_dict()
X_tr = df[train_mask][features].fillna(medians)
y_tr = df[train_mask]['power_ratio']
X_val = df[val_mask][features].fillna(medians)
y_val = df[val_mask]['power_ratio']
cap_val = df[val_mask]['capacity_mw']
pow_val = df[val_mask]['power_mw']

print(f'Training: {len(X_tr)} rows, Validation: {len(X_val)} rows, Features: {len(features)}')
print()

results = []

# 1. 当前参数
m = HistGradientBoostingRegressor(
    learning_rate=0.08, max_depth=6, max_iter=90,
    min_samples_leaf=40, l2_regularization=0.05, random_state=42)
m.fit(X_tr, y_tr)
p = m.predict(X_val).clip(0, 1.15) * cap_val
r = {'name':'当前参数','r2':r2_score(pow_val,p),'mae':mean_absolute_error(pow_val,p),
     'rmse':np.sqrt(mean_squared_error(pow_val,p))}
results.append(r)
print(f"[{r['name']}] R2={r['r2']:.4f} MAE={r['mae']:.2f}MW RMSE={r['rmse']:.2f}MW")

# 2. 适度调优
m = HistGradientBoostingRegressor(
    learning_rate=0.05, max_depth=8, max_iter=200,
    min_samples_leaf=20, l2_regularization=0.01, random_state=42)
m.fit(X_tr, y_tr)
p = m.predict(X_val).clip(0, 1.15) * cap_val
r = {'name':'适度调优','r2':r2_score(pow_val,p),'mae':mean_absolute_error(pow_val,p),
     'rmse':np.sqrt(mean_squared_error(pow_val,p))}
results.append(r)
print(f"[{r['name']}] R2={r['r2']:.4f} MAE={r['mae']:.2f}MW RMSE={r['rmse']:.2f}MW")

# 3. 激进调优
m = HistGradientBoostingRegressor(
    learning_rate=0.03, max_depth=10, max_iter=300,
    min_samples_leaf=15, l2_regularization=0.005, max_leaf_nodes=63, random_state=42)
m.fit(X_tr, y_tr)
p = m.predict(X_val).clip(0, 1.15) * cap_val
r = {'name':'激进调优','r2':r2_score(pow_val,p),'mae':mean_absolute_error(pow_val,p),
     'rmse':np.sqrt(mean_squared_error(pow_val,p))}
results.append(r)
print(f"[{r['name']}] R2={r['r2']:.4f} MAE={r['mae']:.2f}MW RMSE={r['rmse']:.2f}MW")

# 4. LightGBM
try:
    import lightgbm as lgb
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    params = {'objective':'regression','metric':'mae','boosting_type':'gbdt',
              'learning_rate':0.03,'max_depth':10,'num_leaves':128,
              'min_data_in_leaf':15,'lambda_l2':0.005,'feature_fraction':0.75,
              'bagging_fraction':0.75,'bagging_freq':1,'num_iterations':300,
              'verbose':-1,'random_state':42,'n_jobs':-1}
    m_lgb = lgb.train(params, lgb_train)
    p = m_lgb.predict(X_val, num_iteration=m_lgb.best_iteration or 300).clip(0, 1.15) * cap_val
    r = {'name':'LightGBM','r2':r2_score(pow_val,p),'mae':mean_absolute_error(pow_val,p),
         'rmse':np.sqrt(mean_squared_error(pow_val,p))}
    results.append(r)
    print(f"[{r['name']}] R2={r['r2']:.4f} MAE={r['mae']:.2f}MW RMSE={r['rmse']:.2f}MW")
except Exception as e:
    print(f"LightGBM: {e}")

print()
best = max(results, key=lambda x: x['r2'])
print(f"=== 最佳: {best['name']} R2={best['r2']:.4f} ===")
if best['r2'] < 0.85:
    print("结论: 纯调参不够, 需要增加特征工程或换模型(如XGBoost/LSTM)")
elif best['r2'] < 0.90:
    print(f"结论: 接近目标, 距离0.90还差{0.90-best['r2']:.4f}")
else:
    print("结论: 已达到0.90目标!")
