import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
data = pd.read_csv('data/data03081.csv')

# 随机选取48个样本作为训练集，其余作为测试集
train_data, test_data = train_test_split(data, train_size=48, random_state=42)

# 请根据你的数据结构修改这些参数
train_features = train_data.drop('permeate flux', axis=1)
train_target = train_data['permeate flux']
test_features = test_data.drop('permeate flux', axis=1)
test_target = test_data['permeate flux']

# 数据标准化
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# 创建并训练神经网络模型
model = MLPRegressor(hidden_layer_sizes=(80,160,80), activation='relu', solver='adam', max_iter=1000, random_state=42)
model.fit(train_features_scaled, train_target)

# 在训练集上进行预测
train_predictions = model.predict(train_features_scaled)

# 在测试集上进行预测
test_predictions = model.predict(test_features_scaled)

# 评估模型
train_mse = mean_squared_error(train_target, train_predictions)
train_mae = mean_absolute_error(train_target, train_predictions)
train_mape = np.mean(np.abs((train_target - train_predictions) / train_target)) * 100
train_r2 = r2_score(train_target, train_predictions)

test_mse = mean_squared_error(test_target, test_predictions)
test_mae = mean_absolute_error(test_target, test_predictions)
test_mape = np.mean(np.abs((test_target - test_predictions) / test_target)) * 100
test_r2 = r2_score(test_target, test_predictions)

# 输出评价指标
print("Training Set:")
print("Mean Squared Error (MSE):", train_mse)
print("Mean Absolute Error (MAE):", train_mae)
print("Mean Absolute Percentage Error (MAPE):", train_mape)
print("R-squared (R²):", train_r2)
print("\nTesting Set:")
print("Mean Squared Error (MSE):", test_mse)
print("Mean Absolute Error (MAE):", test_mae)
print("Mean Absolute Percentage Error (MAPE):", test_mape)
print("R-squared (R²):", test_r2)
