# 导入所需的库
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler 类
from sklearn.svm import SVR

# 读取数据文件
data = pd.read_csv('data/data03081.csv')  # 数据文件路径

# 分离特征和目标变量
X = data.drop(columns=['permeate flux'])  # 目标变量列名
y = data['permeate flux']

# 随机选择48个样本作为训练集，剩余的22个样本作为测试集
train_indices = np.random.choice(X.index, size=48, replace=False)  # 随机选择48个索引作为训练集
test_indices = np.setdiff1d(X.index, train_indices)  # 从所有索引中移除训练集索引，剩余的索引作为测试集

# 获取训练集和测试集的特征和目标变量
X_train, X_test = X.loc[train_indices], X.loc[test_indices]  # 根据训练集和测试集的索引获取相应的特征数据
y_train, y_test = y.loc[train_indices], y.loc[test_indices]  # 根据训练集和测试集的索引获取相应的目标变量数据

# 使用 MinMaxScaler 对数据进行归一化处理
scaler = MinMaxScaler()  # 创建 MinMaxScaler 对象
X_train_scaled = scaler.fit_transform(X_train)  # 对训练集进行归一化处理
X_test_scaled = scaler.transform(X_test)        # 对测试集进行归一化处理

# 创建支持向量机回归器对象
svr_regressor = SVR(kernel='rbf', C=150, gamma=0.25, epsilon=0.1)  # 使用 RBF 核，并设置 C、γ 和 ε 参数

# 使用归一化后的训练集拟合回归器
svr_regressor.fit(X_train_scaled, y_train)

# 使用训练好的回归器对训练集进行预测
y_pred_train = svr_regressor.predict(X_train_scaled)

# 计算训练集的评价指标
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
r2_train = r2_score(y_train, y_pred_train)

# 使用训练好的回归器对测试集进行预测
y_pred_test = svr_regressor.predict(X_test_scaled)

# 计算测试集的评价指标
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
r2_test = r2_score(y_test, y_pred_test)

# 打印评价指标
print("训练集评价指标：")
print("Mean Absolute Error (MAE):", mae_train)
print("Root Mean Squared Error (RMSE):", rmse_train)
print("Mean Absolute Percentage Error (MAPE):", mape_train)
print("R-squared (R²):", r2_train)

# 打印测试集的评价指标
print("\n测试集评价指标：")
print("Mean Absolute Error (MAE):", mae_test)
print("Root Mean Squared Error (RMSE):", rmse_test)
print("Mean Absolute Percentage Error (MAPE):", mape_test)
print("R-squared (R²):", r2_test)
