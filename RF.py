# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取数据文件
data = pd.read_csv('data/data03081.csv')  # 数据文件路径

# 分离特征和目标变量
X = data.drop(columns=['permeate flux'])  # 目标变量列名
y = data['permeate flux']

# 设置随机种子
np.random.seed(75)

# 随机选择48个样本作为训练集，剩余的22个样本作为测试集
train_indices = np.random.choice(X.index, size=48, replace=False)  # 随机选择48个索引作为训练集
test_indices = np.setdiff1d(X.index, train_indices)  # 从所有索引中移除训练集索引，剩余的索引作为测试集

# 获取训练集和测试集的特征和目标变量
X_train, X_test = X.loc[train_indices], X.loc[test_indices]  # 根据训练集和测试集的索引获取相应的特征数据
y_train, y_test = y.loc[train_indices], y.loc[test_indices]  # 根据训练集和测试集的索引获取相应的目标变量数据

# 创建随机森林回归器对象，设置各个超参数
rf_regressor = RandomForestRegressor(
    n_estimators=8,  # 决策树数量，设为8
    max_depth=10,  # 树的最大深度，设为10
    min_samples_split=2,  # 内部节点再划分所需最小样本数，设为2
    max_features=4  # 最大特征数，设为4
)

# 使用训练集拟合回归器
rf_regressor.fit(X_train, y_train)

# 使用训练好地回归器对测试集进行预测
y_pred_train = rf_regressor.predict(X_train)
y_pred_test = rf_regressor.predict(X_test)

# 计算训练集和测试集的评价指标
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
r2_train = r2_score(y_train, y_pred_train)

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

print("\n测试集评价指标：")
print("Mean Absolute Error (MAE):", mae_test)
print("Root Mean Squared Error (RMSE):", rmse_test)
print("Mean Absolute Percentage Error (MAPE):", mape_test)
print("R-squared (R²):", r2_test)

import matplotlib.pyplot as plt

# 计算相对偏差（百分比）
train_relative_deviation = (y_train - y_pred_train) / y_train * 100
test_relative_deviation = (y_test - y_pred_test) / y_test * 100

# 绘制训练集和测试集残差图，并手动指定字体
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_train, train_relative_deviation, c='blue', marker='o', label='训练集残差')
plt.scatter(y_pred_test, test_relative_deviation, c='green', marker='s', label='测试集残差')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('训练集和测试集残差图')
plt.xlabel('预测值')
plt.ylabel('相对偏差（%）')  # 修改纵坐标标签
plt.legend()
plt.grid(True)
# 添加随机森林回归器的标签
plt.text(plt.xlim()[0], plt.ylim()[0], 'RF', fontsize=12, verticalalignment='bottom')

# 手动指定字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.show()