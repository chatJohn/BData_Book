import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
# 读入CSV文件

path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")  #path为当前目录的上一级目录
data_path = path+'/user_data/'
data = pd.read_csv(data_path+'vehicles_data_20231024.csv', sep=' ')

# 将特征和目标变量拆分
y = np.array(data['车型报价'])
del data['车型报价']
X = np.array(data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#将三个回归模型打包（极端随机树、随机森林、决策树）
regressors = [
    ExtraTreesRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
]
head = 4

#训练模型
for model in regressors[:head]:
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    #画折线图比较真实值和预测值差别（选取200个测试样本点）
    plt.figure()
    plt.plot(y_test[:200], label='True')
    plt.plot(y_pred_test[:200], label='Predicted', linestyle='--')
    plt.legend()
    plt.title(model)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.show()
    #模型评估指标（解释方差分、平均绝对误差、均方误差）
    print(model)
    print("\tExplained variance train:", explained_variance_score(y_train, y_pred_train))
    print("\tExplained variance valid:", explained_variance_score(y_test, y_pred_test))
    print("\tMean absolute error:", mean_absolute_error(y_test, y_pred_test))
    print('\tMean Squared Error:', mean_squared_error(y_test, y_pred_test))
    print("真实值：", end="")
    print(np.round(y_test[:10], 2))
    print("预测值：", end="")
    print(np.round(y_pred_test[:10], 2))

#模型超参数优化过程
# param_grid ={'min_samples_split':range(1,10,1)}
#
# tree = GridSearchCV(DecisionTreeRegressor(max_depth=34, min_samples_leaf=1, min_samples_split=8), param_grid)
# tree.fit(X_train, y_train)
# y_pred_train = tree.predict(X_train)
# y_pred_test = tree.predict(X_test)
# print(DecisionTreeRegressor())
# print(tree.best_params_)
# print("优化后：")
# print("\tExplained variance train:", explained_variance_score(y_train, y_pred_train))
# print("\tExplained variance valid:", explained_variance_score(y_test, y_pred_test))
# print("\tMean absolute error:", mean_absolute_error(y_test, y_pred_test))
# print('\tMean Squared Error:', mean_squared_error(y_test, y_pred_test))
# print()

#
# param_grid = {'max_depth': range(80, 200, 10)}
# GS = GridSearchCV(RandomForestRegressor(n_estimators=330, max_features=7), param_grid, cv=10)
# GS.fit(X_train, y_train)
# y_pred_train = GS.predict(X_train)
# y_pred_test = GS.predict(X_test)
# print(GS.best_params_)
# print(RandomForestRegressor())
# print("优化后：")
# print("\tExplained variance train:", explained_variance_score(y_train, y_pred_train))
# print("\tExplained variance valid:", explained_variance_score(y_test, y_pred_test))
# print("\tMean absolute error:", mean_absolute_error(y_test, y_pred_test))
# print('\tMean Squared Error:', mean_squared_error(y_test, y_pred_test))
# print()


# param_grid = {'n_estimators': range(400,600,20)}
# GS = GridSearchCV(ExtraTreesRegressor(), param_grid, cv=10)
# GS.fit(X_train, y_train)
# y_pred_train = GS.predict(X_train)
# y_pred_test = GS.predict(X_test)
# print(GS.best_params_)
# print(ExtraTreesRegressor())
# print("优化后：")
# print("\tExplained variance train:", explained_variance_score(y_train, y_pred_train))
# print("\tExplained variance valid:", explained_variance_score(y_test, y_pred_test))
# print("\tMean absolute error:", mean_absolute_error(y_test, y_pred_test))
# print('\tMean Squared Error:', mean_squared_error(y_test, y_pred_test))
# print()
#
regressors = [
    ExtraTreesRegressor(),
    RandomForestRegressor(n_estimators=330, max_features=7),
    DecisionTreeRegressor(max_depth=87, min_samples_leaf=2, min_samples_split=7),
]