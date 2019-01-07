# coding = utf-8

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 加载数据
load_data = datasets.load_boston()

# 样本集
data_x = load_data.data

data_y = load_data.target

# 加载模型：线性回归模型
model = LinearRegression()
model.fit(data_x,data_y)

# 模型参数
# print(model.coef_)
# print(model.intercept_)

# 实例化模型时的参数设置
# print(model.get_params())

# 模型评价（百分率）
print(model.score(data_x,data_y))

# 验证
# print(model.predict(data_x[:4,:]))
# print(data_y[:4])

# 自建数据
x,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=1)

# 图例
# plt.scatter(x,y)
# plt.show()