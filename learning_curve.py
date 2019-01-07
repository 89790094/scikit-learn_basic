# coding = utf-8

# 手写数字数据集
from sklearn.datasets import load_digits
from sklearn.svm import SVC
# 学习曲线模块:检视过拟合
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
x = digits.data
y = digits.target

train_size, train_loss, test_loss = learning_curve(SVC(gamma=0.0008), x, y, cv=10, scoring='neg_mean_squared_error', train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_size, train_loss_mean, 'o-', color="r",label="Training")
plt.plot(train_size, test_loss_mean, 'o-', color="g",label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend()
plt.show()