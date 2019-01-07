# coding = utf-8

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate,cross_val_score,train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

# 分离训练集和测试集
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=4)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train,y_train)
# 用测试集预测得分率
# print(knn.score(x_test,y_test))

# KNN近邻分类器
# knn = KNeighborsClassifier(n_neighbors=5)
# 交叉验证
# score = cross_val_score(knn,x,y,cv=5,scoring='accuracy')
# 平均值
# print(score.mean())

# 参数调优与选择

k_range = range(1,31)
k_score = list()

for k in k_range:
    # 模型
    knn = KNeighborsClassifier(n_neighbors=k)
    # 分类
    score = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
    # 回归误差(平均方差)推荐：误差越小越好
    loss = -cross_val_score(knn,x,y,cv=10,scoring='neg_mean_squared_error')
    #k_score.append(score.mean())
    k_score.append(loss.mean())

plt.plot(k_range,k_score)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()