# coding = utf-8

from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state=22, n_clusters_per_class=1, scale=100)

# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()

# 处理数据范围(-1，1)
x = preprocessing.scale(x)

# 降维处理,数据范围,默认(0,1)
# x = preprocessing.minmax_scale(x,feature_range=(0,1))


# 数据分组(训练集、测试集)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

# 加载模型：svm支持向量机
clf = SVC(gamma='auto')
clf.fit(x_train,y_train)

# 测试得分率
print(clf.score(x_test,y_test))
