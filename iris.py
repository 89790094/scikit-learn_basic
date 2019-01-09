# coding = utf-8

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载数据集
iris = datasets.load_iris()
# 加载数据
iris_x = iris.data
# 分类
iris_y = iris.target

# 划分数据集：样本集与测试集
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3, random_state=4)

# 定义模型：K近邻分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# 测试得分率
print(knn.score(x_test, y_test))

# 模型验证
# print(knn.predict(x_test))
# print(y_test)
