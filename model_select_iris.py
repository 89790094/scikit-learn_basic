# coding=utf-8
# coding = utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 分类：Iris-setosa（山鸢尾）、Iris-versicolor（杂色鸢尾），Iris-virginica（维吉尼亚鸢尾）
SPECIES_LABLE = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

# 使用的特征列：SepalLengthCm（花萼长度）、SepalWidthCm（花萼宽度）、PetalLengthCm（花瓣长度）、PetalWidthCm（花瓣宽度）
FEATURE_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


# 不同k值对模型的影响
def investigate_knn(iris_data, select_cols, k):
    # 取值
    x = iris_data[select_cols].values
    y = iris_data['lable'].values
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=10)
    #
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    # 打印预测准确度
    # print('k={},accuracy={:.2f}'.format(k,knn.score(x_test,y_test)))
    return knn.score(x_test, y_test)


def main():
    # 读取样本
    iris_data = pd.read_csv('./data/Iris.csv', index_col='Id')
    # 标签分离
    iris_data['lable'] = iris_data['Species'].map(SPECIES_LABLE)
    # 取值
    x = iris_data[FEATURE_COLS].values
    y = iris_data['lable'].values
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=10)

    model_dict = {'knn': KNeighborsClassifier(n_neighbors=6),
                  'logisticRe': LogisticRegression(multi_class='auto', solver='lbfgs', C=0.15),
                  'SVM': SVC(gamma='auto', C=0.5)}

    for name, model in model_dict.items():
        # 训练模型
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        print(accuracy)


if __name__ == '__main__':
    main()
