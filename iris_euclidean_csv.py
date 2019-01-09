# coding = utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean

from ai_utils import do_eda_plot_for_iris


# 分类：Iris-setosa（山鸢尾）、Iris-versicolor（杂色鸢尾），Iris-virginica（维吉尼亚鸢尾）
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']

# 使用的特征列：SepalLengthCm（花萼长度）、SepalWidthCm（花萼宽度）、PetalLengthCm（花瓣长度）、PetalWidthCm（花瓣宽度）
feature_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']


# 找最近距离的训练样本，取其标签作为预测样本的标签
def get_predict_label(test_sample_feature,train_data):
    distance_list = []
    # 训练样本特征
    for idx,row in train_data.iterrows():
        train_sample_feature = row[feature_cols].values
        # 计算距离:欧氏距离
        dis = euclidean(test_sample_feature, train_sample_feature)
        distance_list.append(dis)

    # 最小距离对应的位置
    pos = np.argmin(distance_list)
    # 通过行号获取行数据
    predict_label = train_data.iloc[pos]['Species']
    return predict_label


# 读取索引，index_col设置索引
iris_data = pd.read_csv('./data/Iris.csv',index_col='Id')
# EDA
# do_eda_plot_for_iris(iris_data)

# 划分数据集：test_size测试集大小；random_state:随机划分
train_data,test_data = train_test_split(iris_data,test_size=.3,random_state=10)

predict_count = 0
for idx,row in test_data.iterrows():
    # 测试样本特征
    test_sample_feature = row[feature_cols].values
    # 预测值
    predict_label = get_predict_label(test_sample_feature, train_data)
    # 真实值
    true_label = row['Species']
    print('样本{}的真实标签{}，预测标签{}'.format(idx, true_label, predict_label))

    if true_label == predict_label:
        predict_count += 1
# 准确率,shape[0]读取数组的一维的长度
accuracy = predict_count / test_data.shape[0]
print('预测准确率{:.2f}%'.format(accuracy * 100))

# 分类器



# print(iris_data.shape)

# EDA:探索性数据分析
# iris_data = pd.read_csv('./data/Iris.csv')
# print(iris_data.describe())
