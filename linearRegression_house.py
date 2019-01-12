# coding = utf-8
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from ai_utils import plot_feat_and_price

DATA_FILE = './data/kc_house_data.csv'

# 选取特征列
# FEATURE_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
FEATURE_COLS = ['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view']
# FEATURE_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built','yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']


def main():
    house_data = pd.read_csv(DATA_FILE, usecols=FEATURE_COLS + ['price'])
    # 观察线形关系
    # plot_feat_and_price(house_data)
    # 构建数据集
    x = house_data[FEATURE_COLS].values
    y = house_data['price'].values
    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=10)
    # 建立线性回归模型
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    # 验证模型（R2）
    r2_score = lr.score(x_test, y_test)
    print(r2_score)

    # 单个样本房价的预测
    single_test_feature = x_test[50, :]
    y_predict = lr.predict([single_test_feature])
    # 真实值与预测值
    print(y_test[50], y_predict)


if __name__ == '__main__':
    main()
