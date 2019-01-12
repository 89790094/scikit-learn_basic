# coding = utf-8
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from ai_utils import plot_feat_and_price

DATA_FILE = './data/kc_house_data.csv'

# 选取特征列
# FEATURE_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
# FEATURE_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built','yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

FEATURE_COLS = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']


def plot_fitting_line(linearRegressionModel, x, y, feature):
    # 权重
    w = linearRegressionModel.coef_
    # 截距
    b = linearRegressionModel.intercept_
    # 画布
    plt.figure()
    # 散点图
    plt.scatter(x, y, alpha=0.5)

    # 直线
    plt.plot(x, w * x + b,c='red')
    plt.title(feature)
    plt.show()


def main():
    house_data = pd.read_csv(DATA_FILE, usecols=FEATURE_COLS + ['price'])
    # 观察线形关系
    # plot_feat_and_price(house_data)
    for feature in FEATURE_COLS:
        # 行向量转为列向量
        x = house_data[feature].values.reshape(-1, 1)
        y = house_data['price'].values

        # 分割数据集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=10)
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        print('特征:{},R2的值：{}'.format(feature, lr.score(x_test, y_test)))

        # 绘制拟合线，单特征表现
        plot_fitting_line(lr, x_train, y_train,feature)


if __name__ == '__main__':
    main()
