# coding = utf-8

from sklearn import datasets
from sklearn.externals import joblib
import pickle

iris = datasets.load_iris()
x, y = iris.data, iris.target

# pickle
with open('m/clf.pickle', 'rb') as f:
    clf = pickle.load(f)
    # 验证
    print(clf.predict(x[0:1]))

# joblib

clf_1 = joblib.load('m/clf.pkl')
# 验证
print(clf_1.predict(x[1:2]))
print(y[2])