# coding = utf-8

from sklearn import datasets
from sklearn import svm

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
x, y = iris.data, iris.target
# training
clf.fit(x,y)

# save model as method one:pickle

import pickle
with open('m/clf.pickle','wb') as f:
    pickle.dump(clf,f)