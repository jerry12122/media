from sklearn import svm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

wine=datasets.load_wine()
X=wine.data
y=wine.target
wine_target = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline']
def test(a):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    clf = svm.SVC(kernel='poly',C=1,gamma=scale)
    clf.fit(X_train,y_train)

    print("Accuracy")
    print(clf.score(X_train,y_train))
    print(clf.score(X_test,y_test))
