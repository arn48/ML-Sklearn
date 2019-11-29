import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

print(len(digits.data))
print(digits.data[1024])
clf = svm.SVC(gamma=0.001, C=100)

X,y = digits.data[:-10], digits.target[:-10]
clf.fit(X,y)

print(clf.predict([digits.data[1024]]))
print(digits.target[1024])
