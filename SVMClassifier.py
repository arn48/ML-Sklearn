import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from skimage.io import imread
import numpy as np
from PIL import Image
import PIL.ImageOps

digits = datasets.load_digits()
img = Image.open('two8.jpg').convert('L')
grayscaleinv = PIL.ImageOps.invert(img)
dt = np.asarray(grayscaleinv)
reshape = dt.reshape(1,64)
print(reshape)
print(digits.data[1024])
print(len(digits.data))
print(digits.data[1024])
clf = svm.SVC(gamma=0.001, C=100)

X,y = digits.data[:-10], digits.target[:-10]
clf.fit(X,y)

print(clf.predict([digits.data[1024]]))
print(digits.target[1024])
print(clf.predict(reshape))
plt.imshow(dt, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()