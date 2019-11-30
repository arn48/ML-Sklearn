import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy as np
from PIL import Image
import PIL.ImageOps


digits = datasets.load_digits()


def extractimagefeature( filename ):
    img = Image.open(filename).resize((8, 8), Image.ANTIALIAS).convert('L')
    grayscaleinv = PIL.ImageOps.invert(img)
    dt = np.asarray(grayscaleinv)
    imagenormarr = []
    for val in dt:
        eachelement = np.divide(val, [16, 16, 16, 16, 16, 16, 16, 16])
        imagenormarr = np.append(imagenormarr, eachelement)
    imagetofeature = np.asarray(imagenormarr).astype(int)
    imgdata = imagenormarr.reshape(8, 8)
    return imagetofeature,imgdata


clf = svm.SVC(gamma=0.00001, C=100)
X,y = digits.data[:-10], digits.target[:-10]
clf.fit(X,y)
imagefeture,imagedata = extractimagefeature('dataset/two8new.jpg')
print("predicted value")
print(clf.predict([imagefeture]))
# plt.imshow(digits.images[22], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
plt.imshow(imagedata, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()