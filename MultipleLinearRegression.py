import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('dataset/CarPrice_Assignment.csv')
print(dataset.describe())
X= dataset.iloc[:,:-1]
Y= dataset.iloc[:,dataset.shape[1]-1]
X = X.drop(['CarName','car_ID'],axis=1)
fueltype = pd.get_dummies(X['fueltype'],drop_first=True)
aspiration = pd.get_dummies(X['aspiration'],drop_first=True)
doornumber = pd.get_dummies(X['doornumber'],drop_first=True)
carbody = pd.get_dummies(X['carbody'],drop_first=True)
drivewheel = pd.get_dummies(X['drivewheel'],drop_first=True)
enginelocation = pd.get_dummies(X['enginelocation'],drop_first=True)
enginetype = pd.get_dummies(X['enginetype'],drop_first=True)
cylindernumber = pd.get_dummies(X['cylindernumber'],drop_first=True)
fuelsystem = pd.get_dummies(X['fuelsystem'],drop_first=True)
X= X.drop(['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'],axis=1)
X = pd.concat([X,fueltype,aspiration,doornumber,drivewheel,enginelocation,enginetype,cylindernumber,fuelsystem],axis=1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
scr = regressor.score(X_test,Y_test)
#predicted_labels = regressor.predict(X_test)
#print(predicted_labels)
#print(accuracy_score(Y_test,predicted_labels))
print(scr*100,'%')