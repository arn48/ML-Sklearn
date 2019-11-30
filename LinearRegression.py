import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

dataset = pd.read_csv('dataset/student_scores.csv')
print(dataset.describe())
dataset.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
#plt.show()
x= dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print (df)
acc = regressor.score(x_test,y_test)
print (acc)

