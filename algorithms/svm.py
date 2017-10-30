from plot_learning_curve import *

import pandas as pd
from sklearn.svm import SVC as svc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

import matplotlib.pylab as plt

import seaborn as sns
sns.despine()


data_original = pd.read_csv('/home/poom/Desktop/ML_SET_Demo/data/ORI.BK.csv')[::-1]

WINDOW = 30
FORECAST = 1

#Create features from previous days data
X_temp = data_original.copy()
X_temp.drop(['Date','Adj Close'], axis=1, inplace=True)
X = X_temp.copy()
for j in range(WINDOW):
    for i in range(len(X_temp.columns)):
        X[X_temp.columns[i]+"-"+str(j+1)] = X_temp.iloc[:,i].shift((j+1)*(-1))
X['Future Close'] = X_temp['Close'].shift(FORECAST)

#Create labels
X['Up'] = X['Future Close'] > X['Close']
X = X.dropna()
y_reg = X['Future Close']
y_class = X['Up']
X.drop(['Future Close', 'Up'], axis=1, inplace=True)

#split into train, test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y_class, test_size=0.3, random_state=101)

clf = svc(C=100).fit(X_train, Y_train)

#predicting our target value
predictions = clf.predict(X_test)
print(predictions)

accuracy = accuracy_score(predictions,Y_test)
print(accuracy)
#c = 1.0 -> 0.514361849391
#c = 100.0 -> 0.518133997785

plot_learning_curve(clf, "Learning Curves", X_train, Y_train)