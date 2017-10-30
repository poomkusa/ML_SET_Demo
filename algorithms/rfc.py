import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, log_loss
from plot_learning_curve import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

import matplotlib.pylab as plt

import seaborn as sns

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
features_train, features_test, labels_train, labels_test = train_test_split(X, y_class, test_size=0.3, random_state=101)

# implementing my classifier
clf = RFC(n_estimators=25, random_state=0).fit(features_train, labels_train)

# Calculate the logloss of the model
prob_predictions_class_test = clf.predict(features_test)
prob_predictions_test = clf.predict_proba(features_test)

logloss = log_loss(labels_test,prob_predictions_test)
accuracy = accuracy_score(labels_test, prob_predictions_class_test, normalize=True,sample_weight=None)
print('accuracy: ')
print(accuracy)
print('\nlogloss: ')
print(logloss)

plot_learning_curve(clf, "Learning Curves", features_train, labels_train)
# predict class probabilities for the tourney set
#prob_predictions_tourney = clf.predict_proba(tournament_data.iloc[:,1:22])

# extract the probability of being in a class 1
#probability_class_of_one = np.array([x[1] for x in prob_predictions_tourney[:]]) # List comprehension

#t_id = tournament_data['t_id']
#
#np.savetxt(
#    '../probability.csv',          # file name
#    probability_class_of_one,  # array to savela
#    fmt='%.2f',               # formatting, 2 digits in this case
#    delimiter=',',          # column delimiter
#    newline='\n',           # new line character
#    header= 'probability')   # file header
#
#np.savetxt(
#    '../t_id.csv',          # file name
#    t_id,                   # array to save
#    fmt='%.d',              # formatting, 2 digits in this case
#    delimiter=',',          # column delimiter
#    newline='\n',           # new line character
#    header= 't_id')   # file header
