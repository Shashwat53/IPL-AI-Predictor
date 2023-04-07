#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
import sklearn.tree
import sklearn.ensemble
import sklearn.svm
import sklearn.model_selection


df = pd.read_csv('matches.csv')

new_df = df[['team1', 'team2', 'winner', 'toss_decision', 'toss_winner']]

new_df.dropna(inplace=True)


X = new_df[['team1', 'team2', 'toss_decision', 'toss_winner']]
y = new_df[['winner']]

all_teams = {}


cnt=0
for i in range(len(df)):
    if df.loc[i]['team1'] not in all_teams:
        all_teams[df.loc[i]['team1']] = cnt
        cnt += 1
        
    if df.loc[i]['team2'] not in all_teams:
        all_teams[df.loc[i]['team2']] = cnt
        cnt += 1

all_teams = np.array(list(all_teams.values()))


if all_teams.ndim > 1:
    all_teams = all_teams.flatten()

if len(all_teams) == 0:
    raise ValueError("all_teams cannot be empty")

le = LabelEncoder()
le.fit(all_teams)

from sklearn.preprocessing import LabelEncoder
teams = LabelEncoder()
teams.fit(all_teams)

encoded_teams = teams.transform(all_teams)

import pickle as pkl

with open('vocab.pkl', 'wb') as f:
    pkl.dump(encoded_teams, f)
with open('inv_vocab.pkl', 'wb') as f:
    pkl.dump(all_teams, f)

X = np.array(X)
y = np.array(y)

new_labels = list(set(X[:, 0]) | set(X[:, 1]) | set(X[:, 3])) # find new labels
teams.fit_transform(new_labels) # update the encoder to include new labels


X[:, 0] = teams.transform(X[:, 0])
X[:, 1] = teams.transform(X[:, 1])
X[:, 3] = teams.transform(X[:, 3])


y[:, 0] = teams.transform(y[:, 0])



fb = {'field' : 0, 'bat' : 1}
for i in range(len(X)):
    X[i][2] = fb[X[i][2]] 


X = np.array(X, dtype='int32')
y = np.array(y, dtype='int32')
y_backup = y.copy()

y = y_backup.copy()

ones, zeros = 0,0
for i in range(len(X)):
    if y[i] == X[i][0] :
        if zeros <= 375:
            y[i] = 0
            zeros += 1
        else:
            y[i] = 1
            ones += 1
            t = X[i][0]
            X[i][0] = X[i][1] 
            X[i][1] = t
        
    
        
    if y[i] == X[i][1] :
        if ones <= 375:
            y[i] = 1
            ones += 1
        else:
            y[i] = 0
            zeros += 1
            t = X[i][0]
            X[i][0] = X[i][1] 
            X[i][1] = t


print(np.unique(y, return_counts=True))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.05)

from sklearn.svm import SVC
model1 = SVC().fit(X_train, y_train)
model1.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier().fit(X_train, y_train)
model2.score(X_test, y_test)


from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=250).fit(X, y)
model3.score(X_test, y_test)

test = np.array([2,4, 1, 4]).reshape(1,-1)
model1.predict(test)
model2.predict(test)
model3.predict(test)

import pickle as pkl

with open('model.pkl', 'wb') as f:
    pkl.dump(model3, f)


with open('model.pkl', 'rb') as f:
    model = pkl.load(f)

model.predict(test)


