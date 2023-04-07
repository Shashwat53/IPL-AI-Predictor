import sklearn.svm
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pandas as pd


app = Flask(__name__)

df = pd.read_csv('matches.csv')
new_df = df[['team1', 'team2', 'winner', 'toss_decision', 'toss_winner']]
new_df.dropna(inplace=True)

X = new_df[['team1', 'team2', 'toss_decision', 'toss_winner']]
y = new_df[['winner']]

all_teams = {}

cnt = 0
for i in range(len(df)):
    if df.loc[i]['team1'] not in all_teams:
        all_teams[len(all_teams)] = df.loc[i]['team1']
        cnt += 1

    if df.loc[i]['team2'] not in all_teams:
        all_teams[len(all_teams)] = df.loc[i]['team2']
        cnt += 1



all_teams = np.array(list(all_teams.values()))

if all_teams.ndim > 1:
    all_teams = all_teams.flatten()

if len(all_teams) == 0:
    raise ValueError("all_teams cannot be empty")

teams = LabelEncoder()
teams.fit(all_teams)

encoded_teams = teams.transform(all_teams)

with open('vocab.pkl', 'wb') as f:
    pkl.dump(encoded_teams, f)
with open('inv_vocab.pkl', 'wb') as f:
    pkl.dump(all_teams, f)

X = np.array(X)
y = np.array(y)
y = y.ravel()


new_labels = list(set(X[:, 0]) | set(X[:, 1]) |
                  set(X[:, 3]))  
teams.fit_transform(new_labels)  

X[:, 0] = teams.transform(X[:, 0])
X[:, 1] = teams.transform(X[:, 1])
X[:, 3] = teams.transform(X[:, 3])

print(y.shape)

if len(y.shape) == 1:
    y = teams.transform(y)
else:
    y[:, 0] = teams.transform(y[:, 0])


fb = {'field': 0, 'bat': 1}
for i in range(len(X)):
    X[i][2] = fb[X[i][2]]

X = np.array(X, dtype='int32')
y = np.array(y, dtype='int32')
y_backup = y.copy()

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

ones, zeros = 0, 0
for i in range(len(X)):
    if y[i] == X[i][0]:
        if zeros <= 375:
            y[i] = 0
            zeros += 1
        else:
            y[i] = 1
            ones += 1
            t = X[i][0]
            X[i][0] = X[i][1]
            X[i][1] = t

    if y[i] == X[i][1]:
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model1 = SVC().fit(X_train, y_train)
model1.score(X_test, y_test)

model2 = DecisionTreeClassifier().fit(X_train, y_train)
model2.score(X_test, y_test)

model3 = RandomForestClassifier(n_estimators=250).fit(X, y)
model3.score(X_test, y_test)

print("SVM score:", model1.score(X_test, y_test))
print("Decision Tree score:", model2.score(X_test, y_test))
print("Random Forest score:", model3.score(X_test, y_test))

test = np.array([2, 4, 1, 4]).reshape(1, -1)
model1.predict(test)
model2.predict(test)
model3.predict(test)

with open('model.pkl', 'wb') as f:
    pkl.dump(model3, f)


with open('model.pkl', 'rb') as f:
    model = pkl.load(f)

model.predict(test)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    team1 = str(request.args.get('list1'))
    team2 = str(request.args.get('list2'))

    if team1 == team2:
        return redirect(url_for('index'))

    toss_win = int(request.args.get('toss_winner'))
    choose = int(request.args.get('fb'))

    print(team1, team2, toss_win, choose)

    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)

    with open('inv_vocab.pkl', 'rb') as f:
        inv_vocab = pkl.load(f)

    team1 = str(request.args.get('list1'))
    team2 = str(request.args.get('list2'))

    
    for team in set(new_df['team1']) | set(new_df['team2']):
        if team not in all_teams:
            all_teams[team] = len(all_teams)
            print(all_teams)

    try:
        cteam1 = teams.transform([team1])[0]
    except IndexError:
        print("Invalid team index: {team1}")
        cteam1 = None
    try:
        cteam2 = teams.transform([team2])[0]
    except IndexError:
        print("Invalid team index: {team2}")
        cteam1 = None


    if cteam1 == cteam2:
        return redirect(url_for('index'))

    lst = np.array([cteam1, cteam2, choose, toss_win],dtype='int32').reshape(1,-1)

    prediction = model.predict(lst)

    if prediction == 0:
        return render_template('predict.html', data=team1)
    else:
        return render_template('predict.html', data=team2)

if __name__ == "__main__":
    app.run(debug=True)
