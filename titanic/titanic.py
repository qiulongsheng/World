# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:26:49 2017

@author: byq
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import  GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
index = test['PassengerId']
y = train['Survived']
num_train = train.shape[0]
X = train.append(test) #合并数据集
X.drop(['PassengerId', 'Survived'], axis=1, inplace=True)

LEnc = LabelEncoder()
Oenc = OneHotEncoder()


#Name
X['NameLen'] = X['Name'].apply(lambda x: len(x))
X['NameTitle'] = X['Name'].apply(lambda x:x.split(',')[1].split('.')[0])
X.drop('Name', axis=1, inplace=True)

#Sex
X['Sex'] = LEnc.fit_transform(X['Sex'])

#Age
X['AgeIsNull'] = X['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
X['Age'] = X[:num_train].groupby(['NameTitle','Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

#Family
individual = X[['Parch', 'SibSp']].apply(np.sum, axis=1).values
X['FamilySize'] = np.where(individual==0, 'Alone',
                         np.where(individual<=3, 'Nuclear', 'Big'))
X.drop(['Parch', 'SibSp'], axis=1, inplace=True)

#Ticket
head_letter = X['Ticket'].str[0]
X['TicketHead'] = np.where(head_letter.isin(['1', '2', '3', 'S', 'P', 'C', 'A']), head_letter,
                        np.where(head_letter.isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
X['TicketHead'] = X['TicketHead']#.astype(str)
X.drop('Ticket', axis=1, inplace=True)

#Cabin
X['CabinHead'] = X['Cabin'].str[0]
Cabin_num = X['Cabin'].apply(lambda x: str(x).split()[-1][1:])
Cabin_num = Cabin_num.apply(lambda x: int(x) if x not in ['', 'an'] else np.NaN)
X['CabinNum'] = pd.qcut(Cabin_num, 3).replace(np.NaN, 'Unknown')
X.drop('Cabin', axis=1, inplace=True)

#Embarked
X['Embarked'].fillna(X['Embarked'].mode(), inplace=True)

#Fare
X['Fare'].fillna(X[:num_train]['Fare'].mean(), inplace=True)

#X['Pclass'] = X['Pclass'].astype(str)

cate_col = ['Pclass', 'NameTitle', 'TicketHead', 'CabinHead', 'CabinNum',
            'Embarked', 'FamilySize']
for column in cate_col:
     X = pd.concat([X, pd.get_dummies(X[column])], axis=1)

X.drop(cate_col, axis=1, inplace=True)


train_, test = X[:num_train], X[num_train:]


#GridSearch:
param = {
    'loss' : ['deviance', 'exponential'],
#    'criterion': ['gini', 'entropy'],
    'n_estimators': [20, 40, 60, 80],
#    'learning_rate': [ 0.05, 0.1, 0.3],
    'max_depth': [2, 3, 5, 7],
#    'min_impurity_split': [1e-3, 1e-4, 1e-5],
    'min_samples_split': [4, 8, 16, 24],
    'min_samples_leaf': [1, 2],
    'max_features': [1.0, 'sqrt'],
    'verbose': [1]
    }
#model = RandomForestClassifier()
'''
{'loss': 'exponential', 'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 2,
'min_samples_split': 8, 'n_estimators': 60, 'verbose': 1}
'''
model = GradientBoostingClassifier()
#scoring='accuracy',
GV = GridSearchCV(model, param, verbose=1, n_jobs=1)
#print('开始GV......')
GV.fit(train_, y) #GV.fit(X_train, y_train)
print(GV.best_params_)
model = GV.best_estimator_

'''
model = RandomForestClassifier(criterion='entropy',
                             n_estimators=50,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=1)
model.fit(train_, y)
'''

#评估
score = model.score(train_, y)
cvs = cross_val_score(model, train_, y, cv=5)
print('score: {:.4f}'.format(score, ))
print('cvs: {:.4f}, {:.4f}'.format(cvs.mean(), cvs.std()))
#print('oob_score:', model.oob_score_)

pred = model.predict(test).round(4)
pred = pd.DataFrame({'PassengerId':index, 'Survived': pred})
pred.to_csv('pred.csv', index=False)

with open('res.txt', 'a') as f:
    cont = [str(model).split('(')[0]]
    cont.append(param.values())
    cont.append(GV.best_params_)
    cont.append([score, cvs.mean(), cvs.std()])
    f.write(str(cont) + '\n')




