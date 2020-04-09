# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:22:58 2020

@author: jinwz
"""

#Titanic

import pandas as pd
import numpy as np

# Data loading
train = pd.read_csv(r"D:\kaggle\titanic\train.csv")
test = pd.read_csv(r"D:\kaggle\titanic\test.csv")

# Find range
a = []
for i in train:
    a.append(train[i].drop_duplicates())

b = []
for i in test:
    b.append(test[i].drop_duplicates())

i = 'Age'
print(max(train[i]),max(test[i]))
print(min(train[i]),min(test[i]))
i = 'Fare'
print(max(train[i]),max(test[i]))
print(min(train[i]),min(test[i]))

# Find Null
print('train:')
for i in train:
    if sum(train[i].isna())>0:
        print('\t',i)
print('test:')
for i in test:
    if sum(test[i].isna())>0:
        print('\t',i)

train[train['Embarked']=="S"].count()[0]
"""
Pcalss - 离散值，有序，无空值（1,2,3）
Name - 去掉此列
Sex - 无序，无空值（male,female）
Age - 连续值，有空（train，test），train：0.42-80，test：0.17-76
SibSp - 无序，无空值（0-5，8）
Parch - 无序，无空值，train：（0-6），test：（0-6,9）
Ticket - 去掉此列
Fare - 有序，有空（test），0-512.3292
Cabin - 无序，有空（很多），去掉此列
Embarked - 无序，有空（train），（S,C,Q） S:644,C:168,Q:77
"""

####Data processing
X = train.iloc[:,2:].drop(['Name','Ticket','Cabin'],axis=1)
y = train.iloc[:,1].values

X['Pclass'] = X['Pclass']/3

X['Sex'] = X['Sex'].replace('female',0).replace('male',1)

X['Age'] = X['Age'].fillna(round(X['Age'].mean()))
X['Age'] = X['Age']/max(X['Age'])
# 当SibSp不等于0-5中任何一值时，即为8
for i in range(6):
    X['SibSp{}'.format(i)] = (X['SibSp'] == i) + 0
X = X.drop(['SibSp'],axis = 1)

for i in range(7):
    X['Parch{}'.format(i)] = (X['Parch'] == i) + 0
X = X.drop(['Parch'],axis = 1)

X['Fare'] = X['Fare']/max(X['Fare'])

X['Embarked'] = X['Embarked'].fillna("S")
X['Embarked1'] = (X['Embarked'] == "S")+0
X['Embarked2'] = (X['Embarked'] == "C")+0
X = X.drop(['Embarked'],axis = 1)

X = X.values

# train_test_split
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 1/3 , random_state = 0)

#### Model training
from sklearn.metrics import confusion_matrix,accuracy_score
# LinearRegression
from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5) + 0

ac = accuracy_score(y_test,y_pred)
print('LinearRegression:',ac)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('LogisticRegression:',ac)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('DecisionTreeClassifier:',ac)

# RandomForestClassification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10 , criterion = 'entropy' , random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('RandomForestClassification:',ac)

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf' , random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('SVM:',ac)

# NaiveBayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('NaiveBayes:',ac)

"""# XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('XGBoost:',ac)
"""
# ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 9 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 19))
classifier.add(Dense(units = 4 , kernel_initializer = 'uniform' , activation = 'relu'))
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'relu'))
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
classifier.fit(X , y , batch_size = 10 , epochs = 200)

#Confusion_Matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


pid = test['PassengerId']
test = test.iloc[:,1:].drop(['Name','Ticket','Cabin'],axis=1)
test['Pclass'] = test['Pclass']/3
test['Sex'] = test['Sex'].replace('female',0).replace('male',1)

test['Age'] = test['Age'].fillna(round(test['Age'].mean()))
test['Age'] = test['Age']/max(test['Age'])
# 当SibSp不等于0-5中任何一值时，即为8
for i in range(6):
    test['SibSp{}'.format(i)] = (test['SibSp'] == i) + 0
test = test.drop(['SibSp'],axis = 1)

for i in range(7):
    test['Parch{}'.format(i)] = (test['Parch'] == i) + 0
test = test.drop(['Parch'],axis = 1)

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test['Fare'] = test['Fare']/max(test['Fare'])

test['Embarked1'] = (test['Embarked'] == "S")+0
test['Embarked2'] = (test['Embarked'] == "C")+0
test = test.drop(['Embarked'],axis = 1)

test = test.values

y_pred = classifier.predict(test)
y_pred = [i[0] for i in y_pred]
y_pred = (y_pred > (max(y_pred)-min(y_pred))/2) + 0



gender_submission = pd.DataFrame([pid.values , y_pred]).T
gender_submission.columns = ['PassengerId','Survived']

gender_submission.to_csv(r'D:\kaggle\titanic\gender_submission.csv' , index = False)