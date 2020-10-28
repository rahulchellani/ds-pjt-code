# -*- coding: utf-8 -*-
"""
DIRECTING CUSTOMERS TO SUBSCRIPTION THROUGH APP BEHAVIOR ANALYSIS
Rahul Chellani

Part-2 of Project: Model Implementation
"""

##### IMPORTING DATA #####

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

dataset = pd.read_csv('new_appdata10.csv')


##### DATA PREPROCESSING #####

X = dataset.drop(['enrolled'], axis = 1)
y = dataset['enrolled']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Removing Identifiers
train_identity = X_train['user']
X_train = X_train.drop(['user'], axis = 1)
test_identity = X_test['user']
X_test = X_test.drop(['user'], axis = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = pd.DataFrame(sc.fit_transform(X_train))
X_test2 = pd.DataFrame(sc.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


##### MODEL TRAINING #####

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print(classification_report(y_test, y_pred))

# Applying k-Fold Cross Validation to find best accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Analyzing Coefficients
pd.concat([pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)


##### MODEL TUNING #####

# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.001, 0.01, 0.1, 0.5, 0.9, 1, 2, 5, 10, 100, 1000], 'penalty': ['l1','l2']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Formatting Final Results
final_results = pd.concat([y_test, test_identity], axis = 1).dropna()
final_results['predicted_results'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_results']].reset_index(drop=True)