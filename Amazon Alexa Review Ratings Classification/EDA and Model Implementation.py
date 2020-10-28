# -*- coding: utf-8 -*-
"""
AMAZON ALEXA REVIEW RATINGS CLASSIFICATION
Rahul Chellani

PROBLEM STATEMENT:
Dataset consists of over 3000 Amazon customer reviews, star ratings, date of review, 
variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots. 

OBJECTIVE:
The objective is to discover insights into consumer reviews and perfrom 
sentiment analysis on the data.

www.kaggle.com/sid321axn/amazon-alexa-reviews
"""

##### IMPORTING DATA #####

# Import libraries
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import time

dataset = pd.read_csv('amazon_alexa.tsv', delimiter= '\t', quoting=3)


##### EDA #####

dataset.head()
dataset.info()

# Plots
sns.countplot(dataset['feedback'], label = "Count") 

sns.countplot(x = 'rating', data = dataset)

dataset['rating'].hist(bins = 5)

plt.figure(figsize = (40,15))
sns.barplot(x = 'variation', y='rating', data=dataset, palette = 'deep')


##### DATA PREPROCESSING #####

dataset = dataset.drop(['date','rating'], axis = 1)

variation_dummies = pd.get_dummies(dataset['variation'], drop_first = True)

dataset.drop(['variation'], axis=1, inplace=True)
dataset = pd.concat([dataset, variation_dummies], axis=1)

# Cleaning the reviews
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 3150):
  review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
alexa_cv = cv.fit_transform(corpus)
dataset.drop(['verified_reviews'], axis=1, inplace=True)
reviews = pd.DataFrame(alexa_cv.toarray())
dataset = pd.concat([dataset, reviews], axis=1)

X = dataset.drop(['feedback'],axis=1)
y = dataset['feedback']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


##### MODEL TRAINING #####

# Training the Logistic Regression model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
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


"""
##### MODEL TUNING #####

# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

# Setting the Criterion to ENTROPY
# Round 1: Entropy
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["entropy"]}
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

# Round 2: Entropy
parameters = {"max_depth": [None],
              "max_features": [3, 5, 7],
              'min_samples_split': [8, 10, 12],
              'min_samples_leaf': [1, 2, 3],
              "bootstrap": [True],
              "criterion": ["entropy"]}
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

# Predicting Test Set
y_pred = grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print(classification_report(y_test, y_pred))

# Setting the Criterion to GINI
# Round 1: Gini
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini"]}
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

# Round 2: Gini
parameters = {"max_depth": [None],
              "max_features": [8, 10, 12],
              'min_samples_split': [2, 3, 4],
              'min_samples_leaf': [8, 10, 12],
              "bootstrap": [True],
              "criterion": ["gini"]}
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

# Predicting Test Set
y_pred = grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print(classification_report(y_test, y_pred))
"""