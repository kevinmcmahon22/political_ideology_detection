import os
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('df_convote_train.csv')
df_test = pd.read_csv('df_convote_test.csv')

X_train = df_train['text']
y_train = df_train['political_party']
X_test = df_test['text']
y_test = df_test['political_party']
# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state = 42)

tf_idf = TfidfVectorizer(stop_words = 'english')
X_train = tf_idf.fit_transform(X_train)
X_test = tf_idf.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_predict)
print(f'LR: {round(test_accuracy*100,5)}%')

# Support Vector Regression - doesn't work
# regr = svm.SVR()
# regr.fit(X_train, y_train)
# regr.predict(X_test)
# y_predict = regr.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_predict)
# print(f'SVR: {round(test_accuracy*100,5)}%')