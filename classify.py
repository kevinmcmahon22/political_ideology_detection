import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

    

convote_data['text'] = convote_data['text'].apply(clean_text)
X = convote_data.text
y = convote_data.party_label
tf_idf = TfidfVectorizer(stop_words = 'english')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
X_train_fit = tf_idf.fit_transform(X_train)
X_test_fit = tf_idf.transform(X_test)
model = LogisticRegression()
model.fit(X_train_fit,y_train)
y_predict = model.predict(X_test_fit)
Y_predict_train = model.predict(X_train_fit)
score_train = accuracy_score(y_train, Y_predict_train)
score_test = accuracy_score(y_test, y_predict)
score_train
score_test