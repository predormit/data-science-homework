import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score

data = fetch_20newsgroups()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_transformed, y_train)

y_pred = classifier.predict(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)