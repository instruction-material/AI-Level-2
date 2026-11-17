"""
Dataset: https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv

Use scikit-learn to build a Naive Bayes classifier for the type of species a flower is. Then run your classifier on some test data to determine its accuracy.
"""

from pathlib import Path
from exercise import fit_classifier

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Dataset: https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv

# read in data
data = pd.read_csv(Path(__file__).resolve().with_name('iris.csv'))

# build feature vectors(X) and label(y) for each vector
X = data.drop("species", axis=1)
y = data["species"]

print(X.head())
print()
print(y.head())
print()

# Convert pandas dataframe to numpy array
X = X.to_numpy()
y = y.to_numpy()

# split into testing and training data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# create and train Naive Bayes classifier
model = fit_classifier(x_train, y_train)

# run the fixed classifier on held-out test data
y_pred = model.predict(x_test)
print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100) + " %")
macro_f1 = f1_score(y_test, y_pred, average="macro")
class_labels = model.classes_
matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
print("GaussianNB macro F1: " + str(macro_f1))
print("Confusion matrix (actual rows, predicted columns):")
print(class_labels)
print(matrix)

# Fit the majority baseline on the same training observations only.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(x_train, y_train)
baseline_predictions = baseline.predict(x_test)
print("Dummy accuracy: " + str(accuracy_score(y_test, baseline_predictions) * 100) + " %")
print("Dummy macro F1: " + str(f1_score(y_test, baseline_predictions, average="macro")))
print()

# run classifier on your own feature vectors
custom_vectors = [
    [5.2, 3.1, 1.6, 0.3],
    [6.0, 3.0, 4.8, 1.1],
    [7.5, 3.0, 6.2, 2.0],
]
# custom_vectors = [[1.2, 2.5, 2.3, 3.6], [2.7, 5.6, 5.2, 8.1], [5.1, 6.2, 3.4, 7.1]]
print(model.predict(custom_vectors))
