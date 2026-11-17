# Gaussian Naive Bayes on Iris

The supplied 150 Iris rows are divided by a seed-42 stratified split into 112 training and 38 held-out rows. The numeric measurements use a Gaussian likelihood. Fit means, variances and class priors on training rows; reuse that same model for test rows and the three provided feature vectors. Compare with a majority baseline fitted only on training labels.

## Named task

Complete fit_classifier in exercise.py. Return GaussianNB(var_smoothing=1e-9) from sklearn.naive_bayes, fitted only on the supplied training feature rows and species labels. The four columns keep their CSV order: sepal length, sepal width, petal length, petal width.

## Run

Use an isolated Python 3.14 environment with the supplied hash-locked libraries:

~~~sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py
~~~

Keep main.py, exercise.py, requirements.txt and the supplied data together when saving or exporting. The provided main program handles input preparation and evaluation. Outputs are the console feature preview, class predictions, accuracy, macro F1 and confusion matrix.

## Worked checks and interpretation

In a separate one-feature hand case with equally likely classes, means 2 and 8 and variance 1, x=2 favors the first class and x=5 ties their Gaussian scores. For the actual Iris run, inspect the learned per-class means and variances and independently count every confusion-matrix cell. Matrix rows are actual species and columns are predicted species.

Role: solution. The fitting function is a complete worked reference.
