# Categorical car acceptability

The retained 1,728-row Car Evaluation table contains six categorical descriptors and its supplied acceptability target. A seed-100 stratified split reserves 432 test rows. All category vocabularies are learned on the other 1,296 rows. Compare accuracy, macro F1, balanced accuracy and the actual-row/predicted-column confusion matrix with a training-only majority baseline. The unknown buying-category probe must not add categories or refit the encoder.

## Named task

Complete fit_classifier in exercise.py. Return a fitted Pipeline with encode and neighbors steps. Use ColumnTransformer from sklearn.compose to select the six FEATURES with OneHotEncoder(handle_unknown="ignore", sparse_output=False) from sklearn.preprocessing, then uniform KNeighborsClassifier with the supplied neighbor count. Fit the whole pipeline only on the provided training rows and targets.

## Run

Use an isolated Python 3.14 environment with the supplied hash-locked libraries:

~~~sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py --output-dir output
~~~

Keep main.py, exercise.py, requirements.txt and the supplied data together when saving or exporting. The provided main program handles input preparation and evaluation. Outputs are classification-report.json.

## Worked checks and interpretation

Two distinct one-hot categories differ in two bit positions, giving Euclidean distance sqrt(2); assigning ordinal integers would impose a different geometry. An unseen category contributes zeros for that descriptor under the declared rule. The UCI labels come from an evaluation model and do not measure real vehicle safety. Attribution: Marko Bohanec (1988), Car Evaluation, https://doi.org/10.24432/C5JP48, CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. The retained header expansion is preserved.

Role: solution. The fitting function is a complete worked reference.
