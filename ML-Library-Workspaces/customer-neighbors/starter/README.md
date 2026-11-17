# Customer segment imitation with KNN

The supplied 200-row table has no measured customer-type target. A bounded, seeded five-cluster model is fitted only on 150 training rows and supplies proxy labels. Fifty held-out rows measure imitation of that fixed rule. The KNN model uses a separate training-only scaler and compares with a training-only majority baseline. Cluster numbers are arbitrary labels. The report includes split IDs, parameters, predictions, confusion matrices and a hypothetical [90,20] query.

## Named task

Complete fit_classifier in exercise.py. Return a fitted Pipeline with scale and neighbors steps: StandardScaler followed by uniform KNeighborsClassifier using the supplied neighbor count, fitted only on features and targets. Import these classes from sklearn.preprocessing, sklearn.neighbors and sklearn.pipeline. The two numeric inputs are annual income and spending score; exclude ID, age and gender.

## Run

Use an isolated Python 3.14 environment with the supplied hash-locked libraries:

~~~sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py --output-dir output
~~~

Keep main.py, exercise.py, requirements.txt and the supplied data together when saving or exporting. The provided main program handles input preparation and evaluation. Outputs are classification-report.json, before_clustering.png and clustered_graph.png.

## Worked checks and interpretation

For training values 0 and 2, the mean and population standard deviation are 1, so a later value 100 transforms to 99. Identical feature vectors remain distinct observations. Labels A,A,B predicted as A,B,B give accuracy 2/3 and macro F1 2/3. These model scores establish agreement with generated cluster labels, rather than real customer behavior.

Role: starter. The fitting function intentionally raises NotImplementedError until it is implemented.
