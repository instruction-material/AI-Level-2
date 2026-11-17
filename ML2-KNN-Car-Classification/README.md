# Categorical car acceptability

Trace the fitted encoder and seven-neighbor rule, then compare the complete reference with a completed starter.

The fitting function uses Pipeline from sklearn.pipeline, OneHotEncoder from
sklearn.preprocessing, ColumnTransformer from sklearn.compose, and
KNeighborsClassifier from sklearn.neighbors. Its
supplied neighbors parameter becomes n_neighbors.

The supplied cars.csv has 1,728 rows and six categorical inputs. Preserve their
recorded strings; arbitrary ordinal numbers introduce unsupported distances.
One-hot encoding fits one vocabulary per column on training observations only.
Changing one known category changes two indicator bits, giving squared distance
two when all other columns agree. An unknown value contributes zeros for that
column; inspect the supplied unknown probe and the retained fitted vocabularies.

The fixed stratified 75/25 split uses seed 100 and retains 1,296 training and 432
test row IDs. Fit the assigned k=7 classifier and a training-majority dummy
baseline. No k or model is chosen from test scores. Report accuracy,
balanced accuracy, macro F1
and confusion matrices in the recorded label order, with actual rows and
predicted columns. Keep the training order and package versions for reproduction.
A different model-selection experiment needs its own validation design.

Run main.py from any directory. It prints summary metrics and writes the complete
classification-report.json, including split IDs, predictions, categories, query
and actual versions. --output-dir selects a local output folder. Use Python 3.14
and python -m pip install --require-hashes -r requirements.txt.

The acceptability target comes from a hierarchical evaluation model, rather than
observed vehicle safety. The six descriptors are inputs; class labels and row
identifiers are excluded from learned features.

Source: Marko Bohanec (1988), Car Evaluation, UCI Machine Learning Repository,
https://doi.org/10.24432/C5JP48 . CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ . The retained course table expands
column names; its original CSV bytes are preserved.
