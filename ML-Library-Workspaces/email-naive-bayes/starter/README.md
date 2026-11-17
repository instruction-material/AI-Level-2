# Fictional email classification

The role includes 64 authored messages plus two deliberately repeated rows. The scaffold removes exact text/label duplicates, rejects contradictory labels and then stratifies raw text into 38 training, 13 validation and 13 final test rows. Labels are ham=0 and spam=1. The training vocabulary must never learn a held-out-only token. Compare the fixed count model with binary token presence on validation rows; final test labels cannot select a model. The original large historical dataset and notebooks remain in their original source folder, outside this complete role.

## Named task

Complete fit_pipeline in exercise.py. Return a fitted Pipeline containing words=CountVectorizer(binary=binary) from sklearn.feature_extraction.text and classifier=MultinomialNB(alpha=1.0) from sklearn.naive_bayes. Import Pipeline from sklearn.pipeline and fit it only on the supplied training messages and labels.

## Run

Use an isolated Python 3.14 environment with the supplied hash-locked libraries:

~~~sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py --output-dir output
~~~

Keep main.py, exercise.py, requirements.txt and the supplied data together when saving or exporting. The provided main program handles input preparation and evaluation. Outputs are spam-report.json and a console summary.

## Worked checks and interpretation

For ham 'meeting meeting note' and spam 'prize prize note', smoothing alpha=1 gives ham likelihoods 3/6,2/6,1/6 and the reverse for spam. Query 'meeting note' has normalized ham probability 3/4. Actual labels 0,0,1 predicted as 0,1,1 give spam precision 1/2, recall 1 and F1 2/3. The easy authored vocabulary is classroom evidence, not evidence of general inbox filtering.

Role: starter. The fitting function intentionally raises NotImplementedError until it is implemented.
