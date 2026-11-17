# Gaussian Naive Bayes for Iris measurements

main.py uses the supplied 150-row iris.csv beside the program. Its four inputs
are sepal/petal lengths and widths in centimetres; species is the observed target.
The fixed, stratified 75/25 split uses random_state=42, giving 112 training and
38 test observations. Fit and compare the assigned GaussianNB and a training-majority
DummyClassifier. No model or hyperparameter is selected from the test scores.

GaussianNB learns a mean and population variance for every feature within each
training class. It combines class priors with Gaussian feature densities under
conditional independence. A variance floor of 1e-9 times the largest overall
training-feature variance protects numerical calculation. These distribution and
independence assumptions are approximations, not biological guarantees.

The program reports test accuracy, macro F1 and a confusion matrix with actual
species in rows and predictions in columns, plus the dummy baseline scores.
A custom vector has no supplied truth label; its predicted species is a probe.
Repeated model selection needs a separate validation or cross-validation design.

Use Python 3.14 and install requirements.txt with
python -m pip install --require-hashes -r requirements.txt. Run main.py from any
directory. The CSV and requirements are complete local inputs.

The preserved historical MultinomialNB implementation remains available for
the lesson's source audit at
https://github.com/instruction-material/AI-Level-2/blob/a445184d038e4fcbf552b87c91fa99a42af59f08/ML3-Naive-Bayes-Iris-Flowers-Classification/main.py .
MultinomialNB's count-feature model differs from the Gaussian measurement model.
