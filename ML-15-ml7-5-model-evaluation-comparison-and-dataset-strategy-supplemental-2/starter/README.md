# Evaluate imbalanced predictions before choosing a model

The supplied table contains 24 validation and 24 test predictions for two fixed fictional candidates. Each partition has A=18, B=4, C=2. The simple candidate always predicts A; the comparison candidate makes a different, explicitly recorded error pattern. These are authored prediction records, not outputs of a fitted real model.

Complete classification_metrics in model.py. Return accuracy, macro_f1, balanced_accuracy, the declared labels, actual-row/predicted-column confusion matrix and per-class support/precision/recall/F1. Divide-by-zero precision, recall and F1 are zero. Macro F1 averages the three declared classes; balanced accuracy averages recall only over classes present in the evaluated truth.

For a separate two-class check with declared labels A and B, actual A,A,B and predicted A,B,B give: accuracy=2/3, balanced accuracy=3/4 and macro F1=2/3. A classifier that predicts the majority in this fixture has 75% accuracy but zero recall for B and C; its balanced accuracy is 1/3 and macro F1 is 2*(18/42)/3=2/7.

Use only validation macro F1 to choose between the two fixed candidates. The simpler candidate wins if it is within 0.02 of the best validation score. Freeze the choice before scoring the test records. Report why accuracy alone hides failures, and inspect the class supports and errors. Changing final test labels must not change the choice.

Use Python 3.10+ and its standard library. Run python3 main.py from any directory; optional --output report.json preserves the complete record. Compute one confusion matrix by hand, implement the metrics, run the supplied table, and compare the complete result with the worked solution. Restart and verify the same source checksum and selection. This task evaluates precomputed outputs; the following grouped-CV project performs actual fitting.

Reference: https://scikit-learn.org/stable/modules/model_evaluation.html

Role: starter.
