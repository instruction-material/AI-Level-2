# Categorical car ratings with training-only one-hot features

The input is the retained course copy of UCI Car Evaluation, from ML2-KNN-Car-Classification/cars.csv. Six ordered/nominal descriptors and the supplied acceptability class are observed source values; cluster labels are never substituted for the source target. These labels derive from a hierarchical evaluation model, not measured safety outcomes. The program samples 40 rows per sorted class with Random(42), then reserves 24/8/8 for train/validation/test. This balanced, bounded sample changes the source prevalence. All six descriptors are one-hot encoded rather than assigned arbitrary numeric distances. Attribution: Marko Bohanec (1988), Car Evaluation, UCI Machine Learning Repository, https://doi.org/10.24432/C5JP48; CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. The retained course copy expands column names; line endings are normalized to LF. Reference: https://archive.ics.uci.edu/dataset/19/car+evaluation

## Implement the named task

Complete fit_encoding in model.py. Return one sorted training vocabulary per categorical column. Encode every column into one indicator per fitted category plus a reserved unknown indicator. Fit vocabularies inside each CV fold. Record what an unseen prototype value does without changing any training vocabulary. The provided main program and other model helpers form a complete execution scaffold.

A prediction uses only the declared input features. IDs, batch names, fixed split markers and target labels are excluded from learned features. For numeric inputs, fit means and population standard deviations only on the current training fold; a constant column uses scale 1. Categorical vocabularies are learned only from that training fold, and one-hot bits are not standardized.

The model comparison fits a most-frequent dummy baseline and the candidates named in the report. Numeric cases compare a Gini decision stump and uniform KNN for k=3 and k=1. Car cases compare the declared KNN rules. The same four grouped training folds are used for every candidate. Mean fold macro F1 selects the candidate; the first declared preference within 0.02 of the best score wins. Fit each candidate on all training rows only after selection evidence is recorded. Validation results are an independent check, not a second tuning opportunity.

Freeze candidate identity and all fitted state before final test prediction. Compute accuracy, macro F1, balanced accuracy, class support/precision/recall/F1 and a confusion matrix whose rows are actual labels and columns predictions. Macro F1 includes every declared training class; zero denominators contribute zero. Balanced accuracy averages recall over classes represented in the evaluated truth. Never choose a model from its final test result.

## Hand-worked checks

For a separate two-class check with declared labels A and B, actual A,A,B and predictions A,B,B give: the confusion matrix is [[1,1],[0,1]]. Accuracy is 2/3; recalls are 1/2 and 1, so balanced accuracy is 3/4. Both class F1 scores are 2/3, hence macro F1 is 2/3. If two equally distant neighbors vote for different classes, the alphabetically first label wins. A zero-distance weighted vote ignores all nonzero-distance neighbors.

With scores dummy=0.50, stump=0.79 and knn=0.80 and preference dummy/stump/knn, the 0.02 rule selects stump. Changing held-out test labels must not alter candidate choice, fold membership, scaler, vocabulary or stored training rows.

## Evidence and reproducibility

1. Write a problem/data card naming the row unit, input fields, target, source/version or checksum, intended use and limitations.
2. Trace the named function with a tiny example before completing it.
3. Record the actual split IDs and group-disjoint folds, every candidate's fold scores, frozen selection rule and preprocessing state.
4. After the choice is frozen, inspect validation and final test metrics, class-level errors and the supplied slices or unknown-category probe.
5. Explain the simpler model decision and a concrete failure mode. Keep the distinction between synthetic/balanced demonstration data and general performance.
6. Restart and run from the same files to reproduce the result and generated model card.

Use Python 3.10+ and the standard library. Run python3 main.py from any directory. Optional --output path/report.json writes the full JSON, test prediction CSV and model card. Inputs are capped at 400 training rows and 64 encoded features; the supplied cases are smaller. KNN stores the fitted training examples; a stump is one threshold, not a full recursive tree.

Evaluation reference: https://scikit-learn.org/stable/modules/model_evaluation.html
Split reference: https://scikit-learn.org/stable/modules/cross_validation.html

Role: solution.
