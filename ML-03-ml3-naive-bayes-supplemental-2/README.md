# Gaussian Naive Bayes: measured features

The fictional parcel fixture supplies eight labelled training rows and four held-out rows, each with two measurements x and y. It is authored practice, not the Iris dataset. Use Python 3.10 or later and its standard library.

Complete fit and log_scores in starter/main.py. Fit each class's prior, two means and two population variances from training rows only. Clamp each variance to the fixed positive variance_floor, default 0.000001, to make constant features usable. This absolute teaching floor differs from a library's automatic variance-smoothing rule.

For a class c, add log(prior[c]) to both Gaussian log densities:
-0.5 * (log(2*pi) + log(variance) + (value-mean)^2/variance).
Choose the largest score; an exact tie chooses the alphabetically first label. The diagonal model assumes the two features are conditionally independent given the class. Its density scores are not accuracy estimates.

## Worked check

Class A has mean (2,2), class B (8,8). Both have population variances (1,1) and prior 1/2. At (2,2), A's score is log(1/2)-log(2*pi), about -2.531024, and B's is 36 lower. At (5,5) both scores are exactly nine lower than that first A score, so the declared tie rule chooses A. The held-out predictions are A,B,A,B, accuracy 1.0 on this tiny fixture.

1. Trace both feature contributions for (2,2) and (5,5).
2. Keep the four test rows outside fit and compare predictions with their labels afterwards.
3. Fit a class with a constant feature and explain the variance floor's effect.
4. Change a measurement or class balance and predict which score changes.
5. Explain why correlated measurements can make this model overconfident.

Run python3 main.py from either role folder, or use its absolute path from any directory. Each role includes both CSV files. The starter raises NotImplementedError at its TODOs; the solution supplies the same contract. Keep the fitted parameters, score trace and held-out predictions.

Run python3 -I -B test_naive_bayes.py from this pack's root for the reference checks.

[Algorithm reference](https://scikit-learn.org/stable/modules/naive_bayes.html).
