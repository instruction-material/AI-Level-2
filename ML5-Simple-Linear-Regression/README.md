# A fitted line and two different statistics

The ten supplied x/y observations form a small least-squares demonstration.
Run main.py with NumPy, scikit-learn and Matplotlib installed; it writes
before.png and after.png to the current directory and predicts y for x=7.

Training R-squared is 1-SSE/SST: compare the fitted line's squared residuals
with squared residuals from predicting the observed mean y. For these same
ten observations it is approximately 0.5796409014551562. This is a training
fit measurement; no independent test set is supplied.

Pearson correlation measures the signed standardized association of x and y.
Here it is approximately 0.7613415143384448. With one predictor and a fitted
intercept, evaluated on the same training observations, R-squared equals
Pearson correlation squared. Their values and meanings remain distinct.

A claim about prediction on unseen observations requires an appropriate
held-out evaluation and baseline. The separate regression projects develop
that workflow. Neither this line nor its correlation establishes causation.
