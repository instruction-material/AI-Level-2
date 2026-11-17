# Linear regression on ten supplied points

The original ten x values 1 through 10 and target values [4,5,6,3,5,7,3,9,13,14] remain in main.py. Plot the points first, fit the line, inspect slope/intercept and predict y for x=7. The plots are written to the current working directory. R-squared compares fitted squared errors with the mean-target baseline; Pearson correlation is a separate statistic. These are fitted-sample demonstrations, with no held-out performance claim.

## Named task

Complete fit_model in exercise.py. Return LinearRegression from sklearn.linear_model fitted to the provided numeric features and targets. Keep the model so the provided caller can inspect its coefficients, intercept, score and predictions.

## Run

Use an isolated Python 3.14 environment with the supplied hash-locked libraries:

~~~sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py
~~~

Keep main.py, exercise.py, requirements.txt and the supplied data together when saving or exporting. The provided main program handles input preparation and evaluation. Outputs are before.png, after.png and the console coefficients, training R-squared, Pearson correlation and x=7 prediction.

## Worked checks and interpretation

For training pairs (0,1),(1,3),(2,5), the slope is 2, intercept 1 and prediction at x=3 is 7. Trace a residual as prediction minus observation. Compare interpolation inside x=1 through 10 with extrapolation outside that range. A training score alone does not establish performance on new observations.

Role: solution. The fitting function is a complete worked reference.
