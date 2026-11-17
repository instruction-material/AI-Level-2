# Country-year linear and regularized regression

The retained 2,938 country-year observations cover 193 countries. Ten missing targets are removed; 19 numeric inputs, including Year, remain after excluding Country, Status and the target. Country is the grouping identity. Seeds 42 and 43 produce mutually disjoint training, validation and test country groups. Training medians impute covariates. The fixed quadratic Ridge penalty stabilizes correlated polynomial features; its alpha is fixed before validation. Choose it only when validation MAE improves by more than 2 percent over linear regression. Freeze the choice, then record both candidates and a training-mean dummy on the same final test rows. Their final-test metrics do not reselect the chosen model. Coefficients express observational associations, not causal effects.

## Named task

Complete fit_model in exercise.py. Return a fitted Pipeline with fill=SimpleImputer(strategy='median', keep_empty_features=True), scale=StandardScaler, polynomial=PolynomialFeatures(degree=degree, include_bias=False), and regression=LinearRegression for degree 1 or Ridge(alpha=1.0) for degree 2. Import the estimators from sklearn.impute, sklearn.preprocessing, sklearn.linear_model and sklearn.pipeline. Fit every learned step only on supplied training features and targets.

## Run

Use an isolated Python 3.14 environment with the supplied hash-locked libraries:

~~~sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py --output-dir output
~~~

Keep main.py, exercise.py, requirements.txt and the supplied data together when saving or exporting. The provided main program handles input preparation and evaluation. Outputs are regression-report.json and a console comparison.

## Worked checks and interpretation

Actual targets [2,4] and predictions [1,5] give MAE=1 year, RMSE=1 year and R-squared=0. The saved report gives two slice dimensions: year relative to the training median, and covariate missingness. It also resamples complete held-out-country error blocks 200 times with seed 42 and reports the central 95 percent of the resulting MAEs. This is conditional resampling variation for the observed country blocks; it supplies no claim of individual prediction, future forecasting or external-population coverage. Keep the original grouped split, fitted state and selected model fixed during error analysis.

Role: solution. The fitting function is a complete worked reference.
