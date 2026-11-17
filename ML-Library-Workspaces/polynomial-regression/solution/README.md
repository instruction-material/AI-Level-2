# Linear and quadratic fits on the supplied curve

The original targets [10,8,7.5,7,6,6,7,9,8.5,11] and x=1 through 10 remain unchanged. The scaffold supplies a linear baseline; implement the quadratic candidate. Compare training MAE, RMSE and R-squared on the same rows and predict the same x=7 with both fitted models. The three plots are written to the current working directory. Additional flexibility can reduce fitted error without establishing generalization.

## Named task

Complete fit_model in exercise.py. Return the pair (transform, model). Fit PolynomialFeatures(degree=degree, include_bias=False) from sklearn.preprocessing on the supplied training features, then fit LinearRegression from sklearn.linear_model on the transformed features and supplied targets. The caller reuses transform.transform for predictions.

## Run

Use an isolated Python 3.14 environment with the supplied hash-locked libraries:

~~~sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py
~~~

Keep main.py, exercise.py, requirements.txt and the supplied data together when saving or exporting. The provided main program handles input preparation and evaluation. Outputs are before.png, linear.png, polynomial.png and the console metrics, coefficients and two x=7 predictions.

## Worked checks and interpretation

For the separate exact curve y=2+3*x+x*x, the quadratic intercept is 2 and coefficients are [3,1]. Transforming x=7 gives [7,49] and predicts 72. Reuse the training transformation on queries. Explain why a lower training error alone cannot select a more complex model for future data.

Role: solution. The fitting function is a complete worked reference.
