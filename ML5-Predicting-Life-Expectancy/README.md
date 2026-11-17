# Linear and polynomial regression on country-year data

The retained `life_expectancy.csv` contains 2,938 country-year observations from 193 countries, with one row per country/year. Its target is the reported `Life expectancy ` field. Source: [Life Expectancy (WHO) dataset](https://www.kaggle.com/kumarajarshi/life-expectancy-who). These are aggregate observations, not individual medical records. The original CSV is preserved unchanged.

Use Python 3.14 in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py --output-dir output
```

The original named scripts can also run directly. Open each notebook from the folder containing `life_expectancy.csv`, select that Python environment and run its readable definition and experiment cells in order. The program reads the CSV beside its script and writes `regression-report.json` plus a concise console result. No remote data fetch occurs during a normal run.

## Named fitting task and comparison

Complete `fit_model(features, targets, degree)` in the starter. Return a fitted pipeline containing `SimpleImputer(strategy="median", keep_empty_features=True)`, `StandardScaler`, `PolynomialFeatures(degree=degree, include_bias=False)` and `LinearRegression`, fitted only on the supplied training rows.

The scaffold removes the ten missing targets, retaining 2,928 rows. Other missing numeric covariates receive training medians inside each pipeline. Country identifies the split group and Status is excluded; the 19 remaining numeric columns, including Year, are inputs. Reserve entire countries with seed 42 for a final test, then reserve validation countries with seed 43 from the remainder. A country appears in exactly one split. The saved row IDs refer to positions in the original CSV. This experiment asks about held-out countries within the observed historical period, not a future-year forecast.

Fit degree 1 and degree 2 on identical training rows. Compare validation MAE, RMSE and R-squared. Select degree 2 only if its validation MAE improves by more than 2 percent; otherwise keep the simpler degree 1. Freeze this decision and fitted state before using final test labels. Evaluate the chosen model and a training-mean `DummyRegressor` on the same final rows. Test scores describe the frozen choice; they do not select it.

## Worked errors and fitted features

For actual values 2,4 and predictions 1,5, absolute errors are 1,1, so MAE=1 year. Squared errors are 1,1, so RMSE=1 year. The actual mean is 3 and total squared deviation is 2; residual sum of squares is also 2, so R-squared=`1-2/2=0`. For actual 0,1 and predictions 2,2, R-squared is -9. It is a dimensionless comparison with the evaluated target mean, not percentage accuracy; MAE and RMSE retain the target's units.

A training column 0,2 has median 1, mean 1 and population standard deviation 1. A later missing value is replaced by 1; the scaled observed training values remain -1,1. With scaled columns a,b, degree 2 contributes a,b,a-squared,a*b,b-squared. The saved imputation values, scaler state, polynomial powers, coefficients and intercept allow predictions to be recomputed independently. Fit all learned preprocessing on training rows, including when comparing degrees.

The relationships are observational. Several contemporaneous health and economic covariates may be closely related to the reported target. Country separation addresses one source of overlap; it establishes neither causal effects nor performance on individual people, future years or another collection process. Large polynomial coefficients and implausible predictions are reasons to inspect model stability and the data, not to reinterpret R-squared as certainty. Keep the split, model choice, errors, limitations and package versions with the result.

The earlier remote-data/test-selected example remains available in [the retained revision](https://github.com/instruction-material/AI-Level-2/tree/fbe141d92a663593d0958021b662c2d50863cd65/ML5-Predicting-Life-Expectancy).

Role: completed reference.
