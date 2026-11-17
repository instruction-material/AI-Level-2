# Linear regression: ten supplied points

The example's x and y arrays are supplied in main.py. Use an isolated Python 3.14 environment, install the hash-locked requirements.txt with python -m pip install --require-hashes -r requirements.txt, then run python main.py. Plots are written to the current working directory.

Inspect the scatter plot, fitted line or curve, printed coefficients and prediction at x=7. Training R-squared is dimensionless; it compares residual squared error with the squared deviations of these same ten target values around their mean. Pearson correlation measures a different quantity. MAE and RMSE, where printed, retain the target's units.

These are fitted-sample demonstrations. Their training errors describe the supplied points; assessing generalization requires separate evaluation data. Polynomial prediction reuses the transformation fitted for training. Keep the distinction between interpolation within x=1 through 10 and extrapolation beyond that range.

Algorithm reference: https://scikit-learn.org/stable/modules/linear_model.html
