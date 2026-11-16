# Polynomial regression: validation-based degree selection

Complete fit_polynomial in the starter. A degree-d row is [1,x,...,x^d]. Build the normal equations X^T*X*coefficients=X^T*y and pass that small system to the supplied pivoted solver. Coefficients are ordered [constant,x,x^2]. Degrees 0 through 2 are supported; at least degree+1 distinct x values are required.

The five training pairs use x=-2,-1,0,1,2 and y=0,0,2,6,12. The quadratic is y=2+3*x+x^2, so its coefficients are [2,3,1]. The fitted linear candidate is y=4+3*x. Validation pairs are (-1.5,-0.25),(0.5,3.75): the linear validation MSE is 1.5625; the quadratic's is zero up to floating-point rounding. Choose degree 2 using only that validation split.

Only then open test pairs (-0.5,0.75),(1.5,8.75). The selected quadratic predicts 0.75 and 8.75. Repeat selection on a linear fixture and verify that an exact validation-loss tie chooses degree 1. Explain why choosing the degree using final test labels leaks information.

This normal-equations solver is a small numerical exercise with well-scaled coordinates. Poorly scaled or rank-deficient systems raise a clear error. Use a QR/SVD least-squares library for larger or ill-conditioned designs.

Use Python 3.10 or later and its standard library. Each role includes the fixed CSV splits beside its main.py. Paths are relative to the script, so python3 main.py also works through an absolute path from another directory.

A prediction error is predicted minus actual. MSE is the average squared error, in squared target units. Its rows must stay aligned; a low training MSE alone does not establish held-out performance. Training targets fit the model. Validation targets select among candidates. Final test targets evaluate the selected model afterwards.

Keep the fitted parameters, one hand calculation and the held-out predictions with their errors. The small authored fixtures are checkable examples, not evidence about an external population.

[Regression reference](https://scikit-learn.org/stable/modules/linear_model.html).

Role: starter.
