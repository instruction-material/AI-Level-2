# Linear regression: slope, intercept and a mean baseline

Complete fit and mse in the starter. The line is prediction=slope*x+intercept. Compute training mean_x and mean_y, then slope=sum((x-mean_x)*(y-mean_y))/sum((x-mean_x)^2) and intercept=mean_y-slope*mean_x. If all training x values are equal, choose slope 0 and intercept mean_y: the slope is unidentifiable, and the declared fallback is the training-mean baseline.

The training pairs are (0,1),(1,3),(2,5),(3,7),(4,9). Their means are 2 and 5; the numerator is 20 and denominator 10, giving slope 2 and intercept 1. Test pairs (1.5,4),(2.5,6) have predictions 4 and 6, so model MSE 0. Predicting the training mean 5 on both test rows has MSE 1.

Trace the fit by hand. Compare the line with the training-mean baseline on the same held-out rows. Change one training target and explain the resulting line; keep test targets outside fitting. Check constant training x values and a single training row.

Use Python 3.10 or later and its standard library. Each role includes the fixed CSV splits beside its main.py. Paths are relative to the script, so python3 main.py also works through an absolute path from another directory.

A prediction error is predicted minus actual. MSE is the average squared error, in squared target units. Its rows must stay aligned; a low training MSE alone does not establish held-out performance. Training targets fit the model. Validation targets select among candidates. Final test targets evaluate the selected model afterwards.

Keep the fitted parameters, one hand calculation and the held-out predictions with their errors. The small authored fixtures are checkable examples, not evidence about an external population.

[Regression reference](https://scikit-learn.org/stable/modules/linear_model.html).

Run python3 -I -B test_regression.py from this pack root for the reference checks.
