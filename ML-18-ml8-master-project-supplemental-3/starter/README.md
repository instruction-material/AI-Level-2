# Reproducible synthetic energy regression capstone

The supplied 120 independent, fictional rows describe temperature, planned active hours and an authored energy_units target. They use a fixed 72/24/24 training/validation/test split. Both features are available at the stated prediction time. IDs and split markers are trace metadata, not inputs. Source and license: authored course fixture; no real building, household or causal interpretation is intended.

Complete polynomial_features in model.py. Degree one returns [1,a,b]. Degree two returns [1,a,b,a*a,a*b,b*b] after applying the training-fitted standardizer. The provided pivoted normal-equation solver fits least squares with a numerical 1e-6 diagonal regularizer on non-intercept weights. This is a small bounded teaching solver, not a general replacement for a numerical linear-algebra library.

Compare a training-median baseline, the linear model and the quadratic model. Fit numeric means and population standard deviations on training rows only. Candidate degree and the 0.25-unit simplicity margin are fixed in advance. Choose the simplest candidate within 0.25 validation MAE units of the best; then freeze model identity and coefficients before opening test targets. Never change the choice from final test metrics.

For standardized input a=2,b=3, the quadratic feature row is [1,2,3,4,6,9]. For actual [1,3] and predicted [2,2], MAE=1, RMSE=1 and R-squared=0. R-squared is undefined (None) for constant true targets rather than a fabricated perfect score.

Record the problem card, input checksum and units, exact split IDs, train-fitted transformations, candidate comparison, validation choice, held-out MAE/RMSE/R-squared, residuals, and the two declared activity slices (<6 and >=6 hours). Explain one error and a limitation, and complete the generated model card with an observed interpretation. Changing test targets must preserve all coefficients, scalers and chosen candidate.

Use Python 3.10+ and the standard library. Run python3 main.py from any working directory. Optional --output report.json writes full metrics, a test residual CSV and a model card. Restart from the supplied files to reproduce results. The script caps training at 400 rows; this case uses 72. The degree-two candidate is an explicit feature-model comparison, not evidence that complexity always helps.

Reference: https://scikit-learn.org/stable/modules/model_evaluation.html

Role: starter.
