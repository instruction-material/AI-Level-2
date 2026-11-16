# Linear-output neuron: regression loss and baseline comparison

Implement loss_gradient for a single linear-output neuron using the supplied housing fixture. Parameters are [area_weight,rooms_weight,bias]on training-standardized features and targets; this variant has zero hidden units.

For prediction z and target y,use squared error (z-y)^2 and derivative 2*(z-y). Multiply that derivative by each feature for the two weight derivatives; the bias derivative is unchanged. Average over the batch. The shared implementation also supports hidden units,but the configured lab uses the linear neuron.

Trace the zero-parameter gradient on one row. Compare the trained neuron with exact least-squares linear regression on the same split. Compare one epoch with thirty using validation evidence,then inspect the held-out MAE/RMSE. The two linear models can share an optimum while reaching different parameters under a finite training budget.

Each role includes main.py,model.py and data/synthetic_housing.csv. The fixed 120-row authored fixture has 72/24/24 train/validation/test rows. Features are area_m2 and rooms,target is value_k in fictional thousands,and region is an error-analysis field. Fit all means/scales on train only. A zero-variance column uses scale 1.

Run python3 main.py using Python 3.10 or later; standard library only. Paths are relative to the script. Use --output results to retain the full report,epoch/residual CSV tables,and training,prediction and residual SVG plots. The prediction plot includes y=x.

Training caps are 500 total rows,30 epochs,32 rows per batch,one hidden layer with at most eight units,and at most 5 CPU seconds. Defaults are batch 16, seed 42,learning rate 0.04 and 3 CPU seconds. Validation MAE selects the saved epoch with patience 6; final test targets never affect fitting or stopping. Predictions are restored to target units before metrics.

The model-choice rule prefers the simplest median,linear or neural candidate within 1.0 fictional value unit of the best validation MAE. Preserve all three held-out MAE/RMSE values,error slices,chosen epoch and the reason to prefer or reject added model complexity.

[Neural regression reference](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

Role: solution.

Optional L2 uses mean squared loss + l2*sum(weight^2); add 2*l2*weight to each weight derivative and exclude biases. Keep each layer's parameter shape consistent with the supplied forward function.
