# Hidden neural regression: gradients and L2 comparison

Implement loss_gradient for the supplied two-input,three-hidden-unit tanh network with a linear output. The fixture,baselines,scaling and training loop are supplied. Hidden unit j occupies parameters[3*j:3*j+3]for its two weights and bias; output weights occupy[3*h:4*h]and the final parameter is the output bias.

For each row,delta=2*(prediction-target). Each output-weight derivative is delta*hidden_value. Each hidden preactivation derivative is delta*output_weight*(1-hidden_value^2); multiply by the corresponding input for an input-weight derivative and use it directly for the hidden bias. Average these derivatives over the batch.

Add L2*sum(weight^2)to the mean squared loss and 2*L2*weight to each weight derivative. Biases are excluded. The default is L2=0.001. Compare --l2 0 and --l2 0.01 under the same seed,split and 30-epoch budget. Select the regularization setting from validation evidence before opening the final test targets. Use the fixed comparison report to judge the neural model against the median and linear baselines.

Each role includes main.py,model.py and data/synthetic_housing.csv. The fixed 120-row authored fixture has 72/24/24 train/validation/test rows. Features are area_m2 and rooms,target is value_k in fictional thousands,and region is an error-analysis field. Fit all means/scales on train only. A zero-variance column uses scale 1.

Run python3 main.py using Python 3.10 or later; standard library only. Paths are relative to the script. Use --output results to retain the full report,epoch/residual CSV tables,and training,prediction and residual SVG plots. The prediction plot includes y=x.

Training caps are 500 total rows,30 epochs,32 rows per batch,one hidden layer with at most eight units,and at most 5 CPU seconds. Defaults are batch 16, seed 42,learning rate 0.04 and 3 CPU seconds. Validation MAE selects the saved epoch with patience 6; final test targets never affect fitting or stopping. Predictions are restored to target units before metrics.

The model-choice rule prefers the simplest median,linear or neural candidate within 1.0 fictional value unit of the best validation MAE. Preserve all three held-out MAE/RMSE values,error slices,chosen epoch and the reason to prefer or reject added model complexity.

[Neural regression reference](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

Role: solution.

Optional L2 uses mean squared loss + l2*sum(weight^2); add 2*l2*weight to each weight derivative and exclude biases. Keep each layer's parameter shape consistent with the supplied forward function.
