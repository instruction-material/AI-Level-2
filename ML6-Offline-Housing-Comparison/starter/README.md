# Offline housing-value model comparison

Compare a training-median baseline, a least-squares linear regressor and a small neural regressor on the supplied 120-row fictional housing table. The fixed split is 72 training, 24 validation and 24 test rows. Features are area_m2 and rooms. region supports error analysis,while house_id and split identify records. value_k is a value in thousands of fictional units. This is authored numerical practice,not real property data or an appraisal.

The default neural model has two inputs,one hidden layer of three tanh units and a linear output. Feature and target means and population standard deviations are fitted on training rows only. A constant column uses scale 1. Targets are restored to original units before calculating MAE and RMSE.

The code uses Python 3.10 or later and its standard library. Each role contains all required code and data. Run python3 main.py from that role,or use an absolute script path from any directory. Run python3 main.py --output results to retain report.json,epochs.csv,residuals.csv and three SVG plots. Open training.svg,predicted-observed.svg and residuals.svg in a browser. The prediction plot includes y=x; residual means prediction minus observation.

## Implement the learner role

The starter leaves train_network unfinished. The supplied helpers implement data validation,training-only scaling,network evaluation and gradients,baselines,metrics and plotting. The solution completes the same workflow. Implement the bounded training loop:

1. Transform training and validation features with the already fitted training scaler. Initialize parameters from the declared seed.
2. Record epoch 0 training and validation MAE; retain a copy of the initial parameters.
3. Shuffle training-row indices with a local seeded generator. Iterate batches; use the provided loss_gradient and subtract learning_rate*gradient.
4. Check the CPU budget between batches. At each completed epoch,record both MAEs in original target units.
5. Copy parameters when validation MAE improves by more than 0.000001. Stop after six consecutive completed epochs without improvement.
6. Return the best validation parameters and their epoch,history,completed epochs and actual stopping reason.

The return contract below defines every required output key. The standard budget is at most 500 total rows,30 epochs,32 rows per batch,eight hidden units in one layer,and 3 CPU seconds for neural training. CLI flags can reduce the budget; enforced maxima remain 500 rows,30 epochs,32batch rows and 5 CPU seconds. The default batch is 16 and learning rate 0.04.

## Worked values

For actual targets [10,20] and predictions [12,17],MAE=(2+3)/2=2.5 and RMSE=sqrt((4+9)/2)=sqrt(6.5). A median baseline uses only training targets. For an even training count,it averages the two middle sorted targets.

With a zero-output neuron,prediction 0 and scaled target 2,the squared loss is 4 and the output derivative is -4. The output is linear; a sigmoid would constrain predictions to [0,1] and change this regression model. Hidden tanh derivatives use 1-h*h. All derivatives are averaged over the current training batch.

Model selection uses validation MAE. Prefer the simplest of median,linear and neural that lies within 1.0 fictional value unit of the best validation MAE. This rule is fixed before opening test targets. The final test scores for all three fixed candidates then explain the comparison; they cannot change the chosen model or stopping point.

## Complete the comparison

Record the data identity,feature meanings,seed,budget,training scaler and selected epoch. Compare validation and test MAE/RMSE for all three candidates. Inspect every test prediction and error slices by region and by the training-median target threshold. Use the curves and plots to explain whether the neural model earns its added complexity. Preserve the report and restart from the first command without hidden state.

Historical Boston Housing scripts remain unchanged in ML6-Predicting-House-Prices for the optional dataset and migration audit. This complete offline workspace is the assigned implementation.

[Neural regression reference](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

Role: starter.

## Return contract and default reference result

train_network returns a dictionary with parameters (the saved numeric list), hidden_units, best_epoch, completed_epochs, stop_reason, history, cpu_seconds, seed, batch_size and l2. Each history entry has epoch, train_mae and validation_mae. Epoch 0 is the initial model. stop_reason is max_epochs, validation_patience or cpu_budget. If the budget stops before a completed epoch, return the initial saved checkpoint. Preserve input rows and the training-fitted scaler.

The default fixture and seed 42 complete 30 epochs and select the linear baseline. Reference held-out results, rounded to six decimals:

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Training median | 38.824833 | 48.648618 |
| Linear regression | 5.676861 | 7.629158 |
| Three-hidden-unit neural network | 7.621252 | 9.462846 |

The corresponding validation MAEs are 44.048083, 6.169554 and 7.240817. The linear model earns the selection under the predeclared one-unit simplicity margin. Explain this result from the validation evidence before opening test targets.
