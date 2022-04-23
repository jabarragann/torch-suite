# Pytorch Suite

Features included in torchsuite
* Checkpoints: Save models and training plots.
* Optuna: Hyperparameter tunning.
* Training loops: Pytorch training loops.


## Install

```
pip install -e .
pip install -e ./pytorch-checkpoint/
```


## Improvements

* Checkout the training log class from this link. (https://jovian.ai/arkonil/fashionmnist-vgg-transfer-learning-pytorch/v/1). It would be cool to print a little ascii table at the end of each epoch showing the current and past endofepoch metrics.
* Add a mechanism to identify the best model. Sometimes your are maxizing acc and sometimes you are minimizing another validation metric. The code should allow for this multiple cases.
* Export checkpoint metrics to CSV for plotjuggler.
* Clean even further the training loop. Implement a system of callbacks that will take care of additional tasks that are not related to training, e.g., saving best models, calculating metrics. You can use as a reference the cb system in keras.(cb_at_epoch_end,cb_at_batch_end,cb_at_training_end)
* Documentation.

## Done
* Allow the user to select which metrics to calculated after each epoch.
