# Pytorch Suite

* Checkpoints
* Optuna
* Training loops


## Install

```
pip install -e .
pip install -e ./pytorch-checkpoint/
```


## Improvements

* Checkout the training log class from this link. (https://jovian.ai/arkonil/fashionmnist-vgg-transfer-learning-pytorch/v/1)
* Add a mechanism to identify the best model. Sometimes your are maxizing acc and sometimes you are minimizing another validation metric. The code should allow for this multiple cases.
* Allow the user to select which metrics to calculated after each epoch.