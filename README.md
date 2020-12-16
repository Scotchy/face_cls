# Set configuration file

First of all, it is necessary to set all paths in config.yaml.

# Train a network

Create a yaml configuration file defining all training and model parameters.

Template: 
```yaml
training:
  epochs: 18
  batch_size: 100

  optimizer:
    obj:Adam : {lr : 0.001}
    module: torch.optim

  scheduler:
    obj:MultiStepLR : {milestones: [2, 6, 10, 14]}
    module: torch.optim.lr_scheduler

  weight: [0.5, 0.5] # Weights for the loss
  loss:
    obj:BCELoss : {}
    module: torch.nn

model:
  obj:Baseline : {}
  module: models

transforms:
  obj:Normalize : {}
  obj:Augmentation1 : {}
  obj:Augmentation2 : {}
  ...
  module: dataset.transforms
```

You can then launch the training with the following command :
```shell
python train.py ./params/param_file.yaml
```

A checkpoint will be saved in `./checkpoints` every two epochs.

# Inferences

Type
```shell
python infer.py ./checkpoints/model_name/e12_inf.pth
```
to make inferences on test set with the model `model_name` (checkpoint from epoch n°12). Inferences will be saved in `inferences/model_name/e12_inf.txt`. 

It is also possible to fit a logistic regressor on the output of different models :
```shell
python infer_vote_cls.py ./checkpoints/model1.pth ./checkpoints/model2.pth
```
