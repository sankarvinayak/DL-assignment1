# Wandb
[Weights and bias](https://wandb.ai) is a framework help to track the machine learning workflow

## Run
Run is like the unit of input for wandb it all will be store together 
## Sweep
It is a set of runs with different set of hyper parameter which can be used to identify the best set or close enough set of hyper parameter while keeping the information about all the runs in a well organized manner.


## Functions
The wand_function contains some function which help in logging data easily in wandb

- `wandb_log_sample_images`
This function given the dataset name(fmnist or mnist) create a list of sample image as the first image of that class and log it into wand
```
wandb_log_sample_images("fmnist")
```
- `run_wandb_sweep`
This function contains a config from which the sweep will be defined in the wand and it will be executed. It will run a set of experiments with different hyper parameter conbination 
```
run_wandb_sweep("cs24m041","project",method="bayes",count=10):
```
- `wand_train`
This is part of the `run_wandb_sweep` or can be called directly it is expecting config from wandb which contains the hyperparameter from the sweep
```
wandb.agent(sweep_id, wand_train,count=count)
```
- `wandb_run_experiment`
This is the function which is getting called when the code is run from the main of the root. It make use of almost all the things discussed sofar and simply execute and log it in wandb

