# Getting started
### Instructions on running the code
Clone the repository
```
git clone https://github.com/sankarvinayak/DL-assignment1.git
```
or Download the [zipfile][https://github.com/sankarvinayak/DL-assignment1/archive/refs/heads/main.zip] and extract in to current directry
```
cd DL-assignment1
python train.py --wandb_entity myname --wandb_project myprojectname
```
example
```
python train.py -wp "cli_try" -we "cs24m041-iit-madras" -b 32 -e 10 -o "nadam" -lr 0.0001 -w_d 0.0005 -w_i "random" -sz 64 -nhl 3 -a "ReLU"
```
The above code will log the following data in the wandb
![Wandb interface](../media/wandb_demo.png )
For more information follow the `DL_assignment_1.ipynb` file in which each step is described more clearly 
### Directory structure
```
.
├── DL_assignment_1.ipynb
├── docs
│   ├── creating_network.md
│   ├── Getting _started.md
│   ├── Training.md
│   └── wandb.md
├── media
│   └── wandb_demo.png
├── README.md
├── src
│   ├── activations
│   │   └── activation_functions.py
│   ├── cli
│   │   └── options.py
│   ├── examples
│   │   └── sample_net.py
│   ├── loss
│   │   └── loss_functions.py
│   ├── models
│   │   └── base.py
│   ├── optimizers
│   │   └── optimization_functions.py
│   ├── utils
│   │   ├── helper.py
│   │   └── metrics.py
│   └── wandb
│       └── wandb_functions.py
└── train.py
```
All the code will be contained in the src directory

## Activaton directory
function are defined in the activation directory
##### Included activation functions
- Sigmoid
- Relu
- Tanh
- Linear

## Optimizer directory
contains optimizer functions
##### Included optimizer functions
- Stochastic gradient descnet
- Momentum based gradient descnet
- Nestrov accilerated gradient descent
- RMSProp
- Adam
- Nadam


## Models directory 
contain mainly Two classes
### fc_layer
This is the main fully connected network layer You have to pass the number of input and output neurons of that layer as the input as well as the initilization method
###### Included initilization methods
- Xavier
- Random

### Network
This is the class which forms the model. Its basic building blocks are the fc_layers defined above

## loss directory 
##### Included loss functions
- CrossEntropy
- Mean Squared error

## Examples
This file contains sample codes 

## utils
This contains two files
### helper
This contain helper functions which can help in reducing the reuse of same code abstracing into a single function
#### Functions
- `construct_network`
- `get_data`
- `create_conf_mat`
- `train_model`
- `train_loop`
### metrics
This contain mainly the accuracy function

## wandb
All the functions related to the wandb will be stored here

Now it contains the following functions
- `wandb_log_sample_images`
- `wand_train`
- `run_wandb_sweep`
- `wandb_run_experiment`

[Creating and getting the data](creating_network.md)
