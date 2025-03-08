# DL-assignment 1
This repository is created for the Assignment 1 of DA6401 Introduction to Deep Learning cousrse(Mitesh M. Khapra Indian Institure of technology Madras) by Sankar Vinayak E P

This repository contains code as well as experiments done on the fasion mnist dataset

This is an implimentation of neural network framework taking inspiration(not copied) from existing framework like Pytorch and Tensorfolow primarily using numpy

It make use of [Weights and bias](https://wandb.ai) framework for experiment tracking. To test out the code given in this either you can make use of the jupyter notebook or use the python file given

### Instructions on running the code
```
python train.py --wandb_entity myname --wandb_project myprojectname
```
For more information follow the `DL_assignment_1.ipynb` file in which each step is described more clearly 
### Directory structure
```
├── DL_assignment_1.ipynb
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
##### Included activation functions
- Sigmoid
- Relu
- Tanh
- Linear
##### Included optimizer functions
- Stochastic gradient descnet
- Momentum based gradient descnet
- Nestrov accilerated gradient descent
- RMSProp
- Adam
- Nadam
##### Included initilization methods
- Xavier
- Random
##### Included loss functions
- CrossEntropy
- Mean Squared error

### Arguments supported
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | 6401_Assignment1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | cs24m041  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 14 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.999 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 4 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | ReLU | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |
<br>
Note: Even after entering the entity name you may be prompted to create or login with wandb at that point select existing account(2) and enter your private key to continue
