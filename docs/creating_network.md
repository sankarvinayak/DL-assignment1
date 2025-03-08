# Creating Network and getting the data
## Creating network
The network constring of multiple fully conneted layer 

First create a `Network` object
```
net=Network()
```
Now append fully connected layers to this network using the `append_layer` function of `fc_layer` class.

You have to pass the input and output size of each layer as well as activation function which is part of `ActivationFn` class and you have to pass the object of the activation function.

You can also change the weight initialization method to `Xavier` the default value is `random`
```
activation_fn=Sigmoid()
weight_initialisation="Xavier"
layer=fc_layer(n_inputs=inp_size, n_output=layer_size, activation_fn=activation_fn,initialization=weight_initialisation)
net.append_layer(layer)
```

Repeat this for all the hidden layer making the input size of first layer the input dimension and for final layer the output size number of class and the activation function of the final layer `Softmax` for predicting the probability distribution over the classes

Alternatively one can make use of the `construct_network` function which given the arguments return the netowk 

## Getting data
As of now the dataset using are limited to fashion mnist and mnist which are loaded from the keras directly.Also one can make ue of the function `get_data` given the dataset name(fmnist or mnist) will return the flattend verision of the trainng and testing as well as the one hot encoded version of the training labels which will be needed during the training
```
x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test=get_data(dataset_name="mnist")
```
Alternatively one can use any other dataset maintining the input and output shape of the netowrk and matching the dimensions of other function without any issue

[Training the network](Training.md)