from activations.activation_functions import ReLU, Sigmoid, Softmax, Tanh
import numpy as np
from models.base import Network, fc_layer
from keras.datasets import fashion_mnist

def one_hot_encode(labels, num_classes):
  """convert output label into one hot encoded format"""
  return np.eye(num_classes)[labels]


def construct_network(inp_size:int,num_layers:int,layer_size:int,out_size:int,activation_f:str,weight_initialisation:str)->Network:
  """ Network constructor return network object which can be trained
  inp_size:int input dimension
  num_layers:int number of hidden layers
  layer_size:int number of nodes in each hidden layer
  out_size:int number of classes
  activation_f:str activation function can be sigmoid,tanh,ReLU
  weight_initialisation:str random initialization default Xavier availabel
  """
  net=Network()
  if(activation_f=='sigmoid'):
    activation_fn=Sigmoid()
  elif(activation_f=='tanh'):
    activation_fn=Tanh()
  elif(activation_f=='ReLU'):
    activation_fn=ReLU()
  net.append_layer(fc_layer(n_inputs=inp_size, n_output=layer_size, activation_fn=activation_fn,initialization=weight_initialisation))
  for i in range(num_layers-2):
    net.append_layer(fc_layer(n_inputs=layer_size, n_output=layer_size, activation_fn=activation_fn,initialization=weight_initialisation))
  net.append_layer(fc_layer(n_inputs=layer_size, n_output=out_size, activation_fn=Softmax(),initialization=weight_initialisation))
  return net


def get_data(dataset_name:str="fmnist")->tuple:
  """
  get the data from teh dataset using the api
  take dataset name can be fmnist for fasion mnist
  return x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test as tuple
  """
  if(dataset_name=="fmnist"):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  x_train_flat=x_train.reshape(x_train.shape[0],-1)
  x_test_flat=x_test.reshape(x_test.shape[0],-1)
  one_hot_y_train=one_hot_encode(y_train,np.unique(y_train).shape[0])
  one_hot_y_test=one_hot_encode(y_test,np.unique(y_test).shape[0])
  return x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test
