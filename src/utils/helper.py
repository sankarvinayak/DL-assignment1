import keras
from activations.activation_functions import ReLU, Sigmoid, Softmax, Tanh
from loss.loss_functions import CrossEntropy
import numpy as np
from models.base import Network, fc_layer
from keras.datasets import fashion_mnist,mnist
from optimizers.optimization_functions import adam, momentum_based_gradient_descent, nadam, nestrov_accelerated_gradient_descent, rmsprop, vanilla_gradient_descent
from utils.metrics import accuracy

def one_hot_encode(labels, num_classes):
  """convert output label into one hot encoded format"""
  return np.eye(num_classes)[labels]

def xavier(n_inputs,n_output):
  """Best for tanh and sigmoid
  introduced in Understanding the difficulty of training deep feedforward neural networks"""
  limit = np.sqrt(6 / (n_inputs + n_output))
  return np.random.uniform(-limit, limit, (n_inputs, n_output))

def mse_softmax_grad(y_hat,y_pred):
  """Gradient of softmax with squared error loss with respect to a_L the preactivaton of last layer
  """
  err=y_hat-y_pred
  sum=np.sum(err*y_hat,axis=1,keepdims=True)
  return y_hat*(err-sum)


def construct_network(inp_size:int,num_layers:int,layer_size:int,out_size:int,activation_f:str,weight_initialisation:str="random")->Network:
  """ Network constructor return network object which can be trained
  inp_size:int input dimension
  num_layers:int number of hidden layers
  layer_size:int number of nodes in each hidden layer
  out_size:int number of classes
  activation_f:str activation function can be sigmoid,tanh,ReLU
  weight_initialisation:str random initialization default Xavier availabel
  """
  np.random.seed(41)
  net=Network()
  if(activation_f=='sigmoid'):
    activation_fn=Sigmoid()
  elif(activation_f=='tanh'):
    activation_fn=Tanh()
  elif(activation_f=='ReLU'):
    activation_fn=ReLU()
  else:
    raise ValueError("Invalid activation function or not implimented yet")
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
  dataset_name:str name of the dataset can be fmnist or mnist
  """
  if(dataset_name=="fmnist"):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  elif(dataset_name=="mnist"):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
  else:
    raise ValueError("Invalid dataset name currently accepting values fmnist(fasion mnist) mnist")
  x_train_flat=x_train.reshape(x_train.shape[0],-1)/255.0
  x_test_flat=x_test.reshape(x_test.shape[0],-1)/255.0
  one_hot_y_train=one_hot_encode(y_train,np.unique(y_train).shape[0])
  one_hot_y_test=one_hot_encode(y_test,np.unique(y_test).shape[0])
  return x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test

def train_loop(x_train_flat,y_train,x_test_flat,y_test,max_iter:int,model:Network,batch_size:int=None,beta=0.0,loss_fn=CrossEntropy,eta=0.01,print_iter=10,lmda=0.0,grad_algo=None):
  """ Training loop for manual training of netowk for classification task
  inputs
  max_iter(number of epoch),model(network being trained),batch_size,
  beta momentum parameter
  loss_fn CrossEntropy or MSE
  eta learnng rate
  print_iter how often the info need to be printed
  lamda weight decay parameter
  grad_algo if not specified vanila gradient descent else correspongding algorithm

  """
  n_samples=x_train_flat.shape[0]
  y_train_one_hot=one_hot_encode(y_train, np.unique(y_train).shape[0])
  if batch_size is None:
      batch_size=n_samples

  for epoch in range(max_iter):
      permutation=np.random.permutation(n_samples)
      x_shuffled=x_train_flat[permutation]
      y_shuffled=y_train_one_hot[permutation]
      epoch_loss=0.0
      num_batches=0

      for i in range(0, n_samples, batch_size):
          batch_end=min(i + batch_size, n_samples)
          X_batch=x_shuffled[i:batch_end]
          Y_batch=y_shuffled[i:batch_end]
          model.zero_grad()
          
          if grad_algo is None or grad_algo=="sgd":
              batch_loss=vanilla_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
          elif grad_algo=="momentum":
              batch_loss=momentum_based_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
          elif grad_algo=="nesterov":
              batch_loss=nestrov_accelerated_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
          elif grad_algo=="rmsprop":
              batch_loss=rmsprop(X_batch,Y_batch,model,loss_fn,eta,beta,lmda=lmda)
          elif grad_algo=="adam":
              batch_loss=adam(X_batch,Y_batch,model,loss_fn,eta,beta1=0.9,beta2=0.999,lmda=lmda)
          elif grad_algo=="nadam":
              batch_loss=nadam(X_batch,Y_batch,model,loss_fn,eta,beta1=0.9,beta2=0.999,lmda=lmda)
          else:
              raise ValueError("Unknown gradient algorithm or not implimented yet")
          
          epoch_loss+=batch_loss
          num_batches+=1

      epoch_loss /= num_batches

      y_pred=model.forward_pass_network(x_train_flat)
      train_acc=accuracy(y_train,y_pred)
      test_pred=model.forward_pass_network(x_test_flat)
      one_hot_y_test=one_hot_encode(y_test,np.unique(y_test).shape[0])
      val_loss=loss_fn(test_pred,one_hot_y_test)
      val_loss=np.mean(val_loss)
      val_acc=accuracy(y_test,test_pred)

      if epoch % print_iter == 0:
          print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Train Accuracy = {train_acc:.4f} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}")


