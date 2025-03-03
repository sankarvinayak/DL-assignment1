from ..activations.activation_functions import ActivationFn, Sigmoid
import numpy as np
from ..activations import *
from ..loss.loss_functions import MSE, CrossEntropy

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

class fc_layer:
  def __init__(self,n_inputs:int,n_output:int,activation_fn:ActivationFn=Sigmoid(),initialization="random"):
    """Initialize the values of the network like weights,bias,momentum values u,sum values s etc
    inputs
    n_inputs:int input size
    n_output:int output size
    activation_fn:ActivationFn defalut sigmoid can be changed to tanh or ReLU
    initialization random by default can be changed to Xavier
    grad_w store gradient of weights
    grad_b store gradient of bias
    u_w stores the history for weights used in momentum and nestrov
    u_b stores the history for bias momentum and nestrov
    v_w stores the history for weights used in rmsprop
    v_b stores the history for bias momentum and rmsprop
    m_w stores the history for weights used in adam and nadam
    m_b stores the history for bias momentum and adam nadam
    """
    np.random.seed(41)
    if initialization=="Xavier":
      self.weights=xavier(n_inputs,n_output)
    elif initialization=="random":
      self.weights=np.random.randn(n_inputs, n_output)*0.01 # reducing the random wright range empirically improve the initial loss
    else:
      raise ValueError("Invalid initialization method or not yet implimented")
    self.bias=np.random.randn(1, n_output)
    self.grad_w=np.zeros_like(self.weights)
    self.grad_b=np.zeros_like(self.bias)
    self.u_w=np.zeros_like(self.weights)
    self.u_b=np.zeros_like(self.bias)
    self.v_w=np.zeros_like(self.weights)
    self.v_b=np.zeros_like(self.bias)
    self.m_w=np.zeros_like(self.weights)
    self.m_b=np.zeros_like(self.bias)
    self.activation_fn=activation_fn
  def forward_pass(self, h_prev):
    """ forward pass through this layer
    input output of previous layer or the input vector for the first layer"""
    self.h_prev=h_prev
    self.a_layer=np.dot(h_prev,self.weights)+self.bias
    self.h_this=self.activation_fn.forward(self.a_layer)
    return self.h_this
  def zeroGrad(self):
    """make all the gradient value to zero equivalent to optimizer.zero_grad of pytorch"""
    self.grad_w=np.zeros_like(self.weights)
    self.grad_b=np.zeros_like(self.bias)

class Network:
  def __init__(self):
    """Initialize with an empty list so that layers can be appended to it similar implimentation is availivale in tf and pytorch"""
    self.layers=[]
  def append_layer(self,layer:fc_layer):
    """
    input layer object of type fc_layer which is equivalnet to dense layer of tf or linear of pytorch
    used to append layers to the network
    """
    self.layers.append(layer)
  def forward_pass_network(self,X):
    """
    do the forward pass thruugh the network and return Y_hat
    """
    self.input=X
    temp=X
    for layer in self.layers:
        temp=layer.forward_pass(temp)
    self.output=temp
    return self.output
  def calculate_grad(self,Y_hat,Y_true,loss_fn=CrossEntropy):
    """takes y_hat and y_true as input
    calculate the gradient from last to the first layer using the the way taught in lecture"""
    batch_size=Y_true.shape[0]
    if loss_fn is MSE:
      grad_aL = mse_softmax_grad(Y_hat, Y_true)
    else:
      grad_aL = (Y_hat - Y_true) / batch_size
    grad_ak=grad_aL
    for k in reversed(range(len(self.layers))):
        layer=self.layers[k]
        if k==0:
            h_prev=self.input
        else:
            h_prev=self.layers[k-1].h_this
        layer.grad_w=np.dot(h_prev.T,grad_ak)
        layer.grad_b=np.sum(grad_ak,axis=0,keepdims=True)
        if k > 0:
            grad_h=np.dot(grad_ak, layer.weights.T)
            prev_layer=self.layers[k-1]
            grad_ak=grad_h*prev_layer.activation_fn.grad(prev_layer.a_layer)
  def zero_grad(self):
    """make all the gradient value to zero equivalent to optimizer.zero_grad of pytorch"""
    for layer in self.layers:
        layer.zeroGrad()