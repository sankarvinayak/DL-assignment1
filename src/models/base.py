from activations.activation_functions import ActivationFn, Sigmoid
import numpy as np
from activations import *
from utils.helper import xavier
np.random.seed(41)

np.random.seed(41)
class fc_layer:
  def __init__(self,n_inputs:int,n_output:int,activation_fn:ActivationFn=Sigmoid(),initialization="random"):
    """Initialize the values of the network like weights,bias,momentum values u,sum values s etc
    inputs
    n_inputs:int input size
    n_output:int output size
    activation_fn:ActivationFn defalut sigmoid can be changed to tanh or ReLU
    initialization random by default can be changed to Xavier
    """
    if initialization=="Xavier":
      self.weights=xavier(n_inputs,n_output)
    else:
      self.weights=np.random.randn(n_inputs, n_output)*0.01 # reducing the random wright range empirically improve the initial loss
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
    # print(temp.shape)
    for layer in self.layers:
        temp=layer.forward_pass(temp)
    self.output=temp
    return self.output
  def calculate_grad(self,Y_hat,Y_true):
    """takes y_hat and y_true as input
    calculate the gradient from last to the first layer using the the way taught in lecture"""
    batch_size=Y_true.shape[0]
    grad_aL=-(Y_true-Y_hat)/batch_size
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
