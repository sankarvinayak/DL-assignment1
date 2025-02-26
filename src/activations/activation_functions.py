from abc import abstractmethod
import numpy as np
class ActivationFn:
  @abstractmethod
  def forward(self,X):
      pass
  @abstractmethod
  def grad(self,X):
      pass


# class Sigmoid(ActivationFn):
#   def forward(self, X):
    # """small modification on the implimentation of sigmoid to avoid overflow encountered in exp better numerical stability"""
    # return np.where(X >= 0, 1/(1 + np.exp(-X)), np.exp(X)/(1 + np.exp(X)))
      # return 1/(1 + np.exp(-X))
class Sigmoid(ActivationFn):
  def forward(self, X):
      out = np.empty_like(X)
      pos_mask = X >= 0
      neg_mask = ~pos_mask
      out[pos_mask] = 1 / (1 + np.exp(-X[pos_mask]))
      out[neg_mask] = np.exp(X[neg_mask]) / (1 + np.exp(X[neg_mask]))
      return out
  def grad(self,X):
      forward_x = self.forward(X)
      return forward_x*(1 - forward_x)
  def grad(self,X):
      forward_x = self.forward(X)
      return forward_x*(1 - forward_x)


class Tanh(ActivationFn):
  def forward(self,X):
    return np.tanh(X)
  def grad(self,X):
    return 1-self.forward(X)**2


class ReLU(ActivationFn):
  def forward(self,X):
    return np.maximum(0,X)
  def grad(self,X):
    return np.where(X > 0, 1, 0)


class Softmax(ActivationFn):
  def forward(self,X):
    X=X-np.max(X,axis=1,keepdims=True)
    # print(np.sum(X))
    exps=np.exp(X)
    return exps/np.sum(exps,axis=1,keepdims=True)
  def grad():
    pass


class Linear(ActivationFn):
  def forward(self,X):
    return X
  def grad():
    return 1
