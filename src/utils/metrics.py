import numpy as np
def accuracy(y_true,y_pred):
  """return fraction of the precision are correct"""
  y_pred=np.argmax(y_pred,axis=1)
  return np.mean(y_pred==y_true)
