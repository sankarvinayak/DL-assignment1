import numpy as np
def accuracy(y_true,y_pred):
  """return fraction of the predictions are correct 
  y_true class label
  y_pred one hot encoded vector"""
  y_pred=np.argmax(y_pred,axis=1)
  return np.mean(y_pred==y_true)
