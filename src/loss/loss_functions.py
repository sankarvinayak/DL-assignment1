import numpy as np

def MSE(prediction,actual):
  """Loss function Mean squared error
  Mean((prediction-actual)**2
  dimension of prediction and actual should be match commonly used for regression tasks(in this project used for classification also)
  """
  assert prediction.shape==actual.shape, "Shapes of prediction and actual must be the same"
  diff=prediction-actual
  sqr=diff**2
  return np.mean(sqr)

# def CrossEntropy(prediction,actual):
#   return -np.sum(actual*np.log(prediction)+1e-8,axis=1)
#   # return -np.log(prediction[np.arange(len(prediction)), np.argmax(actual, axis=1)])

def CrossEntropy(prediction, actual):
  """Cross entropy loss function sum of negative log likyhood added 1e-10 to avoid the error Commonly used for classification taks"""
  return -np.sum(actual * np.log(prediction+1e-10), axis=1)
