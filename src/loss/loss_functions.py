def MSE(prediction,actual):
  assert prediction.shape==actual.shape, "Shapes of prediction and actual must be the same"
  diff=prediction-actual
  sqr=diff**2
  return np.mean(sqr)
# def CrossEntropy(prediction,actual):
#   return -np.sum(actual*np.log(prediction)+1e-8,axis=1)
#   # return -np.log(prediction[np.arange(len(prediction)), np.argmax(actual, axis=1)])

def CrossEntropy(prediction, actual):
  """Cross entropy loss function sum of negative log likyhood added 1e-10 to aboid the error """
  return -np.sum(actual * np.log(prediction+1e-10), axis=1)
