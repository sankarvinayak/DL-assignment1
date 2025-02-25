def one_hot_encode(labels, num_classes):
  """convert output label into one hot encoded format"""
  return np.eye(num_classes)[labels]
