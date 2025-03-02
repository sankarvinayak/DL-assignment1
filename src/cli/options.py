import argparse
def get_args():
  parser = argparse.ArgumentParser(
      description="Train a neural network with specified hyperparameters and options."
  )
  parser.add_argument( '-wp', '--wandb_project',type=str,default='6401_Assignment1')
  parser.add_argument( '-we', '--wandb_entity',type=str,default='cs24m041')
  parser.add_argument( '-d', '--dataset',type=str,default='fashion_mnist')
  parser.add_argument( '-e', '--epochs',type=int,default=1)
  parser.add_argument( '-b', '--batch_size',type=int,default=4)
  parser.add_argument( '-l', '--loss',type=str,default='cross_entropy')
  parser.add_argument( '-o', '--optimizer',type=str,default='sgd')
  parser.add_argument( '-lr', '--learning_rate',type=float,default=0.1)
  parser.add_argument( '-m', '--momentum',type=float,default=0.5)

  parser.add_argument( '-beta', '--beta',type=float,default=0.5)

  parser.add_argument( '-beta1', '--beta1',type=float,default=0.5)
  parser.add_argument( '-beta2', '--beta2',type=float,default=0.5)
  parser.add_argument( '-eps', '--epsilon',type=float,default=0.5)
  parser.add_argument( '-w_d', '--weight_decay',type=float,default=0)

  parser.add_argument( '-nhl', '--num_layers',type=int,default=1)

  parser.add_argument( '-sz', '--hidden_size',type=int,default=4)
  parser.add_argument( '-a', '--activation',type=str,default='sigmoid')



