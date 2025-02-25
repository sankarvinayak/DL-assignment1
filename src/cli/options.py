import argparse
def get_args():
  parser = argparse.ArgumentParser(
      description="Train a neural network with specified hyperparameters and options."
  )
  parser.add_argument( '-wp', '--wandb_project',type=str,default='myprojectname')
