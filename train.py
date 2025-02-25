from src.cli.options import get_args
def main():
  args = get_args()
  wandb_entity=args.wandb_entity
  wandb_project=args.wandb_project
  dataset=args.dataset
  epoch=args.epoch
  batch_size=args.batch_size
  loss_fn=args.loss_fn
  lr=args.lr
  momentum=args.momentum
  beta=args.beta
  beta1=args.beta1
  beta2=args.beta2
  epsilon=args.epsilon
  lmda=args.decay
  weight_init=args.weight_init
  optimizer=args.optimizer
  activation_fn=args.activation_fn
  num_hidden=args.num_hidden
  hidden_size=args.hidden_size

if __name__ == '__main__':
    main()
