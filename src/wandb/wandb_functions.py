from loss.loss_functions import CrossEntropy
from optimizers.optimization_functions import adam, momentum_based_gradient_descent, nadam, nestrov_accelerated_gradient_descent, rmsprop, vanilla_gradient_descent
from utils.helper import construct_network, get_data
from utils.metrics import accuracy
import wandb
import numpy as np
from keras.datasets import fashion_mnist,mnist

def wand_train(config=None):
  """loop which takes input from the wanb and run the model evalueate and send log to server"""

  with wandb.init(config=config):

    config = wandb.config

    num_hidden_layers=config.num_hidden_layers
    hidden_layer_size=config.hidden_layer_size
    lmda=config.weight_decay
    optimizer=config.optimizer
    weight_initialisation=config.weight_initialisation
    activation_function=config.activation_function
    lr=config.learning_rate
    num_epochs=config.epochs
    batch_size=config.batch_size
    dataset_name=config.dataset

    x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test=get_data(dataset_name=dataset_name)
    inp_shp=x_train_flat.shape[1]
    out_shp=np.unique(y_train).shape[0]
    net=construct_network(inp_size=inp_shp,num_layers=num_hidden_layers,layer_size=hidden_layer_size,out_size=out_shp,activation_f=activation_function,weight_initialisation=weight_initialisation)

    n_samples=x_train_flat.shape[0]
    if optimizer=='sgd':
      batch_size=1
    loss_fn=CrossEntropy

    for epoch in range(num_epochs):

      permutation=np.random.permutation(n_samples)
      x_train=x_train_flat[permutation]
      one_hot_y_train=one_hot_y_train[permutation]
      train_loss=0
      num_batches=0

      for i in range(0, n_samples, batch_size):
        batch_end=min(i + batch_size, n_samples)
        x_batch=x_train[i:batch_end]
        y_batch=one_hot_y_train[i:batch_end]

        net.zero_grad()
        if optimizer=='sgd':
          batch_loss=vanilla_gradient_descent(x_batch,y_batch,net,loss_fn,eta=lr,lmda=lmda)
        elif optimizer=='momentum':
          batch_loss=momentum_based_gradient_descent(x_batch,y_batch,net,loss_fn,eta=lr,lmda=lmda)
        elif optimizer=='nesterov':
          batch_loss=nestrov_accelerated_gradient_descent(x_batch,y_batch,net,loss_fn,eta=lr,lmda=lmda)
        elif optimizer=='rmsprop':
          batch_loss=rmsprop(x_batch,y_batch,net,loss_fn,eta=lr,lmda=lmda)
        elif optimizer=='adam':
          batch_loss=adam(x_batch,y_batch,net,loss_fn,eta=lr,lmda=lmda)
        elif optimizer=='nadam':
          batch_loss=nadam(x_batch,y_batch,net,loss_fn,eta=lr,lmda=lmda)

        train_loss+=batch_loss
        num_batches+=1

      train_loss/=num_batches
      val_loss=loss_fn(net.forward_pass_network(x_test_flat),one_hot_y_test)
      val_loss=np.mean(val_loss)
      y_pred=net.forward_pass_network(x_train_flat)
      train_acc=accuracy(y_train,y_pred)
      y_pred=net.forward_pass_network(x_test_flat)
      val_acc=accuracy(y_test,y_pred)

      wandb.log({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch+1
        })
  wandb.finish()
    
