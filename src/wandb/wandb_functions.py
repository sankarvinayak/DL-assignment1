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
    np.random.seed(41)
    model=construct_network(inp_size=inp_shp,num_layers=num_hidden_layers,layer_size=hidden_layer_size,out_size=out_shp,activation_f=activation_function,weight_initialisation=weight_initialisation)

    n_samples=x_train_flat.shape[0]
    loss_fn=CrossEntropy
    n_samples = x_train_flat.shape[0]

    for epoch in range(num_epochs):
        permutation = np.random.permutation(n_samples)
        X_shuffled = x_train_flat[permutation]
        Y_shuffled = one_hot_y_train[permutation]
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:batch_end]
            Y_batch = Y_shuffled[i:batch_end]
            model.zero_grad()
            if optimizer =="sgd" :
                batch_loss = vanilla_gradient_descent(X_batch, Y_batch, model, loss_fn,  eta=lr,lmda=lmda)
            elif optimizer == "momentum":
                batch_loss = momentum_based_gradient_descent(X_batch, Y_batch, model, loss_fn,  eta=lr,lmda=lmda)
            elif optimizer == "nesterov":
                batch_loss = nestrov_accelerated_gradient_descent(X_batch, Y_batch, model, loss_fn,  eta=lr,lmda=lmda)
            elif optimizer == "rmsprop":
                batch_loss = rmsprop(X_batch, Y_batch, model, loss_fn, eta=lr,  lmda=lmda)
            elif optimizer == "adam":
                batch_loss = adam(X_batch, Y_batch, model, loss_fn, eta=lr, lmda=lmda)
            elif optimizer == "nadam":
                batch_loss = nadam(X_batch, Y_batch, model, loss_fn, eta=lr,  lmda=lmda)

            epoch_loss += batch_loss
            num_batches += 1

        epoch_loss /= num_batches
        y_pred = model.forward_pass_network(x_train_flat)
        train_acc = accuracy(y_train, y_pred)
        test_pred = model.forward_pass_network(x_test_flat)
        val_loss=loss_fn(test_pred,one_hot_y_test)
        val_loss=np.mean(val_loss)
        val_acc=accuracy(y_test,test_pred)

        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Train Accuracy = {train_acc:.4f} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}")
        wandb.log({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "epoch": epoch+1
        })
  # wandb.finish()