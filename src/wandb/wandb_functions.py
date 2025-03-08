from ..loss.loss_functions import CrossEntropy
from ..optimizers.optimization_functions import adam, momentum_based_gradient_descent, nadam, nestrov_accelerated_gradient_descent, rmsprop, vanilla_gradient_descent
from ..utils.helper import construct_network, get_data, train_loop, train_model
from ..utils.metrics import accuracy
import wandb
import numpy as np
from keras.datasets import fashion_mnist,mnist

import wandb


def wandb_log_sample_images(project: str, entity: str, dataset_name: str):
    if dataset_name.lower() == "fmnist":
        (x_train, y_train), _ = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name.lower() == "mnist":
        (x_train, y_train), _ = mnist.load_data()
        class_names = [str(i) for i in range(10)]
    else:
        raise ValueError("Invalid dataset name currently accepting values fmnist(fasion mnist) mnist")
    sample_images = {}
    for label in range(10):
        idx = np.where(y_train == label)[0][0]
        sample_images[label] = x_train[idx]
    list_log = []
    for label in sorted(sample_images.keys()):
        caption = class_names[label]
        im = sample_images[label]
        list_log.append(wandb.Image(im, caption=caption))
    wandb.log({f"{dataset_name} each class samples": list_log})
    




def wand_train(config=None):
  """Loop which takes input from wandb, runs training and evaluation, and sends logs to the server.
  training set is split into 90% train and 10% validation.
  use as wandb.agent(sweep_id, wand_train)"""
  with wandb.init(config=config):
    config = wandb.config

    num_hidden_layers = config.num_hidden_layers
    hidden_layer_size = config.hidden_layer_size
    lmda = config.weight_decay
    optimizer = config.optimizer
    weight_initialisation = config.weight_initialisation
    activation_function = config.activation_function
    lr = config.learning_rate
    num_epochs = config.epochs
    batch_size = config.batch_size
    dataset_name = config.dataset

    np.random.seed(41)
    x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test=get_data(dataset_name=dataset_name)
    n_samples=x_train_flat.shape[0]
    indices=np.random.permutation(n_samples)
    train_cutoff=int(0.9*n_samples)
    train_id,val_idx=indices[:train_cutoff],indices[train_cutoff:]
    x_train_new=x_train_flat[train_id]
    one_hot_y_train_new=one_hot_y_train[train_id]
    y_train_new=y_train[train_id]
    x_val=x_train_flat[val_idx]
    one_hot_y_val=one_hot_y_train[val_idx]
    y_val=y_train[val_idx]

    inp_shp=x_train_flat.shape[1]
    out_shp=np.unique(y_train).shape[0]
    model=construct_network(inp_size=inp_shp, num_layers=num_hidden_layers,layer_size=hidden_layer_size, out_size=out_shp,activation_f=activation_function,weight_initialisation=weight_initialisation)

    loss_fn = CrossEntropy
    n_train = x_train_new.shape[0]

    for epoch in range(num_epochs):
        permutation = np.random.permutation(n_train)
        X_shuffled = x_train_new[permutation]
        Y_shuffled = one_hot_y_train_new[permutation]
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, n_train, batch_size):
            batch_end = min(i + batch_size, n_train)
            X_batch = X_shuffled[i:batch_end]
            Y_batch = Y_shuffled[i:batch_end]
            model.zero_grad()
            if optimizer=="sgd":
                batch_loss=vanilla_gradient_descent(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda)
            elif optimizer=="momentum":
                batch_loss=momentum_based_gradient_descent(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda)
            elif optimizer=="nesterov":
                batch_loss=nestrov_accelerated_gradient_descent(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda)
            elif optimizer=="rmsprop":
                batch_loss=rmsprop(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda)
            elif optimizer=="adam":
                batch_loss=adam(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda)
            elif optimizer=="nadam":
                batch_loss=nadam(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda)

            epoch_loss+=batch_loss
            num_batches+=1

        epoch_loss/=num_batches
        y_pred_train=model.forward_pass_network(x_train_new)
        train_acc=accuracy(y_train_new, y_pred_train)
        y_pred_val=model.forward_pass_network(x_val)
        val_loss=np.mean(loss_fn(y_pred_val, one_hot_y_val))
        val_acc=accuracy(y_val, y_pred_val)

        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Train Accuracy = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        wandb.log({"train_accuracy":train_acc,"val_accuracy":val_acc,"train_loss":epoch_loss,"val_loss":val_loss,"epoch":epoch+1})


def wandb_run_experiment(args):
    print(args)
    wandb_entity=args.wandb_entity
    wandb_project=args.wandb_project
    dataset=args.dataset 
    epochs=args.epochs
    batch_size=args.batch_size 
    loss=args.loss
    optimizer=args.optimizer
    lr=args.learning_rate
    momentum=args.momentum 
    beta=args.beta 
    beta1=args.beta1
    beta2=args.beta2
    weight_init=args.weight_init
    epsilon=args.epsilon 
    lmda=args.weight_decay
    num_layers=args.num_layers
    hidden_size=args.hidden_size
    activation=args.activation
    config={}
    if dataset=="fashion_mnist":
        dataset='fmnist'
    config['dataset']=dataset
    config['epochs']=epochs
    config['batch_size']=batch_size
    config['loss']=loss
    config['lr']=lr
    config['optimizer']=optimizer
    if optimizer=="momentum" or optimizer=="nag":
        config['momentum']=momentum
    elif optimizer=="rmsprop":
        config['beta']=beta
    elif optimizer=='adam' or optimizer=='nadam':
        config["beta1"]=beta1
        config["beta2"]=beta2
        config["epsilon"]=epsilon
    config['weight_decay']=lmda
    config['num_layers']=num_layers
    config['hidden_size']=hidden_size
    config['activation']=activation

    run=wandb.init(entity=wandb_entity,project=wandb_project,config=config)
    wandb_log_sample_images(wandb_project,wandb_entity,dataset)
    np.random.seed(41)
    train_model(dataset_name=dataset,num_hidden_layers=num_layers,hidden_layer_size=hidden_size,num_epochs=epochs,activation_function=activation,batch_size=batch_size,optimizer=optimizer,lr=lr,lmda=lmda,momentum=momentum,beta=beta,beta1=beta1,beta2=beta2,epsilon=epsilon,logging=True)
    run.finish()

