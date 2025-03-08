# Training and evaluating the model
Disclaimer: The models and code are primarily intended for clasification only, slight modificaion needed to work with regression


Once the network and dataset are setup one can train the model.

One can make use of the `train_loop` function passing the model and other hyper parameter which will automatically handle the training or one can define their own training loop easily
```
train_loop(x_train_flat,y_train,x_test_flat,y_test,max_iter:int,model:Network,batch_size:int=None,beta=0.0,loss_fn=CrossEntropy,eta=0.01,print_iter=10,lmda=0.0,grad_algo=None)->None:
```
## Loss function 
### Cross entropy
Sum(- Px logQx) which is commonly used with the classification algorithms give how surprised are we given the output and labels
### Mean squared error
Mean((actual-prediciton)^2) is a common easier version says how much the data point differs from the other
## Optimizers
There are variety of optimization function included in this project

All of them are expecting the input as the batch data,rather than handling on therir own
#### SGD
This is the real old Gradient descent 
```
vanilla_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
```
#### Momentum
This is Gradient descent with momentum which in baground call the `vanilla_gradient_descent` with a non zero beta(momentum) value
```
momentum_based_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
```
### Nestrov accelerated Gradient descent
Try to reduce the U terns of momentum by calculating the momentum at the look ahed point
```
nestrov_accelerated_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
```
### RMSProp
Adjust the leaning rate for datapoint in direction which have more sparse data
```
rmsprop(X_batch,Y_batch,model,loss_fn,eta,beta,lmda=lmda)
```
### Adam
A combination of momentum as well as the adaptive learning rates
```
adam(X_batch,Y_batch,model,loss_fn,eta,beta1=0.9,beta2=0.999,lmda=lmda)
```
### Nadam
Modified version of adam
```
nadam(X_batch,Y_batch,model,loss_fn,eta,beta1=0.9,beta2=0.999,lmda=lmda)
```

All of these optimization function return the loss calculated for the dataset and make the updates

## Evaluation
The primary matric for evluation defined here is the accuracy as well as the loss function

`accuracy` function expectes the raw one hot encoded vectors(logits) and the correct list of labels and return the accuracy value
```
accuracy(y_val, y_pred_val)
```

### Visualizing
There are no visualization plot included in this as it make use of wandb for that.
#### Confusion matrix
One can pass the predicition, true value vector as well as the class names to the 
`create_conf_mat` function which will return a plotly object confusion matrix.
```
create_conf_mat(y_test,y_pred,class_names,"Confusion Matrix") 
```


[Wandb](wandb.md)