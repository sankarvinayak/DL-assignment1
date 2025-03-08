# import keras
import wandb
from ..activations.activation_functions import Linear, ReLU, Sigmoid, Softmax, Tanh
from ..loss.loss_functions import CrossEntropy
import numpy as np
from ..models.base import Network, fc_layer, one_hot_encode
from keras.datasets import fashion_mnist,mnist
from ..optimizers.optimization_functions import adam, momentum_based_gradient_descent, nadam, nestrov_accelerated_gradient_descent, rmsprop, vanilla_gradient_descent
from ..utils.metrics import accuracy
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

def construct_network(inp_size:int,num_layers:int,layer_size:int,out_size:int,activation_f:str,weight_initialisation:str="random")->Network:
  """ Network constructor return network object which can be trained
  inp_size:int input dimension
  num_layers:int number of hidden layers
  layer_size:int number of nodes in each hidden layer
  out_size:int number of classes
  activation_f:str activation function can be sigmoid,tanh,ReLU
  weight_initialisation:str random initialization default Xavier availabel
  """
  np.random.seed(41)
  net=Network()
  if(activation_f=='sigmoid'):
    activation_fn=Sigmoid()
  elif(activation_f=='tanh'):
    activation_fn=Tanh()
  elif(activation_f=='ReLU'):
    activation_fn=ReLU()
  elif(activation_f=='identity'):
     activation_fn=Linear()
  else:
    raise ValueError("Invalid activation function or not implimented yet")
  net.append_layer(fc_layer(n_inputs=inp_size, n_output=layer_size, activation_fn=activation_fn,initialization=weight_initialisation))
  for i in range(num_layers-1): #there was a one off error here
    net.append_layer(fc_layer(n_inputs=layer_size, n_output=layer_size, activation_fn=activation_fn,initialization=weight_initialisation))
  net.append_layer(fc_layer(n_inputs=layer_size, n_output=out_size, activation_fn=Softmax(),initialization=weight_initialisation))
  return net


def get_data(dataset_name:str="fmnist")->tuple:
  """
  get the data from teh dataset using the api
  take dataset name can be fmnist for fasion mnist
  return x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test as tuple
  dataset_name:str name of the dataset can be fmnist or mnist
  """
  if(dataset_name=="fmnist"):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  elif(dataset_name=="mnist"):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
  else:
    raise ValueError("Invalid dataset name currently accepting values fmnist(fasion mnist) mnist")
  x_train_flat=x_train.reshape(x_train.shape[0],-1)/255.0
  x_test_flat=x_test.reshape(x_test.shape[0],-1)/255.0
  one_hot_y_train=one_hot_encode(y_train,np.unique(y_train).shape[0])
  one_hot_y_test=one_hot_encode(y_test,np.unique(y_test).shape[0])
  return x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test

def mpl2plotly(cmap, entries=10):
    step=1.0/(entries-1); cs=[]
    for k in range(entries):
        clr=cmap(k*step)
        cs.append([k*step, f'rgb({int(clr[0]*255)},{int(clr[1]*255)},{int(clr[2]*255)})'])
    return cs
def create_conf_mat(y_true:list,y_pred:list,class_labels:list,title="Confusion Matrix"):
    n=len(class_labels)
    cm=np.zeros((n,n),dtype=int)
    for t,p in zip(y_true,y_pred): cm[int(t),int(p)] += 1
    ps=mpl2plotly(sns.color_palette("mako",as_cmap=True),10)
    ht=list()
    for i in range(n):
        row=[]
        for j in range(n):
            val=cm[i,j]
            if i==j: msg=f"correctly predicted {class_labels[j]}: {val}"
            else: msg=f"class {class_labels[i]} predicted as {class_labels[j]}: {val}"
            row.append(msg)
        ht.append(row)
    fig=go.Figure(data=go.Heatmap(z=cm,x=class_labels,y=class_labels,text=ht,hovertemplate='%{text}<extra></extra>',colorscale=ps,hoverongaps=False,showscale=True))
    for i in range(n):
        for j in range(n):
            fc="white" if cm[i,j]>np.max(cm)/2 else "black"
            fig.add_annotation(x=class_labels[j],y=class_labels[i],text=str(cm[i,j]),showarrow=False,font=dict(color=fc))
    fig.update_layout(title=title,xaxis_title="Predicted Label",yaxis_title="True Label",xaxis=dict(tickmode='array',tickvals=class_labels),yaxis=dict(tickmode='array',tickvals=class_labels),template='plotly_white',font=dict(family="Arial",size=12),margin=dict(l=40,r=40,t=40,b=40))
    return fig





def train_model(dataset_name,num_hidden_layers,hidden_layer_size,num_epochs,activation_function,weight_initialisation,batch_size,optimizer,lr,lmda,momentum=0.9,beta=0.9,beta1=0.9,beta2=0.999,epsilon=0.000001,logging=False):
    np.random.seed(41)
    x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test=get_data(dataset_name=dataset_name)
    n_samples=x_train_flat.shape[0]
    indices=np.random.permutation(n_samples)
    train_cutoff=int(0.9*n_samples)
    train_ids,val_ids=indices[:train_cutoff],indices[train_cutoff:]
    x_train_new=x_train_flat[train_ids]
    one_hot_y_train_new=one_hot_y_train[train_ids]
    y_train_new=y_train[train_ids]
    x_val=x_train_flat[val_ids]
    one_hot_y_val=one_hot_y_train[val_ids]
    y_val=y_train[val_ids]

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
                batch_loss=momentum_based_gradient_descent(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda,beta=momentum)
            elif optimizer=="nesterov":
                batch_loss=nestrov_accelerated_gradient_descent(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda,beta=momentum)
            elif optimizer=="rmsprop":
                batch_loss=rmsprop(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda,beta=beta)
            elif optimizer=="adam":
                batch_loss=adam(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda,beta1=beta1,beta2=beta2,epsilon=epsilon)
            elif optimizer=="nadam":
                batch_loss=nadam(X_batch,Y_batch,model,loss_fn,eta=lr,lmda=lmda,beta1=beta1,beta2=beta2,epsilon=epsilon)

            epoch_loss+=batch_loss
            num_batches+=1

        epoch_loss/=num_batches
        y_pred_train=model.forward_pass_network(x_train_new)
        train_acc=accuracy(y_train_new, y_pred_train)
        y_pred_val=model.forward_pass_network(x_val)
        val_loss=np.mean(loss_fn(y_pred_val, one_hot_y_val))
        val_acc=accuracy(y_val, y_pred_val)

        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Train Accuracy = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        if logging:
            wandb.log({"train_accuracy":train_acc,"val_accuracy":val_acc,"train_loss":epoch_loss,"val_loss":val_loss,"epoch":epoch+1})
    if logging:
      y_pred=np.argmax( model.forward_pass_network(x_test_flat),axis=1)
      if dataset_name=='fmnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
      elif dataset_name=='mnist':
        class_names=[str(i) for i in  range(10)]

      fig=create_conf_mat(y_test,y_pred,class_names,"Confusion Matrix") 
      wandb.log({"plot": wandb.Plotly(fig)})
      
       
    return model







def train_loop(x_train_flat,y_train,x_test_flat,y_test,max_iter:int,model:Network,batch_size:int=None,beta=0.0,loss_fn=CrossEntropy,eta=0.01,print_iter=10,lmda=0.0,grad_algo=None):
  """ Training loop for manual training of netowk for classification task
  inputs
  max_iter(number of epoch),model(network being trained),batch_size,
  beta momentum parameter
  loss_fn CrossEntropy or MSE
  eta learnng rate
  print_iter how often the info need to be printed
  lamda weight decay parameter
  grad_algo if not specified vanila gradient descent else correspongding algorithm

  """
  n_samples=x_train_flat.shape[0]
  y_train_one_hot=one_hot_encode(y_train, np.unique(y_train).shape[0])
  if batch_size is None:
      batch_size=n_samples

  for epoch in range(max_iter):
      permutation=np.random.permutation(n_samples)
      x_shuffled=x_train_flat[permutation]
      y_shuffled=y_train_one_hot[permutation]
      epoch_loss=0.0
      num_batches=0

      for i in range(0, n_samples, batch_size):
          batch_end=min(i + batch_size, n_samples)
          X_batch=x_shuffled[i:batch_end]
          Y_batch=y_shuffled[i:batch_end]
          model.zero_grad()
          
          if grad_algo is None or grad_algo=="sgd":
              batch_loss=vanilla_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
          elif grad_algo=="momentum":
              batch_loss=momentum_based_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
          elif grad_algo=="nesterov":
              batch_loss=nestrov_accelerated_gradient_descent(X_batch,Y_batch,model,loss_fn,beta,eta,lmda=lmda)
          elif grad_algo=="rmsprop":
              batch_loss=rmsprop(X_batch,Y_batch,model,loss_fn,eta,beta,lmda=lmda)
          elif grad_algo=="adam":
              batch_loss=adam(X_batch,Y_batch,model,loss_fn,eta,beta1=0.9,beta2=0.999,lmda=lmda)
          elif grad_algo=="nadam":
              batch_loss=nadam(X_batch,Y_batch,model,loss_fn,eta,beta1=0.9,beta2=0.999,lmda=lmda)
          else:
              raise ValueError("Unknown gradient algorithm or not implimented yet")
          
          epoch_loss+=batch_loss
          num_batches+=1

      epoch_loss /= num_batches

      y_pred=model.forward_pass_network(x_train_flat)
      train_acc=accuracy(y_train,y_pred)
      test_pred=model.forward_pass_network(x_test_flat)
      one_hot_y_test=one_hot_encode(y_test,np.unique(y_test).shape[0])
      val_loss=loss_fn(test_pred,one_hot_y_test)
      val_loss=np.mean(val_loss)
      val_acc=accuracy(y_test,test_pred)

      if epoch % print_iter == 0:
          print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Train Accuracy = {train_acc:.4f} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}")


