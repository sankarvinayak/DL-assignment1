def vanilla_gradient_descent(X_batch,Y_batch,model:Network,loss_fn,eta:float=0.1,beta:float=0,lmda:float=0):
  """
  Gradient descent default function for gradient descent weritten in such a way that momentum based GD can directly call
  X_batch input batch
  Y_batch output batch
  model:Network model to train on
  loss_fn loss function to use cross entropy in this case
  eta:float=0.1 learning rate
  beta:float=0.5 momentum
  lmda:float=0 decay parameter
  """
  Y_hat=model.forward_pass_network(X_batch)
  batch_loss_vec=loss_fn(Y_hat,Y_batch)
  batch_loss=np.mean(batch_loss_vec)
  model.calculate_grad(Y_hat,Y_batch)
  for layer in model.layers:
    layer.u_w=beta*layer.u_w+eta*(layer.grad_w+lmda*layer.weights)
    layer.u_b=beta*layer.u_b+eta*(layer.grad_b+lmda*layer.bias)
    layer.weights-=layer.u_w
    layer.bias-=layer.u_b
  return batch_loss

def momentum_based_gradient_descent(X_batch,Y_batch,model:Network,loss_fn,eta:float=0.1,beta:float=0.5,lmda:float=0):
  """
  Momentum based Gradient Descent 
  X_batch input batch
  Y_batch output batch
  model:Network model to train on
  loss_fn loss function to use cross entropy in this case
  eta:float=0.1 learning rate
  beta:float=0.5 momentum
  lmda:float=0 decay parameter
  Directly calls vanilla gradient descent with momentum
  """
  return vanilla_gradient_descent(X_batch=X_batch,Y_batch=Y_batch,model=model,loss_fn=loss_fn,eta=eta,beta=beta,lmda=lmda)

def nestrov_accelerated_gradient_descent(X_batch,Y_batch,model:Network,loss_fn,eta:float=0.1,beta:float=0.5,lmda:float=0):
  """
  Nestrov accelerated gradient descent (NAG)
  X_batch input batch
  Y_batch output batch
  model:Network model to train on
  loss_fn loss function to use cross entropy in this case
  eta:float=0.1 learning rate
  beta:float=0.5 momentum
  lmda:float=0 decay parameter
  """
  original_params = []
  for layer in model.layers:
    original_params.append((layer.weights.copy(), layer.bias.copy()))
  for layer in model.layers:
    layer.weights-= beta*layer.u_w
    layer.bias-= beta*layer.u_b
  Y_hat = model.forward_pass_network(X_batch)
  batch_loss_vec=loss_fn(Y_hat, Y_batch)
  batch_loss=np.mean(batch_loss_vec)
  model.calculate_grad(Y_hat, Y_batch)
  for id, layer in enumerate(model.layers):
    orig_w, orig_b=original_params[id]
    grad_w_with_decay=layer.grad_w+(lmda*orig_w)
    grad_b_with_decay=layer.grad_b+(lmda*orig_b)
    layer.u_w= (beta*layer.u_w)+(eta*grad_w_with_decay)
    layer.u_b= (beta*layer.u_b)+(eta*grad_b_with_decay)
    layer.weights= orig_w-layer.u_w
    layer.bias=orig_b-layer.u_b
  return batch_loss


def rmsprop(X_batch,Y_batch,model:Network,loss_fn,eta:float=0.1,beta:float=0.9,lmda:float=0,epsilon:float=0.000001):
  """
  RMSProp 
  X_batch input batch
  Y_batch output batch
  model:Network model to train on
  loss_fn loss function to use cross entropy in this case
  eta:float=0.1 learning rate
  beta:float=0.5 momentum
  lmda:float=0 decay parameter
  """
  Y_hat=model.forward_pass_network(X_batch)
  batch_loss_vec=loss_fn(Y_hat,Y_batch)
  batch_loss=np.mean(batch_loss_vec)
  model.calculate_grad(Y_hat,Y_batch)
  for layer in model.layers:
    grad_w=layer.grad_w+(lmda*layer.weights)
    grad_b=layer.grad_b+(lmda*layer.bias)
    layer.s_w=(beta*layer.s_w)+((1-beta)*(grad_w**2))
    layer.s_b=(beta*layer.s_b)+((1-beta)*(grad_b**2))
    layer.weights-=eta*(grad_w/(np.sqrt(layer.s_w)+epsilon))
    layer.bias-=eta*(grad_b/(np.sqrt(layer.s_b)+epsilon))
  return batch_loss

def adam(X_batch,Y_batch,model:Network,loss_fn,eta:float=0.1,beta1:float=0.9,beta2=0.999,lmda:float=0,epsilon:float=0.000001):
  """
  Adam optimizer 
  X_batch input batch
  Y_batch output batch
  model:Network model to train on
  loss_fn loss function to use cross entropy in this case
  eta:float=0.1 learning rate
  beta1:float=0.9 momentum
  beta2:float=0.9 momentum
  lmda:float=0 decay parameter
  """
  pass

def nadam(X_batch,Y_batch,model:Network,loss_fn,eta:float=0.1,beta1:float=0.9,beta2=0.999,lmda:float=0,epsilon:float=0.000001):
  """
  nAdam optimizer 
  X_batch input batch
  Y_batch output batch
  model:Network model to train on
  loss_fn loss function to use cross entropy in this case
  eta:float=0.1 learning rate
  beta1:float=0.9 momentum
  beta2:float=0.9 momentum
  lmda:float=0 decay parameter
  """
  pass

