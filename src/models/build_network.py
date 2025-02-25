def construct_network(inp_size:int,num_layers:int,layer_size:int,out_size:int,activation_f:str,weight_initialisation:str)->Network:
  """ Network constructor return network object which can be trained
  inp_size:int input dimension
  num_layers:int number of hidden layers
  layer_size:int number of nodes in each hidden layer
  out_size:int number of classes
  activation_f:str activation function can be sigmoid,tanh,ReLU
  weight_initialisation:str random initialization default Xavier availabel
  """
  net=Network()
  if(activation_f=='sigmoid'):
    activation_fn=Sigmoid()
  elif(activation_f=='tanh'):
    activation_fn=Tanh()
  elif(activation_f=='ReLU'):
    activation_fn=ReLU()
  net.append_layer(fc_layer(n_inputs=inp_size, n_output=layer_size, activation_fn=activation_fn,initialization=weight_initialisation))
  for i in range(num_layers-2):
    net.append_layer(fc_layer(n_inputs=layer_size, n_output=layer_size, activation_fn=activation_fn,initialization=weight_initialisation))
  net.append_layer(fc_layer(n_inputs=layer_size, n_output=out_size, activation_fn=Softmax(),initialization=weight_initialisation))
  return net
