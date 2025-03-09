import numpy as np


from ..utils.helper import construct_network, get_data, train_loop


def main():
    #Creates a network and train it as a demo
    np.random.seed(41) # help in reproducability
    x_train_flat,y_train,x_test_flat,y_test,one_hot_y_train,one_hot_y_test=get_data(dataset_name="fmnist")
    inp_shp=x_train_flat.shape[1]
    out_shp=np.unique(y_train).shape[0] 
    
    model=construct_network(inp_size=inp_shp,num_layers=3,layer_size=32,out_size=out_shp) # Create a 1 input layer 2 hidden layer 1 output layer network due to the softmax layer will give output probabilities
    print(model.forward_pass_network(x_train_flat)) #return probability over 10 classes
    
    train_loop(x_train_flat,y_train,max_iter=5,model=model,print_iter=1)





if __name__ == '__main__':
    main()
