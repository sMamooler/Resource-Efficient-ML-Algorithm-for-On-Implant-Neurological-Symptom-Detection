import torch
from sklearn.cluster import KMeans
import torch
import numpy as np
import torch.nn as nn
from model import LSTM
import copy
import tqdm



def quantize_network(net):
    """
    Function that quantizes the netwrok parameters by replacing them with the index of their coresponding cluster

    Parameters
    ----------
    net: LSTM
        LSTM netowrk to be quantized
    Returns
    -------
    net: LSTM
        network with quantized parameters
    """
    for item in net.state_dict().items():
        weight_name = item[0]
        weights_mat = item[1]
        
        weights_quantized_vect, centroids = net.quantized_state_dict[weight_name]
        mat = weights_quantized_vect.reshape(weights_mat.shape)

        #save the centroids
        net.centroids[weight_name] = centroids
        #change the weights
        net.state_dict()[weight_name].copy_(mat)
    
    
    return net
            
def compute_quantized_weights(net, k):
    """
    Function that uses clusters network parameters into bins and saves the indices and cluster centroids in network's quantized_state_dict

    Parameters
    ----------
    net: LSTM
        LSTM netowrk to be quantized
    k : int
        numbe of clusters
    Returns
    -------
    net: LSTM
        network with non-empty quantized_state_dict
    """

    
    for item in net.state_dict().items():
        weight_name = item[0]
        weights_mat = item[1]
       
        weights_vect = torch.flatten(weights_mat).reshape(-1, 1)

        k = min(k,weights_vect.shape[0])
    
        #perform k-mean clustering on the weights
        kmeans = KMeans(n_clusters=k, random_state=0).fit(weights_vect)
        cluster_ids_x, cluster_centers = kmeans.labels_, kmeans.cluster_centers_ 
        cluster_ids_x, cluster_centers = torch.ByteTensor(cluster_ids_x), torch.Tensor(cluster_centers)

        
        net.quantized_state_dict[weight_name] = (cluster_ids_x, cluster_centers[:,0])

        ## uncomment to see the indices and centroids for each layer:
        # print("layer name: {n}".format(n=weight_name))
        # print("cluster indices:{c}".format(c=cluster_ids_x))
        # print("cluster centroids: {i}".format(i=cluster_centers[:,0]))

    return net
        


def quantized_train(TrainX, TrainY, TestX, TestY, net, lossfunc, optimizer, num_epoch, clip = 5, Finger=0):
    """
    Function that tunes the centroids using gradients of original network parameters

    Parameters
    ----------
    TrainX, TestX: numpy array of shape [#datapoints, seq_len, input_dim]
        contain feature vectors of train and test data
    TrainY, TestY: numpy array of shape [#datapoints, 1]
        contain labels of train and test data
    net: LSTM
        the network to be trained
    lossfunc: function
        the cost function to optimize
    optimizer: 
        optimizer used for training
    num_epoch: int
        number of epochs to train
    clip: int
        used for gradiant clipping
    Finger: int (0,1,2,3,4)
        finger index
    Returns
    -------
    list_corr_train: list
        contains correlation coefficient between prediction and labels of train data at every epoch
    list_corr_val: list
        contains correlation coefficient between prediction and labels of validation data at every epoch
    list_corr_test: list
        contains correlation coefficient between prediction and labels of test data at every epoch

    """
    seq_len = TrainX.shape[1]
    train_length = int(0.8*len(TrainX))
    val_length = len(TrainX)-train_length
    train_data = TrainX[:train_length]
    val_data = TrainX[train_length:]
    train_label = TrainY[:train_length]
    val_label = TrainY[train_length:]
    list_corr_train = []
    list_corr_val = []
    list_corr_test = []

    print("Tuning Centroids...")
    pbar = tqdm.tqdm(total=num_epoch, desc='Finger '+str(Finger), position=0, ascii=True)
    for epoch in range(num_epoch):
        
        

        #prepare train, validation, and test data
        x = torch.from_numpy(train_data).float()
        y = torch.from_numpy(train_label).float()
        xv = torch.from_numpy(val_data).float()
        yv = torch.from_numpy(val_label).float()
        xt = torch.from_numpy(TestX).float()
        yt = torch.from_numpy(TestY).float()

        # initialize hidden state 
        h = net.init_hidden(train_data.shape[0])
        hv = net.init_hidden(val_data.shape[0])
        ht = net.init_hidden(TestX.shape[0])

        # compute model predictions and loss
        pred, h = net(x, h)
        loss = lossfunc(pred[-1,:,:], y)
       
        # do a backward pass and a gradient update step
        optimizer.zero_grad()
        loss.backward()
        ## gradient clipping - prevents gradient explosion 
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        #update the centroids using the gradiant of the original weights:
        params = net.state_dict(keep_vars=True)
        lr= 0.02
        for item in net.quantized_state_dict.items():
            weight_name = item[0]
            weights_quantized_vect, centroids = item[1][0], item[1][1]
            
            
            weights_mat = params[weight_name]
            weight_grad_mat = weights_mat.grad
        
            weight_grad_vect = torch.flatten(weight_grad_mat)
            centroid_grad = []
      
            for k in range(centroids.shape[0]):
                grad_sum = torch.sum(weight_grad_vect[weights_quantized_vect==k])
                centroid_grad.append(grad_sum)
            centroid_grad = torch.Tensor(centroid_grad)
        
            net.quantized_state_dict[weight_name]= (weights_quantized_vect, centroids-lr*centroid_grad)
            
    
        pbar.update(1)

    return list_corr_train, list_corr_val, list_corr_test
   
    



