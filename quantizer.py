import torch
import kmeans1d
import torch
import numpy as np
from Loss import corr_coeff, corr_coeff_loss
import torch.nn as nn
from model import LSTM
from torch.utils.data import Dataset, DataLoader
import copy



def quantize_network(net):
    for item in net.state_dict().items():
        weight_name = item[0]
        weights_mat = item[1]
        weights_quantized_vect, centroids = net.quantized_state_dict[weight_name]
        mat = weights_quantized_vect.reshape(weights_mat.shape)

        #save the centroids
        net.centroids[weight_name] = centroids
        #change the weights
        net.state_dict()[weight_name].copy_(mat)
    
    #free the quantized_state_dict
    net.quantized_state_dict = None
    return net
            
def compute_quantized_weights(net, k):

    for item in net.state_dict().items():
        weight_name = item[0]
        weights_mat = item[1]


        weights_vect = torch.flatten(weights_mat)
   
        #perform k-mean clustering on the weights
        cluster_ids_x, cluster_centers = kmeans1d.cluster(weights_vect,k)
        cluster_ids_x, cluster_centers = torch.ByteTensor(cluster_ids_x), torch.Tensor(cluster_centers)
        net.quantized_state_dict[weight_name] = (cluster_ids_x, cluster_centers)
        

def quantized_train_batch_train(batch, net, lossfunc, optimizer, lr, clip = 5):
    input, target = batch['input'], batch['target']
    
    x = input.float()
    y = target.float()
        
    # initialize hidden state 
    h_init = net.init_hidden(input.shape[0])
    
    #predict using the original model
    pred, h = net(x, h_init)
    loss = corr_coeff_loss(pred[-1,:,:], y)
    
    #optimize the originla weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    

    #update the centroids using the gradiant of the original weights:
    params = net.state_dict(keep_vars=True)
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
        
    # gradient clipping - prevents gradient explosion 
    #nn.utils.clip_grad_norm_(net.parameters(), clip)
   
    return pred, target, loss
        
def quantized_train(net, dataset, num_epoch=10, batch_size=32, lr=0.2):

    dataloader = DataLoader(dataset, batch_size, drop_last=True)
    lossfunc =  nn.L1Loss()
    optimizer = torch.optim.Adamax(net.parameters(),lr=0.002)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)
   
    
    for epoch in range(num_epoch):
        
        losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            pred, y, loss = quantized_train_batch_train(batch, net, lossfunc, optimizer, lr)
            losses.append(loss.item())
            if batch_idx==0:
                preds = pred
                ys = y
            else:
                preds = torch.cat((preds, pred), dim=1)
                ys = torch.cat((ys,y), dim=0)
              
            
        losses = np.array(losses)
       

        if (epoch+1)%5 == 0: #num_epoch-1:
            
            corrcoef = corr_coeff(preds[-1,:,:],ys).item() #np.corrcoef(preds, ys)
            # preds = preds[-1,:,:].detach().numpy().reshape((-1,))
            # ys = ys.detach().numpy().reshape((-1,))
            # np_corr_coeff = np.corrcoef(preds, ys)
            # print(np_corr_coeff)
            print ('Epoch [%d/%d], Average Batch Loss: %.4f' %(epoch+1, num_epoch, np.mean(losses)))
            print ('Correlation Coefficient : {corrcoef}'.format(corrcoef=corrcoef))
   

   



    

