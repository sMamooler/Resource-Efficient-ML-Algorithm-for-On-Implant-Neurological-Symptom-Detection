import torch
import torch.nn as nn
import numpy as np
import tqdm


 

def train(TrainX, TrainY, TestX, TestY, net, lossfunc, optimizer, num_epoch = 60, clip = 5, Finger = 0):

    """
    Function that trains the netwok

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
        
        
    

        #compute corelation coefficient between predictions and labels 
        corrcoef = np.corrcoef(pred[-1,:,:].detach().numpy().reshape((-1,)),y.detach().numpy().reshape((-1,)))
        list_corr_train += [corrcoef[0,1]]
        print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epoch, loss.item()))
        print ('Correlation coefficient train : {corrcoef}'.format(corrcoef=corrcoef[0,1]))

        net.eval()
        with torch.no_grad():
            predv, hv = net(xv, hv)
            corrcoefv = np.corrcoef(predv[-1,:,:].detach().numpy().reshape((-1,)),yv.detach().numpy().reshape((-1,)))
            list_corr_val += [corrcoefv[0,1]]
            print ('Correlation coefficient validation: {corrcoef}'.format(corrcoef=corrcoefv[0,1]))

            predt, ht = net(xt, ht)
            corrcoeft = np.corrcoef(predt[-1,:,:].detach().numpy().reshape((-1,)),yt.detach().numpy().reshape((-1,)))
            list_corr_test += [corrcoeft[0,1]]
            
            print ('Correlation coefficient test: {corrcoef}'.format(corrcoef=corrcoeft[0,1]))

    return list_corr_train, list_corr_val, list_corr_test
  
  
    

