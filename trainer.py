import torch
import numpy as np
from Loss import corr_coeff, corr_coeff_loss
import torch.nn as nn
from model import LSTM
from torch.utils.data import Dataset, DataLoader, random_split


def batch_train(batch, net, lossfunc, optimizer, clip = 5):
    input, target = batch['input'], batch['target']
    
    # TODO: Step 1 - create torch variables corresponding to features and labels
        

    #x = TrainX.reshape([seq_len, TrainX.shape[0],TrainX.shape[1]])
    x = input.float()
    y = target.float()
        
    # initialize hidden state 
    h = net.init_hidden(input.shape[0])
   
    pred, h = net(x, h)
        
    #target = torch.reshape(y, (-1,)).long()
    
    loss = lossfunc(pred, y)
    #loss = corr_coeff_loss(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping - prevents gradient explosion 
    #nn.utils.clip_grad_norm_(net.parameters(), clip)
    optimizer.step()

    return pred, target, loss
        


def train(net, dataset, num_epoch=10, batch_size=32):
    train_length = int(0.8*len(dataset))
    val_length = len(dataset)-train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    train_dataloader = DataLoader(train_dataset, batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size, drop_last=True)
    
    lossfunc =  nn.MSELoss()#nn.L1Loss()
    #optimizer = torch.optim.Adamax(net.parameters(),lr=0.05)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
   
    for epoch in range(num_epoch):
        
        
        
        ################################train####################################
       
        losses = []
        for batch_idx, batch in enumerate(train_dataloader):
            net.train()
            pred, y, loss = batch_train(batch, net, lossfunc, optimizer)
            losses.append(loss.item())
            if batch_idx==0:
                preds = pred
                ys = y
            else:
                preds = torch.cat((preds, pred), dim=0)
                ys = torch.cat((ys,y), dim=0)
              
            
        losses = np.array(losses)
       

        if (epoch+1)%1 == 0: #num_epoch-1:
            
            #corrcoef = corr_coeff(preds[-1,:,:],ys).item() #np.corrcoef(preds, ys)
            preds = preds.detach().numpy().reshape((-1,))
            ys = ys.detach().numpy().reshape((-1,))
            np_corr_coeff = np.corrcoef(preds, ys)
            # print(np_corr_coeff)
            print ('Epoch [%d/%d]:' %(epoch+1, num_epoch))
            print('Total Loss: %.4f' %(np.mean(losses)))
            print ('Correlation Coefficient : {corrcoef}'.format(corrcoef=np_corr_coeff[0][1]))
        
        #####################################validation#####################################
        val_losses = []
        val_pred = []
        
        for batch_idx, batch in enumerate(val_dataloader):

            with torch.no_grad():
                net.eval()
                input, target = batch['input'], batch['target']
        
                # TODO: Step 1 - create torch variables corresponding to features and labels
                    

                #x = TrainX.reshape([seq_len, TrainX.shape[0],TrainX.shape[1]])
                x = input.float()
                y = target.float()
                    
                # initialize hidden state 
                h = net.init_hidden(input.shape[0])
                pred, h = net(x, h)
                val_loss = lossfunc(pred, y)
                val_losses.append(val_loss.item())

                if batch_idx==0:
                    preds = pred
                    ys = y
                else:
                    preds = torch.cat((preds, pred), dim=0)
                    ys = torch.cat((ys,y), dim=0)
            #loss = corr_coeff_loss(pred[-1,:,:], y)
        if (epoch+1)%1 == 0:   
            #corrcoef = corr_coeff(preds[-1,:,:],ys).item()
            preds = preds.detach().numpy().reshape((-1,))
            ys = ys.detach().numpy().reshape((-1,))
            np_corr_coeff = np.corrcoef(preds, ys)
            print ('Total Validation Loss: %.4f' %(np.mean(val_losses)))
            print ('Validation Correlation Coefficient : {corrcoef}'.format(corrcoef=np_corr_coeff[0][1]))
