import numpy as np
import math 
import torch 
import torch.nn as nn
import scipy.io as sio 
import copy

class LSTM(nn.Module):
    
    def __init__(self, input_dim, output_dim, batch_size, seq_len, n_hidden= 10 ,n_layers = 1): # no dropout for now 
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        #used in quantization:
        self.quantized_state_dict = {}
        self.centroids = {}
      
        
       

        """self.net = nn.Sequential(nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True), 
                         nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True), 
                        nn.Linear(n_hidden, (TrainY.shape[1])))"""
        #lstm layers
        self.lstm = nn.LSTM(self.input_dim, self.n_hidden, self.n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden, self.n_layers, batch_first=True)
        #self.lstm3 = nn.LSTM(self.n_hidden, self.n_hidden, self.n_layers, batch_first=False)
        #output layer
        self.fc1 = nn.Linear(self.n_hidden, self.output_dim)
        self.fc2 = nn.Linear(self.seq_len, self.output_dim)
        self.act = nn.Tanh() #nn.ReLU()
    
    def binarize_weights(self, ind_layer) : 
        weights = self.net[ind_layer].weight_ih_l[0] 
        for w in weights : 
            if w >= 0 : 
                w = 1
            else : 
                w = -1 
        self.net[ind_layer].weight_ih_l[k]  = weights 

    
    def forward(self, input, hidden):
        ''' Forward pass through the network. 
            These inputs are x, Ifand the hidden/cell state `hidden`. '''
        
        ## Get the outputs and the new hidden state from the lstm


        #our input has shape [batch_size, seq_len, input_dim] but lstm wants [seq_len, batch_size, input_dim]
        #reshaping does not ahcieve what we want here so we need to reconstrcut the input the way lstm wants:
        # new_input = torch.ones((self.seq_len, input.shape[0], self.input_dim))
        # for i in range(self.seq_len):
        #     new_input[i] = input[:,i,:]
       
        # input = new_input
      
     
        r_output, hidden = self.lstm(input, hidden)
        r_output, hidden = self.lstm2(r_output, hidden)

        out = self.fc1(r_output)
        out = torch.squeeze(out)
        out = self.fc2(out)
       
        return out, hidden


    #this name should be modified to forward
    def quantized_predict(self, input, hidden):

       
        orig_weights = {}
        old_state_dict = copy.deepcopy(self.state_dict())

        #replace the current weight with the quantized weights
        for item in self.state_dict().items():
                
            weight_name = item[0]
            weights_mat = item[1]
            orig_weights[weight_name] = weights_mat.clone()
           

            centroids = self.centroids[weight_name]
            weights_vect = torch.flatten(weights_mat)
           
            weights_vect = torch.LongTensor(weights_vect.long())
            vect = torch.index_select(centroids, 0, weights_vect)
            mat = vect.reshape(weights_mat.shape)
            
            self.state_dict()[weight_name].copy_(torch.nn.Parameter(mat))
           
         
        new_input = torch.ones((self.seq_len, input.shape[0], self.input_dim))
        for i in range(self.seq_len):
            new_input[i] = input[:,i,:]
        input = new_input

        r_output, hidden = self.lstm(new_input, hidden)
        r_output, hidden = self.lstm2(r_output, hidden)
        out = self.fc(r_output)
        
        #change the weights back once done with prediction
        self.load_state_dict(old_state_dict)
      
        
        return out, hidden


        

    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden = (hidden_state, cell_state)

        return hidden