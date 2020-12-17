
import torch.nn as nn
import torch 
import numpy as np
import copy
import torch.nn.utils.prune as prune

from pruning import ThresholdPruning


class LSTM(nn.Module):
    
    def __init__(self, input_dim, output_dim, seq_len, n_hidden= 10 , n_layers = 1, fixed_pt_quantize = False):
        """
        Function that initializes our lstm netwrok with two lstm layers and one linear layer

        Parameters
        ----------
        input_dim: int
            number of features
        output_dim: int
            pediction's dimension
        seq_len: int
            length of sequence of data
        n_hidden: int
            number of hidden nodes of eacch lstm layer
        n_layers: int
            number of layers in each lstm layer
      
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.fixed_pt_quantize = fixed_pt_quantize


        #used in quantization:
        self.quantized_state_dict = {}
        self.centroids = {}

        
        #lstm layers
        self.lstm = nn.LSTM(self.input_dim, self.n_hidden, self.n_layers, batch_first=False)
        #self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden, self.n_layers, batch_first=False)
        #linear layer
        self.fc = nn.Linear(self.n_hidden, self.output_dim)
        self.act = nn.Sigmoid()
    
    
    def binarize_weights(self):

        """
        Function that binarizes the parameters of the given layer

        Parameters
        ----------
        ind_layer: int
            the index of the layer to be binarized
      
        """ 
        
        net = self.lstm
        for item in net.state_dict().items():
            weight_name = item[0]
            weights_mat = item[1]
            for idx, w_ in enumerate(weights_mat) : 
                with torch.no_grad():
                    arr = weights_mat[idx]
                    arr[arr>=0]=1 
                    arr[arr<0]=-1
                    weights_mat[idx] = arr
        
            net.state_dict()[weight_name].copy_(weights_mat)
            
            
    def quantize_fixed_pt(self):

        """
        Function that does 3 decimal fixed point quantization on the weights of the layer
        Parameters
        ----------
        None
      
        """ 
        net = self.lstm
        for item in net.state_dict().items():
            weight_name = item[0]
            weights_mat = item[1]
            for idx, w_ in enumerate(weights_mat) : 
                with torch.no_grad():
                    arr = weights_mat[idx]
                    arr = np.round(arr,3)
                    weights_mat[idx] = arr
        
            net.state_dict()[weight_name].copy_(weights_mat)

    def threshold_pruning(self):
        """
        Function that pruns the weights
        """

        for k in range(self.n_layers):
            parameters_to_prune = ((self.lstm, "weight_ih_l"+str(k)), (self.lstm, "weight_hh_l"+str(k)) )
            prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold= 0.125)
        


    def forward(self, input, hidden, quant=False):
        """
        Function that forward pass through the network 

        Parameters
        ----------
        TrainX: matrix of shape [#datapoints, seq_len, input_dim]
            contain feature vectors of train data
        hidden: pair of matrices of shape [n_layer, batch_size, n_hidden]
            hidden/cell state of LSTM layers
        quant: boolean
            if set, the network is quantized
        Returns
        -------
        out: matrix of shape [seq_len, batch_size, n_hidden]
            scaler used to standardize the data
        hidden: pair of matrices of shape [n_layer, batch_size, n_hidden]
            hidden/cell state of LSTM layers
        """ 
        if (not self.fixed_pt_quantize) and (not quant):
            ##our input has shape [batch_size, seq_len, input_dim] but lstm wants [seq_len, batch_size, input_dim], we need to reconstrcut the input the way lstm wants:
            new_input = torch.ones((self.seq_len, input.shape[0], self.input_dim))
            for i in range(self.seq_len):
                new_input[i] = input[:,i,:]

           
            r_output, hidden = self.lstm(new_input, hidden)
        
            ## put x through the fully-connected layer
            out = self.act(r_output)
            out = self.fc(out)           
            
            return out, hidden

        if self.fixed_pt_quantize:
            ##our input has shape [batch_size, seq_len, input_dim] but lstm wants [seq_len, batch_size, input_dim], we need to reconstrcut the input the way lstm wants:
            new_input = torch.ones((self.seq_len, input.shape[0], self.input_dim))
            for i in range(self.seq_len):
                new_input[i] = input[:,i,:]
        
        
            self.quantize_fixed_pt()
            r_output, hidden = self.lstm(new_input, hidden)
        
            ## put x through the fully-connected layer
            out = self.act(r_output)
            out = self.fc(out)
           
            
            
            return out, hidden

        if quant:
            orig_weights = {}
            old_state_dict = copy.deepcopy(self.state_dict())
            tmp_state_dict = copy.deepcopy(self.state_dict())
            
            #computes the final parameters from quantized parameters and centroids 
            for item in self.state_dict().items():
                    
                weight_name = item[0]
                weights_mat = item[1]
                orig_weights[weight_name] = weights_mat.clone()
            

                centroids = self.centroids[weight_name]
                weights_vect = torch.flatten(weights_mat)
            
                indices = torch.LongTensor(weights_vect.long())
                for i in range(weights_vect.shape[0]):
                    weights_vect[i] = centroids[indices[i]]
                    

            
                mat = weights_vect.reshape(weights_mat.shape)

                
                tmp_state_dict[weight_name].copy_(mat)
                
            #update the model's state_dict with final parameters 
            self.load_state_dict(tmp_state_dict) 

        
            ##our input has shape [batch_size, seq_len, input_dim] but lstm wants [seq_len, batch_size, input_dim], we need to reconstrcut the input the way lstm wants:
            new_input = torch.ones((self.seq_len, input.shape[0], self.input_dim))
            for i in range(self.seq_len):
                new_input[i] = input[:,i,:]
            

            r_output, hidden = self.lstm(new_input, hidden)
            #r_output, hidden = self.lstm2(r_output, hidden)
            out = self.act(r_output)
            out = self.fc(out)
            
            
            #change the weights back to quantized parameters once done with prediction
            self.load_state_dict(old_state_dict)

        
            return out, hidden
    
    def quantized_predict(self, input, hidden):
        """
        Function that forward pass through the quantized network

        Parameters
        ----------
        TrainX: matrix of shape [#datapoints, seq_len, input_dim]
            contain feature vectors of train data
        hidden: pair of matrices of shape [n_layer, batch_size, n_hidden]
            hidden/cell state of LSTM layers

        Returns
        -------
        out: matrix of shape [seq_len, batch_size, n_hidden]
            scaler used to standardize the data
        hidden: pair of matrices of shape [n_layer, batch_size, n_hidden]
            hidden/cell state of LSTM layers
        """ 

       
        orig_weights = {}
        old_state_dict = copy.deepcopy(self.state_dict())
        tmp_state_dict = copy.deepcopy(self.state_dict())
        
        #computes the final parameters from quantized parameters and centroids 
        for item in self.state_dict().items():
                
            weight_name = item[0]
            weights_mat = item[1]
            orig_weights[weight_name] = weights_mat.clone()
           

            centroids = self.centroids[weight_name]
            weights_vect = torch.flatten(weights_mat)
           
            indices = torch.LongTensor(weights_vect.long())
            for i in range(weights_vect.shape[0]):
                weights_vect[i] = centroids[indices[i]]
                

         
            mat = weights_vect.reshape(weights_mat.shape)

            
            tmp_state_dict[weight_name].copy_(mat)
            
        #update the model's state_dict with final parameters 
        self.load_state_dict(tmp_state_dict) 

       
        ##our input has shape [batch_size, seq_len, input_dim] but lstm wants [seq_len, batch_size, input_dim], we need to reconstrcut the input the way lstm wants:
        new_input = torch.ones((self.seq_len, input.shape[0], self.input_dim))
        for i in range(self.seq_len):
            new_input[i] = input[:,i,:]
        input = new_input

        r_output, hidden = self.lstm(new_input, hidden)
        r_output, hidden = self.lstm2(r_output, hidden)
        out = self.fc(r_output)
        
        #change the weights back to quantized parameters once done with prediction
        self.load_state_dict(old_state_dict)

    
        return out, hidden


    def init_hidden(self, batch_size):
        """
        Function that initializes hidden state

        Parameters
        ----------
        batch_size: int
            batch size!

        Returns
        -------
        hidden: pair of matrices of shape [n_layer, batch_size, n_hidden]
            hidden/cell state of LSTM layers
        """ 
        # Create two new tensors with sizes n_layers x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden = (hidden_state, cell_state)

        return hidden


