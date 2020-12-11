import numpy as np
import scipy.io as sio 

import torch.nn as nn
import torch 
from torch.utils.data import Dataset, DataLoader

from preprocessing_funcs import get_spikes_with_history, standardize, remove_outliers
from model import LSTM
from trainer import train
from evaluator import test, quantited_test
from FingerDataset import FingerDataset
from Loss import corr_coeff, corr_coeff_loss
from quantizer import compute_quantized_weights, quantized_train, quantize_network


if __name__ == '__main__':
    for Idx_subject in list([10]):#,11,12]): # 3 subjects index 10-12
        for Finger in list([0]): #,1,2,3,4]): # 5 fingers for each subject. 0:thumb, 1:index, 2:middle ...

            #load training data (TrainX: feature vectors, TrainY: labels)
            matData = sio.loadmat('data/BCImoreData_Subj_'+str(Idx_subject)+'_200msLMP.mat')
            TrainX = matData['Data_Feature'].transpose()
            TrainY = matData['SmoothedFinger']
            TrainY = TrainY [:,Finger]
            TrainY = TrainY.reshape(TrainY.shape[0],1)
            #load testing data (TestX: feature vectors, TestY: labels)
            matData = sio.loadmat('data/BCImoreData_Subj_'+str(Idx_subject)+'_200msLMPTest.mat')
            TestX = matData['Data_Feature'].transpose()
            TestY = matData['SmoothedFinger']
            TestY = TestY[:,Finger]
            TestY = TestY.reshape(TestY.shape[0],1)
            
            #standardize and remove outliers from the data
            TrainX = standardize(TrainX)
            TrainX = remove_outliers(TrainX)
            TestX = standardize(TestX)

            # from here, we reconstruct the input by "looking back" a few steps
            bins_before= 20 #How many bins of neural data prior to the output are used for decoding
            bins_current=1 #Whether to use concurrent time bin of neural data
            bins_after=0 #How many bins of neural data after the output are used for decoding

           
            
            TrainX=get_spikes_with_history(TrainX,bins_before,bins_after,bins_current)

            TrainX, TrainY = TrainX[bins_before:,:,:], TrainY[bins_before:,]
         
            TestX=get_spikes_with_history(TestX,bins_before,bins_after,bins_current)
            TestX, TestY = TestX[bins_before:,:,:], TestY[bins_before:,]
            
            # Now, we reconstructed TrainX/TestX to have a shape (num_of_samples, sequence_length, input_size)
            
            print(TrainX.shape)
            # You can fit this to the LSTM

            print("run for finger ", Finger)
            
            input_dim = TrainX.shape[2]
            output_dim = TrainY.shape[1]

            #for smaller batch size we get unreasonable results
            batch_size = 256 #TrainX.shape[0]
            seq_len = TrainX.shape[1]
            n_hidden = 10
            n_layers = 10

            net = LSTM(input_dim, output_dim, batch_size, seq_len, n_hidden, n_layers)
            
            #train the model
            train_dataset = FingerDataset(TrainX, TrainY)
            try:
                train(net, train_dataset, num_epoch=500, batch_size=batch_size)
            except KeyboardInterrupt:
                #save the model
                PATH_pre_trained = 'pre_trained_model'
                torch.save(net.state_dict(), PATH_pre_trained)
                print("model saved")


            # Preprocess the data may leed to better performance. e.g. StandardScaler 
            
            net.load_state_dict(torch.load('pre_trained_model'))
            test_dataset = FingerDataset(TestX, TestY)
            test(net, test_dataset, nn.MSELoss(), batch_size=256)
            
            


            ###################QUANTIZATION###############################################
            # #quantize the weight matrices
            # #net.load_state_dict(torch.load(PATH_pre_trained))
            # k=8

            # #initialize the quantiezed weights using the weights from the trained netwrok:
            # compute_quantized_weights(net,k)
            # #re-train after quantization
            # quantized_train(net, train_dataset, num_epoch=5, batch_size=batch_size, lr=0.02)
            # #replace the netwrok parameters with the quantized ones
            # net=quantize_network(net)

            # #save the model_
            # PATH_trained = 'trained_model'
            # #torch.save(net.state_dict(), PATH_trained)
            # #net.load_state_dict(torch.load(PATH_trained))

            # ###from now on the quantized_predict should be used for prediction

            # #test the predictions resulting from the quantized parameters
            # quantited_test(net, test_dataset, batch_size=TestX.shape[0])