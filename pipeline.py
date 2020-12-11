import numpy as np
import torch 
import torch.nn as nn
import scipy.io as sio 


Idx_subject = 10
Finger = [0,1,2,3,4]


#prepare train and test data
for Finger in list([0,1,2,3,4]): # 5 fingers for each subject. 0:thumb, 1:index, 2:middle ...
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


input_dim = TrainX.shape
hidden_dim = 10
n_layers = 1

lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)