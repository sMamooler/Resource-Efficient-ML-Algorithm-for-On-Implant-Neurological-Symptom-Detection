import numpy as np
import scipy.io as sio 
import torch.nn as nn
import torch 
import matplotlib.pyplot as plt
import os
import argparse

from preprocessing_funcs import get_spikes_with_history, preprocessing, remove_outliers
from model import LSTM
from trainer import train
from quantizer import quantize_network, compute_quantized_weights, quantized_train


#set this to the root directory where you want to save and load data and figures
root = os.path.join('drive')
#set this to the diectory where you want to save data and checkpoints
data_path = os.path.join(root, 'data')
#set this to the diectory where you want to save checkpoints
checkpoint_path = os.path.join(root, 'checkpoints')
#set this to directory where you want to save figures
figure_path = os.path.join(root, 'figures')


parser = argparse.ArgumentParser()
parser.add_argument("--pre_trained", help="Set to True if you want to use pre-trained model", type=bool)
parser.add_argument("--fixed_pt_quantization", help="Set to True if you want to use fixed point quantization", type=bool)
parser.add_argument("--prunning", help="Set to True if you want to use prunning", type=bool)
parser.add_argument("--trained_quantization", help="Set to True if you want to use trained quantization", type=bool)

args = parser.parse_args()
pre_trained = args.pre_trained
fixed_pt_quantize = args.fixed_pt_quantize
prunning = args.prunning
trained_quantization = args.trained_quantization

for Idx_subject in list([10]): # 3 subjects index 10-12

       
        for Finger in list([3]): # 5 fingers for each subject. 0:thumb, 1:index, 2:middle ...
            
            #load training data (TrainX: feature vectors, TrainY: labels)
            matData = sio.loadmat(data_path + '/BCImoreData_Subj_'+str(Idx_subject)+'_200msLMP.mat')
            TrainX = matData['Data_Feature'].transpose()
            TrainY = matData['SmoothedFinger']
            TrainY = TrainY [:,Finger]
            TrainY = TrainY.reshape(TrainY.shape[0],1)
            #load testing data (TestX: feature vectors, TestY: labels)
            matData = sio.loadmat(data_path + '/BCImoreData_Subj_'+str(Idx_subject)+'_200msLMPTest.mat')
            TestX = matData['Data_Feature'].transpose()
            TestY = matData['SmoothedFinger']
            TestY = TestY[:,Finger]
            TestY = TestY.reshape(TestY.shape[0],1)
            
            # preprocessing 
            TrainX = remove_outliers(TrainX)
            
            x_scaler, y_scaler, TrainX, TestX, TrainY, TestY  = preprocessing(TrainX,TestX,TrainY,TestY)
            
            
            # from here, we reconstruct the input by "looking back" a few steps
            bins_before= 20 #How many bins of neural data prior to the output are used for decoding
            bins_current=1 #Whether to use concurrent time bin of neural data
            bins_after=0 #How many bins of neural data after the output are used for decoding
            
            TrainX=get_spikes_with_history(TrainX,bins_before,bins_after,bins_current)
            TrainX, TrainY = TrainX[bins_before:,:,:], TrainY[bins_before:,]
         
            TestX=get_spikes_with_history(TestX,bins_before,bins_after,bins_current)
            TestX, TestY = TestX[bins_before:,:,:], TestY[bins_before:,]
            
            # Now, we reconstructed TrainX/TestX to have a shape (num_of_samples, sequence_length, input_size)
            # We can fit this to the LSTM
            
            print("Run for finger "+str(Finger))

            n_hidden = 20
            n_layers = 5
            n_epochs =  50 
            input_dim = TrainX.shape[2]
            output_dim = TrainY.shape[1]
            seq_len =  TrainX.shape[1]

            net = LSTM(input_dim, output_dim, seq_len,  n_hidden, n_layers, fixed_pt_quantize = fixed_pt_quantize)

            lossfunc = nn.MSELoss()

            optimizer = torch.optim.Adamax(net.parameters(), lr=0.003)
            if prunning:

                print("Pruning============================================================================")
            
                net.train()
                if pre_trained:
                    net.load_state_dict(torch.load('f'+str(Finger)+'_trained_pruned_model'))
                else:
                    net.train()
                    net.threshold_pruning()
                    #train the prunned model:
                    try:
                        corr_train, corr_val, corr_test = train(TrainX, TrainY, TestX, TestY, net, lossfunc, optimizer, num_epoch=n_epochs, clip = 5, Finger = Finger)
                    except KeyboardInterrupt:
                        #save the model
                        print("saving...")
                    PATH_pre_trained = 'f'+str(Finger)+'_trained_pruned_model'
                    torch.save(net.state_dict(), PATH_pre_trained)
                    print("trained pruned model saved")

                net.eval()
                pred,h = net(torch.from_numpy(TestX).float(), net.init_hidden(TestX.shape[0]))
               
            elif trained_quantization:
                print("Trained Quantization===================================================================")
                net.train()
                if pre_trained:
                    net.load_state_dict(torch.load('f'+str(Finger)+'_trained_quantized_model'))
                else:
                    k=8
                    #initialize the quantiezed weights using the weights from the trained netwrok:
                    net = compute_quantized_weights(net,k)
                    net.train()
                    
                    #train the quantized netwok
                    corr_train, corr_val, corr_test = quantized_train(TrainX, TrainY,TestX,TestY, net, lossfunc, optimizer, num_epoch = n_epochs, clip = 5)
                    #set the model's parameters to their quantized version
                    net = quantize_network(net)
                    torch.save(net.state_dict(), 'f'+str(Finger)+'_trained_quantized_model')
                    print("trained quantized model saved!")

                net.eval()
                pred,h = net(torch.from_numpy(TestX).float(), net.init_hidden(TestX.shape[0]), quant=True)
               


            else:
                print("Baseline ===================================================================")
                net.train()
                ##training the initial model
                if pre_trained:
                    net.load_state_dict(torch.load('f'+str(Finger)+'_trained_model'))

                else:

                    try:
                        corr_train, corr_val, corr_test = train(TrainX, TrainY,TestX,TestY, net, lossfunc, optimizer, num_epoch = n_epochs, clip = 5, Finger=Finger)
                    except KeyboardInterrupt:
                        #save the model
                        print("saving...")
                    PATH_pre_trained = 'f'+str(Finger)+'_trained_model'
                    torch.save(net.state_dict(), PATH_pre_trained)
                    print("model saved")

                ##test initial model
                net.eval()
                pred,h = net(torch.from_numpy(TestX).float(), net.init_hidden(TestX.shape[0]))


            pred = pred.detach().numpy()[-1,:,:]
            pred = y_scaler.inverse_transform(pred)
            TestY = y_scaler.inverse_transform(TestY)
            pred = pred.reshape((-1,))
            corrcoef = np.corrcoef(pred,TestY.reshape((-1,)))
            TestYShifted = TestY
            x = np.arange(TestYShifted.shape[0])
            
           
            fig_label = plt.figure(figsize=(15,10))
            plt.title("Subject_" + str(Idx_subject) + "_Finger_"+str(Finger))
            plt.plot(x, TestYShifted)
            plt.plot(x, pred)
            fig_label.savefig(figure_path + "/Subject_" + str(Idx_subject) + "_Finger_"+str(Finger))

            
            print ('Correlation coefficient test : {corrcoef}'.format(corrcoef=corrcoef[0,1]))  