import torch
from torch.utils.data import Dataset, DataLoader
from Loss import corr_coeff_loss, corr_coeff
import numpy as np


def test(net, dataset, loss_fn, batch_size=32):
    dataloader = DataLoader(dataset, batch_size, drop_last=True)
    net.eval()
    losses = []
    for batch_idx, batch in enumerate(dataloader):
        
        with torch.no_grad():
            input, target = batch['input'], batch['target']

            x = input.float()
            y = target.float()

            h = net.init_hidden(input.shape[0])
            pred, _ = net(x, h)

            loss = loss_fn(pred, y)
            losses.append(loss.item())


            if batch_idx==0:
                preds = pred
                ys = y
            else:
                preds = torch.cat((preds, pred), dim=1)
                ys = torch.cat((ys,y), dim=0)
           

    preds = preds.detach().numpy().reshape((-1,))
    ys = ys.detach().numpy().reshape((-1,))
    np_corr_coeff = np.corrcoef(preds, ys)

    #corr = corr_coeff(pred[-1,:,:],target)
    print("Prediction Loss: {l}".format(l=np.mean(losses)))
    print("Prediction Correlation Coefficient: {l}".format(l=np_corr_coeff[0][1]))
    

    

def quantited_test(net, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size, drop_last=True)

    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            input, target = batch['input'], batch['target']
            h = net.init_hidden(batch_size)
            pred, _ = net.quantized_predict(input.float(), h)
            target = batch['target']
            loss = corr_coeff_loss(pred, target)
            print("Prediction loss for the quantized model: {l}".format(l=loss))