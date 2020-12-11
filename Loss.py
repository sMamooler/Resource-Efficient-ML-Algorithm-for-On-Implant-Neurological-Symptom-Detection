import torch 



def corr_coeff(x,y):
    # mean_x = torch.mean(x,1)
    # xm = x.sub(mean_x.expand_as(x))
    # c = xm.mm(xm.t())
    # c = c/(x.size(1)-1)

    # d = torch.diag(c)
    # stddev = torch.pow(d,0.5)
    # c = c.div(stddev.expand_as(c))
    # c = c.div(stddev.expand_as(c).t())


    # c = torch.clamp(c,-1.0,1.0)
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    # if (y==0).all():
    #     return torch.tensor(0.0,requires_grad=True)
    # else:
    corr= torch.sum(vx * vy) / (torch.sqrt(torch.mean(vx ** 2)) * torch.mean(torch.sum(vy ** 2)))
        
    corr = torch.clamp(corr,-1.0,1.0)
    return corr

# train using correlation coefficient for loss 
def corr_coeff_loss(x,y):
    cost = corr_coeff(x,y)
        #cost = np.corrcoef(pred[-1,:,:].detach().numpy().reshape((-1,)),y.detach().numpy().reshape((-1,)))
    return 1-cost