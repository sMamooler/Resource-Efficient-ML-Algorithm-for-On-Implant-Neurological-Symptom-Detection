import torch 

def corr_coeff(x,y):
    """
    Function that computes coelation coefficient between two tensors with same size

    Returns
    -------
    corr: 
        correlation coefficient between x and y
    """ 
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr= torch.sum(vx * vy) / (torch.sqrt(torch.mean(vx ** 2)) * torch.mean(torch.mean(vy ** 2)))
        
    corr = torch.clamp(corr,-1.0,1.0)
    return corr

 
def corr_coeff_loss(x,y):
    """
    Function that computes 1-coelation coefficient between two tensors with same size

    Returns
    -------
    corr: 
        1-correlation coefficient between x and y
    """ 
    cost = corr_coeff(x,y)
    return 1-cost