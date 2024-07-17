import torch
import time
def embendding(x,L):
    pi = 3.14159265358979323846264338327950288
    dim = x.shape[2]
    out = []
    for i in range(dim):
        coor = x[:,:,i]
        gamma = []
        for j in range(L):
            gamma.append(torch.sin(2.0**j * coor * pi).unsqueeze(2))
            gamma.append(torch.cos(2.0**j * coor * pi).unsqueeze(2))
        gamma = torch.cat((gamma),dim=2)
        out.append(gamma)
    out = torch.cat((out),dim=2)
    return out


def inverse_embed(x,L):
    pi = 3.14159265358979323846264338327950288
    dim = 4
    out = []
    for i in range(dim):
        coor = x[:,:,i*10:(i+1)*10]
        gamma = torch.zeros(x.shape[0],x.shape[1]).cuda()
        for j in range(L):
            gamma += torch.sin(2.0**j * coor[:,:,j] * pi)
            gamma += torch.sin(2.0**j * coor[:,:,j] * pi)
        out.append(gamma.unsqueeze(2))
    out = torch.cat((out),dim=2)
    return out
