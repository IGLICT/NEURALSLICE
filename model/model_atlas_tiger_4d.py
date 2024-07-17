import os
from re import I

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(PointNet, self).__init__()
        
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, zdim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(zdim)

        self.fc1 = nn.Linear(zdim, zdim)
        self.fc2 = nn.Linear(zdim, zdim)
        self.fc_bn1 = nn.BatchNorm1d(zdim)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)),inplace=True)
        x = F.relu(self.bn2(self.conv2(x)),inplace=True)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.zdim)
        
        ms = F.relu(self.fc_bn1(self.fc1(x)),inplace=True)
        ms = self.fc2(ms)

        return ms


class MappingTemplet(nn.Module):
    def __init__(self, zdim, point_dim=4):
        super(MappingTemplet, self).__init__()

        self.conv1 = nn.Conv1d(point_dim+zdim, (point_dim+zdim)//2, 1)
        self.conv2 = nn.Conv1d((point_dim+zdim)//2, (point_dim+zdim)//4, 1)
        self.conv3 = nn.Conv1d((point_dim+zdim)//4, 128, 1)
        self.conv4 = nn.Conv1d(128, point_dim, 1)

        self.bn1 = nn.BatchNorm1d((point_dim+zdim)//2)
        self.bn2 = nn.BatchNorm1d((point_dim+zdim)//4)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, input, latent):

        x = input.transpose(1,2)

        latent = latent.repeat(1,1,x.size()[2])
        x = torch.cat((latent,x),dim=1)
        x = F.relu(self.bn1(self.conv1(x)),inplace=True)
        x = F.relu(self.bn2(self.conv2(x)),inplace=True) 
        x = F.relu(self.bn3(self.conv3(x)),inplace=True) 
        x = torch.tanh(self.conv4(x))

        x = x.transpose(1,2)

        return x

# class Atlas_4d(nn.Module):

#     def __init__(self, encoder_type = 'image', zdim=1024):
#         super(Atlas_4d, self).__init__()

#         print("Atlas-4d with {} length embedding initialized".format(zdim))

#         self.v, self.t = read_4obj('./model/tour_small.4obj')

#         self.t = torch.from_numpy(self.t).long()
#         self.v = torch.from_numpy(self.v) / 6
#         self.num_local_mlp = 25   
#         self.num_encoder = 25  
#         self.attention = MultiheadAttention(zdim,8)     
#         if encoder_type == 'point':
#             print("Choosing Point Encoder")
#             self.encoder = nn.ModuleList([PointNet(zdim) for i in range(0,self.num_encoder)]).float()  
#         else:
#             print(" Invalid choice of encoder!! ")
        
#         self.decoder = nn.ModuleList([MappingTemplet(zdim=zdim) for i in range(0,25)]).float()


#     def forward(self, input):

#         batch_size = input.shape[0]
#         num_point = input.shape[1]
#         outs = []
#         latent = []

#         for i in range(self.num_encoder):
#             latent.append(self.encoder[i](input).view(1,batch_size,-1))
#         latent = torch.cat((latent),dim=0)
#         # latent,_ = self.attention(latent1,latent1,latent1)

#         vertices = self.v.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
#         face = self.t.unsqueeze(0).repeat(batch_size,1,1).type_as(input)

#         num_local = vertices.shape[1] // self.num_local_mlp
#         for i in range(0,self.num_local_mlp-1):
#             points = vertices[:,i*num_local:(i+1)*num_local,:]
#             outs.append(self.decoder[i](points,latent[i].unsqueeze(2)))
#         points = vertices[:,(self.num_local_mlp-1)*num_local:vertices.shape[1],:]
#         outs.append(self.decoder[i](points,latent[i].unsqueeze(2)))

#         outs = torch.cat((outs),dim=1)
  
#         return outs, face, latent

# class Atlas_4d(nn.Module):
#     def __init__(self, encoder_type = 'image', zdim=1000):
#         super(Atlas_4d, self).__init__()

#         print("Atlas-tiger with {} length embedding initialized".format(zdim))

#         self.v, self.t = read_4obj('./model/tiger.4obj')

#         self.t = torch.from_numpy(self.t).long()
#         self.v = torch.from_numpy(self.v) / 4
       
#         if encoder_type == 'point':
#             print("Choosing Point Encoder")
#             self.encoder = PointNet(zdim).float()  
#         else:
#             print(" Invalid choice of encoder!! ")
        
#         self.decoder = MappingTemplet(zdim).float()

#     def forward(self, input):

#         batch_size = input.shape[0]
#         num_point = input.shape[1]

#         latent = self.encoder(input)
#         code = latent.unsqueeze(1)    
#         latent = latent.unsqueeze(2)

#         vertices = self.v.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
#         face = self.t.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
#         vertices0 = vertices
#         pred = self.decoder(vertices, latent)

#         return pred, face, latent


class Atlas_4d(nn.Module):

    def __init__(self, encoder_type = 'image', zdim=1024):
        super(Atlas_4d, self).__init__()

        print("Atlas-4d with {} length embedding initialized".format(zdim))

        self.v, self.t = read_4obj('./model/tour_small.4obj')

        self.t = torch.from_numpy(self.t).long()
        self.v = torch.from_numpy(self.v) / 6
        self.num_local_mlp = 25   
        self.num_encoder = 25     
        if encoder_type == 'point':
            print("Choosing Point Encoder")
            self.encoder = PointNet(zdim).float()  
        else:
            print(" Invalid choice of encoder!! ")
        
        self.decoder = nn.ModuleList([MappingTemplet(zdim=zdim) for i in range(0,25)]).float()


    def forward(self, input):

        batch_size = input.shape[0]
        num_point = input.shape[1]
        outs = []

        latent = self.encoder(input)

        vertices = self.v.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
        face = self.t.unsqueeze(0).repeat(batch_size,1,1).type_as(input)

        num_local = vertices.shape[1] // self.num_local_mlp
        for i in range(0,self.num_local_mlp-1):
            points = vertices[:,i*num_local:(i+1)*num_local,:]
            outs.append(self.decoder[i](points,latent.unsqueeze(2)))
        points = vertices[:,(self.num_local_mlp-1)*num_local:vertices.shape[1],:]
        outs.append(self.decoder[i](points,latent.unsqueeze(2)))

        outs = torch.cat((outs),dim=1)
  
        return outs, face, latent


# class MapingSphere(nn.Module):
#     def __init__(self, zdim, point_dim=4):
#         super(MapingSphere, self).__init__()

#         self.conv1 = nn.Conv1d(point_dim+zdim, point_dim+zdim, 1)
#         self.conv2 = nn.Conv1d(point_dim+zdim,  1024, 1)
#         self.conv3 = nn.Conv1d(1024, 256, 1)
#         self.conv4 = nn.Conv1d(256, 128, 1)
#         self.conv5 = nn.Conv1d(128, 64, 1)
#         self.conv6 = nn.Conv1d(64, 32, 1)
#         self.conv7 = nn.Conv1d(32, point_dim, 1)

#         self.bn1 = nn.BatchNorm1d(point_dim+zdim)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.bn5 = nn.BatchNorm1d(64)
#         self.bn6 = nn.BatchNorm1d(32)

#     def forward(self, input, latent):

#         x = input.transpose(1,2)

#         latent = latent.repeat(1,1,x.size()[2])
#         x = torch.cat((latent,x),dim=1)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = torch.tanh(self.conv7(x))

#         x = x.transpose(1,2)

#         return x


def load_partial_pretrained(mymodel, path):

    pretrained_dict = torch.load(path)
    model_dict = mymodel.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict) 
    mymodel.load_state_dict(model_dict)
    print('Load pretrained model: ',path)

def read_4obj(filename):

    fin = open(str(filename))
    lines = fin.readlines()
    fin.close()

    vertices = []
    tetrahedrons = []

    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        if line[0] == 'v':
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            w = float(line[4])
            vertices.append([x,y,z,w])
        if line[0] == 't':
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            w = int(line[4].split("/")[0])
            tetrahedrons.append([x,y,z,w])

    vertices = np.array(vertices, np.float32)
    tetrahedrons = np.array(tetrahedrons, np.float32)

    return vertices, tetrahedrons
