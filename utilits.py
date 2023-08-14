import numpy as np
import torch
import random
import torch.nn as nn


def slice(pred, tetra, alpha_):

    tetra = tetra.long()
    # pred = pred - torch.mean(pred, axis=1, keepdim=True)
    pointlist = []
    mesh_points = []
    facelist = []

    for i in range(tetra.size()[0]):
        
        tetr = tetra[i]
        pre = pred[i]
        alpha = alpha_[i]
        # target_slice = pre[:, 3]
        # noise = torch.empty_like(target_slice).uniform_(*(-1e-4,1e-4))
        # target_slice = torch.where(target_slice == alpha, target_slice + noise, target_slice)
        # # pre[:, 3] = target_slice
        # pre = torch.cat((pred[i,:,0:3],target_slice[:,None]),dim=1)
        p1 = pre[tetr[:,0],:].view(-1,1,4)
        p2 = pre[tetr[:,1],:].view(-1,1,4)
        p3 = pre[tetr[:,2],:].view(-1,1,4)
        p4 = pre[tetr[:,3],:].view(-1,1,4)
        
        Tetrah = torch.cat((p1,p2,p3,p4),dim=1)

        edge = torch.zeros(tetra.size()[1],6).type_as(pred)
        point = torch.zeros(tetra.size()[1],6,2,4).type_as(pred)

        edge[:,0] = (Tetrah[:,0,3]- alpha) * (Tetrah[:,1,3]- alpha)
        point[:,0,0,:] = Tetrah[:,0,:]
        point[:,0,1,:] = Tetrah[:,1,:]
        edge[:,1] = (Tetrah[:,0,3]- alpha) * (Tetrah[:,2,3]- alpha)
        point[:,1,0,:] = Tetrah[:,0,:]
        point[:,1,1,:] = Tetrah[:,2,:]
        edge[:,2] = (Tetrah[:,0,3]- alpha) * (Tetrah[:,3,3]- alpha)
        point[:,2,0,:] = Tetrah[:,0,:]
        point[:,2,1,:] = Tetrah[:,3,:]
        edge[:,3] = (Tetrah[:,1,3]- alpha) * (Tetrah[:,2,3]- alpha)
        point[:,3,0,:] = Tetrah[:,1,:]
        point[:,3,1,:] = Tetrah[:,2,:]
        edge[:,4] = (Tetrah[:,1,3]- alpha) * (Tetrah[:,3,3]- alpha)
        point[:,4,0,:] = Tetrah[:,1,:]
        point[:,4,1,:] = Tetrah[:,3,:]
        edge[:,5] = (Tetrah[:,2,3]- alpha) * (Tetrah[:,3,3]- alpha)
        point[:,5,0,:] = Tetrah[:,2,:]
        point[:,5,1,:] = Tetrah[:,3,:]

        index = torch.nonzero(edge<0)

        m = point[index[:,0],index[:,1],0,0] - point[index[:,0],index[:,1],1,0]
        n = point[index[:,0],index[:,1],0,1] - point[index[:,0],index[:,1],1,1]
        p = point[index[:,0],index[:,1],0,2] - point[index[:,0],index[:,1],1,2]
        q = point[index[:,0],index[:,1],0,3] - point[index[:,0],index[:,1],1,3]

        x0 = point[index[:,0],index[:,1],0,0]
        y0 = point[index[:,0],index[:,1],0,1]
        z0 = point[index[:,0],index[:,1],0,2]
        w0 = point[index[:,0],index[:,1],0,3]

        x = ( alpha-w0)*m/q + x0
        y = ( alpha-w0)*n/q + y0
        z = ( alpha-w0)*p/q + z0

        x = x.view(-1,1)
        y = y.view(-1,1)
        z = z.view(-1,1)

        point = torch.cat((x,y,z),dim=1).view(1,-1,3)

        ll = (torch.arange(len(index))+1).float().type_as(pred)
        edge0 = torch.zeros(tetra.size()[1],6).type_as(pred)

        zero = torch.zeros_like(edge).type_as(pred)
        one = torch.ones_like(edge).type_as(pred)
        edge0[index[:,0],index[:,1]] = ll

        edgenum = torch.where(edge0 > 0, one, zero)
        edgenum = torch.sum(edgenum,dim = 1)
        index_edge3 = torch.nonzero(edgenum==3)
        index_edge4 = torch.nonzero(edgenum==4)
        face3 = torch.zeros(index_edge3.size()[0],3).type_as(pred)
        face4 = torch.zeros(index_edge4.size()[0]*4,3).type_as(pred)
        edge3 = edge0[index_edge3,:].view(-1,6)
        edge4 = edge0[index_edge4,:].view(-1,6)
        index_3 = torch.nonzero(edge3!=0)
        index_4 = torch.nonzero(edge4!=0) 

        face3 = edge3[index_3[:,0],index_3[:,1]].view(index_edge3.size()[0],3)
        edge4 = edge4[index_4[:,0],index_4[:,1]].view(index_edge4.size()[0],4)

        face4[0:index_edge4.size()[0],:] = edge4[:,0:3]
        face4[index_edge4.size()[0]:index_edge4.size()[0]*2,:] = edge4[:,1:4]
        face4[index_edge4.size()[0]*2:index_edge4.size()[0]*3,0:2] = edge4[:,0:2]
        face4[index_edge4.size()[0]*2:index_edge4.size()[0]*3,2] = edge4[:,3]
        face4[index_edge4.size()[0]*3:index_edge4.size()[0]*4,0] = edge4[:,0]
        face4[index_edge4.size()[0]*3:index_edge4.size()[0]*4,1:3] = edge4[:,2:4]      

        face = torch.cat((face3,face4,face3),dim=0).view(1,-1,3)-1
        facelist.append(face)
        mesh_points.append(point)


    return mesh_points, facelist

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

def write_obj4(pointls,meshs,files):

    fout = open(files, 'w')

    fout.write(str(len(pointls))+'\n')

    for i in range(len(pointls)):

        fout.write('v' + ' '+str(pointls[i,0]) + ' ' + str(pointls[i,1]) + ' ' + str(pointls[i,2]) + ' ' + str(pointls[i,3]) + "\n")

    fout.write(str(len(meshs))+ '\n')
    for i in range(len(meshs)):

        mesh = meshs[i]
        fout.write('t' + ' ' + str(int(mesh[0])) + ' ' + str(int(mesh[1])) + ' ' + str(int(mesh[2])) + ' ' + str(int(mesh[3]))+"\n")

    fout.close()
