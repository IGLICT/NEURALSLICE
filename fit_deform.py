import os
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import join_meshes_as_batch, Meshes
import trimesh
from utilits import read_4obj, load_partial_pretrained, slice, write_obj4



class PointNetMix(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(PointNetMix, self).__init__()
        
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, zdim, 1)

        self.fc1 = nn.Linear(zdim*2, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
    
        x = x.transpose(1, 2)
        x = F.leaky_relu(self.conv1(x),0.2,inplace=True)
        x = F.leaky_relu(self.conv2(x),0.2,inplace=True)
        x = self.conv3(x)
        x_max = torch.max(x, 2, keepdim=True)[0]
        x_max = x_max.view(-1, self.zdim)
        x_avg = torch.mean(x, 2, keepdim=True)
        x_avg = x_avg.view(-1, self.zdim)
        mix = torch.cat((x_max,x_avg),dim=1)
        
        ms = F.leaky_relu(self.fc1(mix),0.2,inplace=True)
        
        ms = self.fc2(ms)
        ms = torch.tanh(ms)
        
        return ms

class lip_mlp(nn.Module):
    def __init__(self, hidden_dim=3) -> None:
        super().__init__()
        self.sp = nn.Softplus()
        self.act = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.transform = nn.Linear(4,4,bias=False)
        self.linears = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(hidden_dim)]).append(nn.Linear(1024,4))
        self.linears.insert(0, nn.Linear(4,1024))
        self.v, self.t = read_4obj('./model/sphere_huge.4obj')
        self.t = torch.from_numpy(self.t).long()
        self.v /= np.max(np.linalg.norm(self.v, axis=1))
        self.v = torch.from_numpy(self.v)  
        self.loc = PointNetMix(256).float()
        print("Template rad:", np.max(np.linalg.norm(self.v, axis=1)))
        
    def scale(self, out, w):
        min_z = torch.min(out[:,:,3],dim=1)[0][:,None]
        max_z = torch.max(out[:,:,3],dim=1)[0][:,None]
        medium = (min_z+max_z)/2
        return w*(max_z-medium) + medium
        
    def normalization(self, W, c):
        absrowsum = torch.sum(torch.abs(W),dim=1)
        scale = torch.minimum(torch.ones((1)).cuda(), c/absrowsum)
        return W * scale[:, None]
    
    def lip_norm(self):
        for i in range(len(self.linears)):
           self.linears[i].weight.data = self.normalization(self.linears[i].weight, 3)

    def deform(self, vertices):
        x = self.act(self.linears[0](vertices))
        for i in range(1,3):
            x = self.act(self.linears[i](x))
        out = torch.tanh(self.linears[-1](x))
        return out
    
    def forward(self, p):
        
        vertices = self.v.unsqueeze(0).repeat(p.shape[0],1,1).type_as(p)
        face = self.t.unsqueeze(0).repeat(p.shape[0],1,1).type_as(p)

        vertices.requires_grad = True

        self.lip_norm()

        out = self.deform(vertices)
        
        w = self.loc(p)
        w = self.scale(out, w)
        return out, face, w


def _loss(out, face, inputs, w):
        pointlist, facelist = slice(out, face, w)
        meshlist = []
        for j in range(len(pointlist)):
            meshlist.append(Meshes(verts=pointlist[j],faces=facelist[j]))

        meshlist = join_meshes_as_batch(meshlist)
        try:
            pts = sample_points_from_meshes(meshes=meshlist,num_samples=2500)
        except:
           print('error slice')
        L1, _ = chamfer_distance(x=inputs,y=pts)

        return L1, pointlist, facelist
                       
def train():

    for i in range(0,iterations):
        optimizer.zero_grad()
        inputs = []
        points = sample_points_from_meshes(meshes,num_samples=2500).cuda()
        points /= 1.5 
        for j in range(len(points)):
            mask = np.random.randint(0, points[j].shape[0], 2000)
            inputs.append((points[j][mask,...]).float().to(device)[None,:,:])
        inputs = torch.cat((inputs),dim=0)
        inputs.requires_grad = True

        out, face, w  = model(inputs)
        out = out[0:1,:,:].repeat(points.shape[0],1,1)
        face = face[0:1,:,:].repeat(points.shape[0],1,1)

        w = torch.sort(w,descending=True)[0]
        
        L1, pointlist, facelist = _loss(out, face, points, w)

        loss = L1
        loss.backward()

        optimizer.step()
        scheduler.step() 
        if i % 100 == 0:
            print(i, "CD Loss:",L1.item(),
                " Learning Rate:",optimizer.state_dict()['param_groups'][0]['lr'])

        if i % 100 == 0:
            for j in range(len(points)):
                v = pointlist[j].view(-1,3).cpu().detach().numpy()
                f = facelist[j].view(-1,3).cpu().detach().numpy()
                mesh = trimesh.Trimesh(vertices=v, faces=f)
                os.makedirs('{}'.format("fit_deform"), exist_ok=True)
                mesh.export('{}/{}.obj'.format("fit_deform",str(i)+"_"+str(j)));
        
        
    torch.save(model.state_dict(),'./fit_deform/step_{}'.format(i))
    write_obj4(out[0].detach().cpu().numpy(),face[0].detach().cpu().numpy(),'./fit_deform/top.4obj')
    return w


torch.cuda.set_device(0)
device = "cuda"

lr = 1e-4
model = lip_mlp().float().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))

objlist = []
for i in ['0000','0010','0020','0030','0039']: 
    # model_path = './dataset3/' + str(i) + '.obj'
    model_path = './mesh_seq2_new/' + i + '.obj'
    mesh = trimesh.load_mesh(model_path)
    v, f = mesh.vertices, mesh.faces
    mesh_th = Meshes(verts=torch.from_numpy(v)[None,:,:].float(),faces=torch.from_numpy(f)[None,:,:])
    objlist.append(mesh_th)
meshes = join_meshes_as_batch(objlist)

points = sample_points_from_meshes(meshes,num_samples=200000)
points = points/1.5
iterations = 6000
w = train()

v, t = read_4obj('./fit_deform/top.4obj')
t = torch.from_numpy(t).long()[None,:,:]
v = torch.from_numpy(v)[None,:,:]
for i in range(100):
    w_ = torch.ones((1,1))*(w.min().item() + ((w.max().item()-w.min().item())/100)*i)
    pointlist, facelist = slice(v, t, w_)
    p = pointlist[0].view(-1,3).numpy()
    f = facelist[0].view(-1,3).numpy()
    mesh = trimesh.Trimesh(vertices=p, faces=f)
    mesh.export('{}/{}.obj'.format("fit_deform/obj",str(i)));
    