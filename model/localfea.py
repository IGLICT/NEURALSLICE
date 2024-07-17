import torch
import time
import trimesh
from src.encoder.pointnet import LocalPoolPointnet
import torch.nn as nn
import torch.nn.functional as F
from src.decoder import LocalDecoder, HyperDeform, refine, mlp_refine
from loss import normal_cons, sli
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, SubdivideMeshes
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_edge_loss
from pytorch3d.structures import join_meshes_as_batch
import numpy as np
from src.encoder.unet3d import UNet3D
from pytorch3d.ops import GraphConv
from model.model_atlas_tiger_4d import load_partial_pretrained, read_4obj

class Local_Fea_Deform(nn.Module):

    def __init__(self, cdim = 32, hiddendim = 32):
        super(Local_Fea_Deform, self).__init__()

        print("Local_Fea_Deform embedding initialized")

        self.encoder = LocalPoolPointnet(c_dim=cdim, dim=3, 
                        hidden_dim=hiddendim, scatter_type='max', 
                        unet3d=True, unet3d_kwargs={'num_levels': 3,
                                    'f_maps': 32,'in_channels': 32,
                                    'out_channels': 32}, 
                        grid_resolution=32, plane_type='grid', 
                        padding=0.1, n_blocks=5).float()

        self.decoder = LocalDecoder(dim=4, c_dim=cdim,
                        hidden_size=hiddendim, n_blocks=5, 
                        leaky=False, sample_mode='bilinear', 
                        padding=0.1).float()

    def forward(self, p, not_jacobian = False):

        grid = self.encoder(p)
        out, face, _,_, = self.decoder(p,grid,not_jacobian)

        return out, face,None, torch.zeros((32,1)).cuda(), None
    
class Local_Fea_Injective_Mapping(nn.Module):
    def __init__(self,cdim = 32, hiddendim = 32, mapping_layers = 10):
        super(Local_Fea_Injective_Mapping, self).__init__()
        
        print("Injective Mapping initialized")
        
        self.mapping_layer = mapping_layers
        
        self.decoder = HyperDeform(zdim=1024,
                                   mapping_layer = mapping_layers).float()
    
        self.loc = PointNetMix(zdim=1024).float()
        
    def forward(self, p):
        

        out, face, detlist = self.decoder(p)
        w = self.loc(p)
        
        out = out.clone() - torch.mean(out.clone(), axis=1, keepdim=True)
        scale_max = torch.max(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        scale_min = -torch.min(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        scale = torch.where(scale_max>scale_min, scale_min, scale_max)
        out[:,:,3] = (out[:,:,3].clone() / (scale+1e-10)) *1.2
        
        return out, face, torch.cat(detlist,dim=2), w
    
class Mesh_refine(nn.Module):
    def __init__(self):
        super(Mesh_refine, self).__init__()
        
        self.corase_mesh = Local_Fea_Deform4().float()
        load_partial_pretrained(self.corase_mesh,"./train_models/4DreconEpoch_590")
        self.refine = refine().float()
        
    def forward(self, p):
        with torch.no_grad():
            pred, tetra, _, w,grid = self.corase_mesh(p, not_jacobian=True) 
            _, _, _, _, pointlist, facelist = sli(pred, tetra, w)
        
        pointlist = self.refine(pointlist, grid)
        return pointlist, facelist

class LocalSubdivRefine(nn.Module):
    def __init__(self) -> None:
        super(LocalSubdivRefine, self).__init__()
        self.corase_mesh = Local_Fea_Deform4(stage=1).float()
        
        self.refine = refine().float()
        self.subdivmesh = SubdivideMeshes().float()
        load_partial_pretrained(self.corase_mesh,'./train_models/lip_norm_and_lap_smooth')
        
    def forward(self, points, L_points=None):
        # with torch.no_grad():
        noise =  0.005 * torch.randn(points.shape[0],points.shape[1],3).cuda()
        pred, tetra, w, grid = self.corase_mesh(points, not_jacobian = True)
        _, _, _, _, pointlist, facelist = sli(pred, tetra, torch.zeros_like(w).cuda())
        L_cd1, L_lap1, mesh, samples,_ = self.list_to_mesh(pointlist, facelist, points)
        
        meshes2 = self.subdivmesh(mesh)
        meshes2 = self.subdivmesh(meshes2)
        meshes2 = self.subdivmesh(meshes2)
        pointlist2, facelist2 = (meshes2.verts_list(), meshes2.faces_list())
        for i in range(len(pointlist)):
            pointlist2[i] = pointlist2[i].unsqueeze(0)
            facelist2[i] = facelist2[i].unsqueeze(0)
        pointlist2 = self.refine(pointlist2, points+noise, samples)
        
        L_cd2, L_lap2, _, _, L_n = self.list_to_mesh(pointlist2, facelist2, points, L_points)
        
        return L_cd1, L_lap1, L_cd2, L_lap2, pointlist2, facelist2, L_n
    
        
    def test(self, points):
        pred, tetra, jacobian, w, grid = self.corase_mesh(points, not_jacobian = True)
        _, _, _, _, pointlist, facelist = sli(pred, tetra, torch.zeros_like(w).cuda())

        meshlist = []
        for i in range(len(pointlist)):

            meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))

        mesh = join_meshes_as_batch(meshlist)
        samples = sample_points_from_meshes(meshes=mesh,num_samples=2562)
        meshes2 = self.subdivmesh(mesh)
        meshes2 = self.subdivmesh(meshes2)
        meshes2 = self.subdivmesh(meshes2)
        pointlist2, facelist2 = (meshes2.verts_list(), meshes2.faces_list())
        for i in range(len(pointlist)):
            pointlist2[i] = pointlist2[i].unsqueeze(0)
            facelist2[i] = facelist2[i].unsqueeze(0)
        t0 = time.time()
        pointlist2 = self.refine(pointlist2, points, samples)
        t1 = time.time()
        return pointlist2, facelist2
    
    def test_merge(self, points):
        # t0 = time.time()
        pred, tetra, jacobian, w, grid = self.corase_mesh(points, not_jacobian = True)
        _, _, _, _, pointlist, facelist = sli(pred, tetra, torch.zeros_like(w).cuda())
        # t1 = time.time()
        meshlist = []
        for i in range(len(pointlist)):
            mesh_pytorch3d = Meshes(verts=pointlist[i],faces=facelist[i])
            mesh_trimesh = trimesh.Trimesh(vertices=mesh_pytorch3d.verts_list()[0].detach().cpu().numpy(),
                              faces=mesh_pytorch3d.faces_list()[0].detach().cpu().numpy(),process=False)
            verts = torch.tensor(mesh_trimesh.vertices).float().cuda()
            faces = torch.tensor(mesh_trimesh.faces).float().cuda()
            mesh_pytorch3d_restored = Meshes(verts=verts[None,:,:], faces=faces[None,:,:])
            meshlist.append(mesh_pytorch3d_restored)
        # t2 = time.time()
        mesh = join_meshes_as_batch(meshlist)
        samples = sample_points_from_meshes(meshes=mesh,num_samples=2562)
        meshes2 = self.subdivmesh(mesh)
        meshes2 = self.subdivmesh(meshes2)
        meshes2 = self.subdivmesh(meshes2)
        pointlist2, facelist2 = (meshes2.verts_list(), meshes2.faces_list())
        for i in range(len(pointlist)):
            pointlist2[i] = pointlist2[i].unsqueeze(0)
            facelist2[i] = facelist2[i].unsqueeze(0)
        t0 = time.time()
        pointlist2 = self.refine(pointlist2, points, samples)
        t1 = time.time()
        return pointlist2, facelist2
        
    def list_to_mesh(self, pointlist, facelist, points, L_points = None):
        meshlist = []
        for i in range(len(pointlist)):
            meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))
        meshlist = join_meshes_as_batch(meshlist)
        L_lap = mesh_laplacian_smoothing(meshes=meshlist, method='uniform')
        if L_points is not None:
            pts = sample_points_from_meshes(meshes=meshlist,num_samples=30000)
            L_cd,_ = chamfer_distance(x=L_points,y=pts)
        else:
            pts = sample_points_from_meshes(meshes=meshlist,num_samples=2562)
            L_cd, _ = chamfer_distance(x=points,y=pts)
        # L_n = mesh_edge_loss(meshlist)
        L_n = normal_cons(meshlist)
        return L_cd, L_lap, meshlist, pts, L_n   

class SubdivRefine(nn.Module):
    def __init__(self) -> None:
        super(SubdivRefine, self).__init__()
        self.corase_mesh = Local_Fea_Deform4(stage=1).float()
        
        self.refine = mlp_refine().float()
        self.subdivmesh = SubdivideMeshes().float()
        # unet3d_kwargs={'num_levels': 3,
        #                 'f_maps': 32,'in_channels': 32,
        #                 'out_channels': 32}
        # self.unet3d = UNet3D(**unet3d_kwargs)
        load_partial_pretrained(self.corase_mesh,'./train_models/lip_norm_and_lap_smooth')
        
    def forward(self, points):
        pred, tetra, w, grid = self.corase_mesh(points)
        _, _, _, _, pointlist, facelist = sli(pred, tetra, torch.zeros_like(w).cuda())
        L_cd1, L_lap1, meshes2, samples,_ = self.list_to_mesh(pointlist, facelist, points)
        
        meshes2 = self.subdivmesh(meshes2)
        meshes2 = self.subdivmesh(meshes2)
        meshes2 = self.subdivmesh(meshes2)
        pointlist2, facelist2 = (meshes2.verts_list(), meshes2.faces_list())
        for i in range(len(pointlist)):
            pointlist2[i] = pointlist2[i].unsqueeze(0)
            facelist2[i] = facelist2[i].unsqueeze(0)
        pointlist2 = self.refine(pointlist2, points, samples)
        
        L_cd2, L_lap2, _, _, L_n = self.list_to_mesh(pointlist2, facelist2, points)
        
        return L_cd1, L_lap1, L_cd2, L_lap2, pointlist2, facelist2,L_n
    
    def list_to_mesh(self, pointlist, facelist, points):
        meshlist = []
        for i in range(len(pointlist)):
            meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))
        meshlist = join_meshes_as_batch(meshlist)
        L_lap = mesh_laplacian_smoothing(meshes=meshlist, method='uniform')
        pts = sample_points_from_meshes(meshes=meshlist,num_samples=2562,return_normals=True)
        L_cd, _ = chamfer_distance(x=points,y=pts)
        L_n = mesh_edge_loss(meshlist)
        return L_cd, L_lap, meshlist, pts, L_n        

class Local_Fea_Deform4(nn.Module):
    
    def __init__(self, cdim = 32, hiddendim = 32, stage = 1):
        super(Local_Fea_Deform4, self).__init__()

        print("Local_Fea_Deform embedding initialized")

        self.encoder = LocalPoolPointnet(c_dim=cdim, dim=3, 
                        hidden_dim=hiddendim, scatter_type='max', 
                        unet3d=True, unet3d_kwargs={'num_levels': 3,
                                    'f_maps': 32,'in_channels': 32,
                                    'out_channels': 32}, 
                        grid_resolution=32, plane_type='grid',
                        padding=0.1, n_blocks=5).float()

        self.decoder = LocalDecoder(dim=4, c_dim=cdim,
                        hidden_size=hiddendim, n_blocks=5, 
                        leaky=False, sample_mode='bilinear', 
                        padding=0.1).float()
        
        self.loc = PointNetMix(zdim=1024)
        self.stage = stage
        if stage == 2:
            load_partial_pretrained(self.encoder, './train_models/encoder')
    def forward(self, p):
        if self.stage==2:
            with torch.no_grad():
                    grid = self.encoder(p)
        else:
            grid = self.encoder(p)
        out, face = self.decoder(p,grid)
        w = self.loc(p)
        
        # if self.stage == 1:
            
        #     out = out.clone() - torch.mean(out.clone(), axis=1, keepdim=True)
        #     scale_max = torch.max(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        #     scale_min = -torch.min(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        #     scale = torch.where(scale_max>scale_min, scale_min, scale_max)
            
        #     noisy = torch.nonzero(scale<1e-10)
        #     if noisy.shape[0]!=0:
        #         print('zero all')
        #         out[noisy[:0],:,3] = torch.rand(noisy.shape[0],out.shape[1]).cuda()
            
        #     out[:,:,3] = (out[:,:,3].clone() / (scale+1e-10)) *1.2

      
        return out, face, w, grid 
    
    
    
class PointNet(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(PointNet, self).__init__()
        
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, zdim, 1)


        self.fc11 = nn.Linear(zdim*2, 512)
        self.fc12 = nn.Linear(512, 1)

        self.fc21 = nn.Linear(zdim*2, 512)
        self.fc22 = nn.Linear(512, 1)
        

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
        
        num_t = F.leaky_relu(self.fc11(mix),0.2,inplace=True)
        num_t = self.fc12(num_t)
        num_t = (torch.sigmoid(num_t)).float()
        w = F.leaky_relu(self.fc21(mix),0.2,inplace=True)
        w = self.fc22(w)
        w = torch.tanh(w)
        
        return num_t, w



class cat_embending(nn.Module):
    def __init__(self,unet3d_kwargs={'num_levels': 3,'f_maps': 32,'in_channels': 32,'out_channels': 32}):
        super(cat_embending, self).__init__()
        
        self.deform = LocalDecoder(dim=4, c_dim=32, hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1).float()

        self.v, self.t = read_4obj('./model/tour_big.4obj')
        self.t = torch.from_numpy(self.t).long()
        # self.v /= np.max(np.linalg.norm(self.v, axis=1))
        self.v = torch.from_numpy(self.v) 
        self.unet3d = UNet3D(**unet3d_kwargs)
        self.locate = PointNet(zdim=512)
        
    def forward(self, p, num_tpe=None):
        vertices = self.v.unsqueeze(0).repeat(p.size(0),1,1).type_as(p)
        face = self.t.unsqueeze(0).repeat(p.size(0),1,1).type_as(p)
        num_tpe, w = self.locate(p)
        fea_grid = torch.ones(p.size(0),32,32,32,32).cuda()* num_tpe.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,32,32,32,32)
        fea_grid = self.unet3d(fea_grid)
        fea = {}
        fea['grid'] = fea_grid
        out, face , _ = self.deform(p,fea)
        out = out[-1]
        
        out = out.clone() - torch.mean(out.clone(), axis=1, keepdim=True)
        scale_max = torch.max(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        scale_min = -torch.min(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        scale = torch.where(scale_max>scale_min, scale_min, scale_max)
        
        noisy = torch.nonzero(scale<1e-10)
        if noisy.shape[0]!=0:
            print('zero all')
            out[noisy[:0],:,3] = torch.rand(noisy.shape[0],out.shape[1]).cuda()
        
        out[:,:,3] = (out[:,:,3].clone() / (scale+1e-10)) *1.2
        # out = out/torch.max(out[:,:,3])
        return out, face, num_tpe, w
    
    def generate(self, p, num_tpe=None):

        fea_grid = torch.ones(p.size(0),32,32,32,32).cuda()* num_tpe.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,32,32,32,32)
        fea_grid = self.unet3d(fea_grid)
        fea = {}
        fea['grid'] = fea_grid
        out, face , _ = self.deform(p,fea)
        out = out[-1]
        
        out = out.clone() - torch.mean(out.clone(), axis=1, keepdim=True)
        scale_max = torch.max(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        scale_min = -torch.min(out[:,:,3].clone(),dim=1)[0].view(p.shape[0],1)
        scale = torch.where(scale_max>scale_min, scale_min, scale_max)
        
        noisy = torch.nonzero(scale<1e-10)
        if noisy.shape[0]!=0:
            print('zero all')
            out[noisy[:0],:,3] = torch.rand(noisy.shape[0],out.shape[1]).cuda()
        
        out[:,:,3] = (out[:,:,3].clone() / (scale+1e-10)) *1.2
        # out = out/torch.max(out[:,:,3])
        
        return out, face

class LocalSubdiv(nn.Module):
    def __init__(self) -> None:
        super(LocalSubdiv, self).__init__()
        self.corase_mesh = Local_Fea_Deform4(stage=1).float()
        
        self.refine = refine().float()
        self.subdivmesh = SubdivideMeshes().float()
        load_partial_pretrained(self.corase_mesh,'./train_models/')
        
    def forward(self, points, L_points=None):
        with torch.no_grad():
            pred, tetra, jacobian, w, grid = self.corase_mesh(points, not_jacobian = True)
        _, _, _, _, pointlist, facelist = sli(pred, tetra, (w).cuda())
        L_cd1, L_lap1, mesh, samples,_ = self.list_to_mesh(pointlist, facelist, points)
        
        meshes2 = self.subdivmesh(mesh)
        meshes2 = self.subdivmesh(meshes2)
        meshes2 = self.subdivmesh(meshes2)
        pointlist2, facelist2 = (meshes2.verts_list(), meshes2.faces_list())
        for i in range(len(pointlist)):
            pointlist2[i] = pointlist2[i].unsqueeze(0)
            facelist2[i] = facelist2[i].unsqueeze(0)
        pointlist2 = self.refine(pointlist2, points, samples)
        
        L_cd2, L_lap2, _, _, L_n = self.list_to_mesh(pointlist2, facelist2, points, L_points)
        
        return L_cd1, L_lap1, L_cd2, L_lap2, pointlist2, facelist2,L_n
    
        
    def test(self, points):
        pred, tetra, jacobian, w, grid = self.corase_mesh(points, not_jacobian = True)
        _, _, _, _, pointlist, facelist = sli(pred, tetra, torch.zeros_like(w).cuda())
        meshlist = []
        for i in range(len(pointlist)):
            meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))
        mesh = join_meshes_as_batch(meshlist)
        samples = sample_points_from_meshes(meshes=mesh,num_samples=2562)
        meshes2 = self.subdivmesh(mesh)
        meshes2 = self.subdivmesh(meshes2)
        meshes2 = self.subdivmesh(meshes2)
        pointlist2, facelist2 = (meshes2.verts_list(), meshes2.faces_list())
        for i in range(len(pointlist)):
            pointlist2[i] = pointlist2[i].unsqueeze(0)
            facelist2[i] = facelist2[i].unsqueeze(0)
        pointlist2 = self.refine(pointlist2, points, samples)
        return pointlist2, facelist2
        
    def list_to_mesh(self, pointlist, facelist, points, L_points = None):
        meshlist = []
        for i in range(len(pointlist)):
            meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))
        meshlist = join_meshes_as_batch(meshlist)
        L_lap = mesh_laplacian_smoothing(meshes=meshlist, method='uniform')
        if L_points is not None:
            pts = sample_points_from_meshes(meshes=meshlist,num_samples=30000)
            L_cd,_ = chamfer_distance(x=L_points,y=pts)
        else:
            pts = sample_points_from_meshes(meshes=meshlist,num_samples=2562)
            L_cd, _ = chamfer_distance(x=points,y=pts)
        # L_n = mesh_edge_loss(meshlist)
        L_n = normal_cons(meshlist)
        return L_cd, L_lap, meshlist, pts, L_n  

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
    
class DeformGen(nn.Module):
    def __init__(self,zdim):
        super(DeformGen, self).__init__()
        
        self.v, self.t = read_4obj('./model/tour_small.4obj')
        self.t = torch.from_numpy(self.t).long()
        # self.v /= np.max(np.linalg.norm(self.v, axis=1))
        self.v = torch.from_numpy(self.v) 
        self.zdim = zdim
        self.conv1 = nn.Conv1d(4+zdim, zdim, 1)
        self.conv2 = nn.Conv1d(zdim, zdim, 1)
        self.conv3 = nn.Conv1d(zdim, zdim, 1)
        self.conv4 = nn.Conv1d(zdim, zdim, 1)
        
        self.conv5 = nn.Conv1d(zdim, 64, 1)
        self.conv6 = nn.Conv1d(64, 4, 1)


        self.bn1 = nn.BatchNorm1d(zdim)
        self.bn2 = nn.BatchNorm1d(zdim)
        self.bn3 = nn.BatchNorm1d(zdim)
        self.bn4 = nn.BatchNorm1d(zdim)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, latent):
        
        vertices = self.v.unsqueeze(0).repeat(latent.size(0),1,1).type_as(latent)
        face = self.t.unsqueeze(0).repeat(latent.size(0),1,1).type_as(latent)

        x = vertices.transpose(1,2)

        latent = latent.repeat(1,self.zdim,x.size()[2])
        x = torch.cat((latent,x),dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2,inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2,inplace=True) 
        x = F.leaky_relu(self.conv3(x), 0.2,inplace=True) 
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2,inplace=True) 
        
        x = F.leaky_relu(self.conv5(x), 0.2,inplace=True) 
        x = torch.tanh(self.conv6(x))

        out = x.transpose(1,2)
        

        out = out.clone() - torch.mean(out.clone(), axis=1, keepdim=True)
        # scale_max = torch.max(out[:,:,3].clone(),dim=1)[0].view(x.shape[0],1)
        # scale_min = -torch.min(out[:,:,3].clone(),dim=1)[0].view(x.shape[0],1)
        # scale = torch.where(scale_max>scale_min, scale_min, scale_max)
        
        # noisy = torch.nonzero(scale<1e-10)
        # if noisy.shape[0]!=0:
        #     print('zero all')
        #     out[noisy[:0],:,3] = torch.rand(noisy.shape[0],out.shape[1]).cuda()
        
        # out[:,:,3] = (out[:,:,3].clone() / (scale+1e-10)) *1.5
        
        
        return out, face


        

class Graph_smooth(nn.Module):

    def __init__(self, hidden_size=32):
        super(Graph_smooth, self).__init__()
        
        self.Gconv1 = GraphConv(4,hidden_size)
        self.Gconv2 = GraphConv(hidden_size,hidden_size)
        self.Gconv3 = GraphConv(hidden_size,hidden_size)
        self.Gconv4 = GraphConv(hidden_size,4)
        self.edge = read_ed('./model/EDGE.ed')
        self.edge = torch.from_numpy(self.edge)

    def forward(self, points, cs):

        edge = self.edge.type_as(points).long()

        outs = []
        for i in range(points.shape[0]):
            c = cs[i:i+1]
            x = F.relu(self.Gconv1(points[i],edge))
            x1 = x.unsqueeze(0) + c
            x2 = F.relu(self.Gconv2(x1[0],edge)) + x1[0]
            x3 = F.relu(self.Gconv3(x2,edge)) + x2
            x4 = torch.tanh(self.Gconv4(x3,edge))
            outs.append(x4.unsqueeze(0))
        out = torch.cat((outs),dim=0)

        return out[:,:,0:4]




        
def read_ed(filename):

    fin = open(str(filename))
    lines = fin.readlines()
    fin.close()

    edge = []

    for i in range(1,len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        x = float(line[0])
        y = float(line[1])
        edge.append([x,y])

    edge = np.array(edge, np.long)

    return edge
