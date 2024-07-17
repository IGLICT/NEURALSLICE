from re import T
from pytorch3d import loss
import numpy as np
import torch
import os
import random
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_gather
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.io import save_obj
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d import _C
import torch.nn.functional as F
# from model.model_atlas_tiger_4d import deform2
import torch
import torch.nn as nn

def normal_loss(points, gt, points_n, gt_n):
    _, idx1, _ = knn_points(points, gt)
    _, idx2, _ = knn_points(gt, points)
    normal2 = knn_gather(points_n,idx2).squeeze(2)
    normal1 = knn_gather(gt_n,idx1).squeeze(2)
    loss1 = torch.mean(1 - (normal1[:,:,None,:] @ points_n[:,:,:,None])**2)
    loss2 = torch.mean(1 - (normal2[:,:,None,:] @ gt_n[:,:,:,None])**2)
    loss = loss1 + loss2
    return loss / 2
    
    
def normal_cons(meshlist):
    
    points, points_n = sample_points_from_meshes(meshes=meshlist,num_samples=50000,return_normals=True)
    gt, gt_n = sample_points_from_meshes(meshes=meshlist,num_samples=50000,return_normals=True)
            
    _, idx1, _ = knn_points(points, gt)
    _, idx2, _ = knn_points(gt, points)
    normal2 = knn_gather(points_n,idx2).squeeze(2)
    normal1 = knn_gather(gt_n,idx1).squeeze(2)
    loss1 = torch.mean(torch.abs(torch.cosine_similarity(normal1,points_n)))
    loss2 = torch.mean(torch.abs(torch.cosine_similarity(normal2,gt_n)))
    loss = loss1 + loss2
    return loss 

def process_points(x,reso):
    x = x/2
    x += 0.5
    x = (x * reso).long()
    index_x = x[:,:,0] + reso * (x[:,:,1]+ reso * x[:,:,2])
    index_num_x = index_x.cpu().numpy()
    index_x = index_x[:,:,None]

    return index_x, index_num_x



def recurrent_chamfer_distance(x, y, reso):
    index_x, index_num_x = process_points(x, reso)
    index_y, index_num_y = process_points(y, reso)
    cnt=0
    totloss = 0
    for i in range(x.size(0)):

        mask_x = np.unique(index_num_x[i:i+1])
        mask_x = torch.from_numpy(mask_x).view(1,1,-1).cuda()
        mask_y = np.unique(index_num_y[i:i+1])
        mask_y = torch.from_numpy(mask_y).view(1,1,-1).cuda()
        cnt=0
        totloss = 0
        mask_x = mask_x.repeat(1,index_x.size(1),1)

        for j in range(mask_x.size(2)):
            a = torch.nonzero(index_x[i:i+1,:,:]==mask_x[:,:,j:j+1])
            b = torch.nonzero(index_y[i:i+1,:,:]==mask_x[:,:,j:j+1])
            points_x = x[a[:,0],a[:,1],:]
            points_y = y[b[:,0],b[:,1],:]
            if points_y.size(0)==0 or points_x.size(0)==0:
                continue
            cnt+=1
            loss,_ = chamfer_distance(points_x[None,:,:],points_y[None,:,:])
            totloss += loss

    if cnt==0:
        return 0
        
    return totloss/cnt
        

def random_mask(reso, prob):

    index_all = torch.arange(0,reso**3).cuda()
    index = torch.randperm(index_all.size(0))
    mask = index_all[index]
    return mask[0:int(len(mask)*prob)]

def generate_masked_points(orx, reso, prob):

    x = orx
    x /= 2
    x += 0.5
    x = (x * reso).long()
    index = x[:,:,0] + reso * (x[:,:,1] + reso * x[:,:,2])
    index_num = index.cpu().numpy()
    index = index[:,:,None]
    masklist = []
    maskedlist = []
    min_p = 9e20

    for i in range(x.size(0)):
       
        mask = np.unique(index_num[i:i+1])
        mask = torch.from_numpy(mask).cuda()
        all = torch.arange(0, mask.size(0)).cuda()
        index_m = torch.randperm(all.size(0))
        mask = mask[index_m]
        mask = mask[0:int(len(mask)*prob)].view(1,1,-1)
        masklist.append(mask)
        mask = mask.repeat(1,index.size(1),1)

        a = torch.nonzero(index[i:i+1,:,:]==mask).cuda()
        maskedlist.append(orx[a[:,0],a[:,1],:])
        if a.size(0)<min_p:

            min_p = len(a)
    
    index = torch.arange(min_p).cuda()
    index = torch.randperm(index.size(0))

    for i in range(x.size(0)):
        
        maskedlist[i] = maskedlist[i][index].unsqueeze(0)

    return torch.cat(maskedlist, dim = 0), masklist
def laplace_tetra(verts, faces):
    
    term = torch.zeros_like(verts)
    norm = torch.zeros_like(verts[..., 0:1])

    v0 = verts[faces[:,0],:]
    v1 = verts[faces[:,1],:]
    v2 = verts[faces[:,2],:]
    v3 = verts[faces[:,3],:]

    term.scatter_add_(0, faces[:,0:1].repeat(1,4), (v1-v0) + (v2-v0) + (v3-v0))
    term.scatter_add_(0, faces[:,1:2].repeat(1,4), (v0-v1) + (v2-v1) + (v3-v1))
    term.scatter_add_(0, faces[:,2:3].repeat(1,4), (v0-v2) + (v1-v2) + (v3-v2))
    term.scatter_add_(0, faces[:,3:4].repeat(1,4), (v0-v3) + (v1-v3) + (v2-v3))

    thr = torch.ones_like(v0) * 3.0
    norm.scatter_add_(0, faces[:, 0:1], thr)
    norm.scatter_add_(0, faces[:, 1:2], thr)
    norm.scatter_add_(0, faces[:, 2:3], thr)
    norm.scatter_add_(0, faces[:, 3:4], thr)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)


class slice(nn.Module):
    def __init__(self, zdim = 1000, alpha=0,epsilon=0.0001):
        super(slice,self).__init__()

        self.alpha = alpha
        self.epsilon = epsilon
        # self.deform = deform2(zdim=zdim).float()

    def forward(self, inputs, pred, tetra, latent = None, stage=None, deformer=None):

        tetra = tetra.long()
        pred = pred - torch.mean(pred, axis=1, keepdim=True)
        # maxw = torch.max(pred[:,:,3])
        # minw = torch.min(pred[:,:,3])
        # pred[:,:,3] /= (maxw-minw)
        # pred = pred - torch.mean(pred, axis=1, keepdim=True)
        # pred[:,:,3] *= 3
        pointlist = []
        mesh_points = []
        facelist = []
        L1 = torch.zeros(1).cuda()
        L2 = torch.zeros(1).cuda()
        L3 = torch.zeros(1).cuda()
        L4 = torch.zeros(1).cuda()
        LossSlice = torch.mean(torch.max((self.epsilon-torch.abs(pred[:,:,3])),0)[0])

        for i in range(tetra.size()[0]):
            
            tetr = tetra[i]
            pre = pred[i]
            L1 += laplace_tetra(pre, tetr)

            p1 = pre[tetr[:,0],:].view(-1,1,4)
            p2 = pre[tetr[:,1],:].view(-1,1,4)
            p3 = pre[tetr[:,2],:].view(-1,1,4)
            p4 = pre[tetr[:,3],:].view(-1,1,4)
            
            Tetrah = torch.cat((p1,p2,p3,p4),dim=1)

            edge = torch.zeros(tetra.size()[1],6).type_as(pred)
            point = torch.zeros(tetra.size()[1],6,2,4).type_as(pred)

            edge[:,0] = (Tetrah[:,0,3]-self.alpha) * (Tetrah[:,1,3]-self.alpha)
            point[:,0,0,:] = Tetrah[:,0,:]
            point[:,0,1,:] = Tetrah[:,1,:]
            edge[:,1] = (Tetrah[:,0,3]-self.alpha) * (Tetrah[:,2,3]-self.alpha)
            point[:,1,0,:] = Tetrah[:,0,:]
            point[:,1,1,:] = Tetrah[:,2,:]
            edge[:,2] = (Tetrah[:,0,3]-self.alpha) * (Tetrah[:,3,3]-self.alpha)
            point[:,2,0,:] = Tetrah[:,0,:]
            point[:,2,1,:] = Tetrah[:,3,:]
            edge[:,3] = (Tetrah[:,1,3]-self.alpha) * (Tetrah[:,2,3]-self.alpha)
            point[:,3,0,:] = Tetrah[:,1,:]
            point[:,3,1,:] = Tetrah[:,2,:]
            edge[:,4] = (Tetrah[:,1,3]-self.alpha) * (Tetrah[:,3,3]-self.alpha)
            point[:,4,0,:] = Tetrah[:,1,:]
            point[:,4,1,:] = Tetrah[:,3,:]
            edge[:,5] = (Tetrah[:,2,3]-self.alpha) * (Tetrah[:,3,3]-self.alpha)
            point[:,5,0,:] = Tetrah[:,2,:]
            point[:,5,1,:] = Tetrah[:,3,:]

            index = torch.nonzero(edge<0)

            L2 += torch.mean(point[:,:,0,3]*point[:,:,1,3])
            
            L3 += torch.mean(
                (torch.norm(point[:,:,0,:]-point[:,:,1,:],dim = 2)-0.4)**2.0
                )

            m = point[index[:,0],index[:,1],0,0] - point[index[:,0],index[:,1],1,0]
            n = point[index[:,0],index[:,1],0,1] - point[index[:,0],index[:,1],1,1]
            p = point[index[:,0],index[:,1],0,2] - point[index[:,0],index[:,1],1,2]
            q = point[index[:,0],index[:,1],0,3] - point[index[:,0],index[:,1],1,3]

            x0 = point[index[:,0],index[:,1],0,0]
            y0 = point[index[:,0],index[:,1],0,1]
            z0 = point[index[:,0],index[:,1],0,2]
            w0 = point[index[:,0],index[:,1],0,3]

            x = (self.alpha-w0)*m/q + x0
            y = (self.alpha-w0)*n/q + y0
            z = (self.alpha-w0)*p/q + z0

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
            # mesh = Meshes(verts=point,faces=face)

            # try:
            #     pts = sample_points_from_meshes(mesh,num_samples=2562)
            #     LossCD1, _ = chamfer_distance(inputs[i:i+1],pts)
            #     Loss4 = mesh_edge_loss(mesh)
            # except:
            #     # save_obj('tttttttt.obj',point[0].cpu(), face[0].cpu())
            #     print('ERROR')
            #     LossCD1 = 0.4

            # L1 += LossCD1
            # L4 += Loss4
            
        L1 /= tetra.size()[0]
        L2 /= tetra.size()[0]
        L3 /= tetra.size()[0]
        # L4 /= tetra.size()[0]
  
        return L1, L2, L3, L4, mesh_points, facelist


def chamfer(x,y):

    N,P1,D = x.shape
    P2 = y.shape[1]
    completeness2,_,_ = knn_points(x,y,K=1)
    accuracy2,_,_ = knn_points(y,x,K=1)
    completeness2 = torch.mean(completeness2)
    accuracy2 = torch.mean(accuracy2)
    completeness = torch.sqrt(completeness2)
    accuracy = torch.sqrt(accuracy2)
    L1 = (completeness + accuracy)
    L2 = (completeness2 + accuracy2)
    
    return L1, L2

def loss_fun(pointlist, facelist, L3, occ, consis,input,occ_gt):

    meshlist = []
    for i in range(len(pointlist)):
        meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))
    meshlist = join_meshes_as_batch(meshlist)
    try:
        pts = sample_points_from_meshes(meshes=meshlist,num_samples=2562)
    except:
        print('1')
    L4 = mesh_edge_loss(meshlist)
    L1, _ = chamfer_distance(input,pts)

    loss_i = F.binary_cross_entropy_with_logits(
        occ, occ_gt, reduction='none')
    loss_occ = loss_i.mean()
    loss_consis = 0
    for i in range(len(pointlist)):
        loss_i = F.binary_cross_entropy_with_logits(
            consis[i], torch.ones(consis[i].shape).cuda()*0.5, reduction='none')
        loss_consis += loss_i.mean()
    loss_consis /= len(pointlist)
    return L1, loss_occ, L3, L4, loss_consis

# def normal_loss(p1,n1,p2,n2):
#     a = knn_points(p1,p2)
#     index1 = a[1]
#     a = knn_points(p2,p1)
#     index2 = a[1]

#     check = torch.sum(n1,dim=1)
#     index_normal = torch.nonzero(check<0)
#     n1[index_normal[:,0],:] = -n1[index_normal[:,0],:]
#     check = torch.sum(n2,dim=1)
#     index_normal = torch.nonzero(check<0)
#     n2[index_normal[:,0],:] = -n2[index_normal[:,0],:]
    
#     con2 = n1[0:1,index2[0,:,0],:]
#     con1 = n2[0:1,index1[0,:,0],:]
#     loss = torch.mean((con1-n1)**2) + torch.mean((con2-n2)**2)
#     return loss

def setup_seed(seed=3407):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def sli(pred, tetra, Balpha, trans = True, return_poly = False):

    tetra = tetra.long()
    if trans == True:
        pred = pred - torch.mean(pred, axis=1, keepdim=True)
    pointlist = []
    mesh_points = []
    facelist = []
    L1 = torch.zeros(1).cuda()
    L2 = torch.zeros(1).cuda()
    L3 = torch.zeros(1).cuda()
    L4 = torch.zeros(1).cuda()


    for i in range(tetra.size()[0]):
        
        alpha = Balpha[i]
        
        tetr = tetra[i]
        pre = pred[i]

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

        L2 += torch.mean(point[:,:,0,3]*point[:,:,1,3])
        
        L3 += torch.mean(
            (torch.norm(point[:,:,0,:]-point[:,:,1,:],dim = 2)-0.4)**2.0
            )

        m = point[index[:,0],index[:,1],0,0] - point[index[:,0],index[:,1],1,0]
        n = point[index[:,0],index[:,1],0,1] - point[index[:,0],index[:,1],1,1]
        p = point[index[:,0],index[:,1],0,2] - point[index[:,0],index[:,1],1,2]
        q = point[index[:,0],index[:,1],0,3] - point[index[:,0],index[:,1],1,3]+1e-20

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

        if return_poly:
                facelist.append([face3,edge4])
                mesh_points.append(point)
                continue

        face4[0:index_edge4.size()[0],:] = edge4[:,0:3]
        face4[index_edge4.size()[0]:index_edge4.size()[0]*2,:] = edge4[:,1:4]
        face4[index_edge4.size()[0]*2:index_edge4.size()[0]*3,0:2] = edge4[:,0:2]
        face4[index_edge4.size()[0]*2:index_edge4.size()[0]*3,2] = edge4[:,3]
        face4[index_edge4.size()[0]*3:index_edge4.size()[0]*4,0] = edge4[:,0]
        face4[index_edge4.size()[0]*3:index_edge4.size()[0]*4,1:3] = edge4[:,2:4]      

        face = torch.cat((face3,face4,face3),dim=0).view(1,-1,3)-1
        facelist.append(face)
        mesh_points.append(point)
        
    # L1 /= tetra.size()[0]
    L2 /= tetra.size()[0]
    L3 /= tetra.size()[0]
    # L4 /= tetra.size()[0]

    return L1, L2, L3, L4, mesh_points, facelist
