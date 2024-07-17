import trimesh
import torch
from src.layers import ResnetBlockFC
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate, normalize_coordinate
from model.model_atlas_tiger_4d import read_4obj
import torchvision.models as models

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

class Map(nn.Module):
    def __init__(self, zdim, point_dim=4):
        super(Map, self).__init__()

        self.conv1 = nn.Conv1d((point_dim+zdim), (point_dim+zdim)//2, 1)
        self.conv2 = nn.Conv1d((point_dim+zdim)//2, (point_dim+zdim)//4, 1)
        self.conv3 = nn.Conv1d((point_dim+zdim)//4, (point_dim+zdim)//8, 1)
        self.conv4 = nn.Conv1d((point_dim+zdim)//8, 3, 1)

        self.bn1 = nn.BatchNorm1d((point_dim+zdim)//2)
        self.bn2 = nn.BatchNorm1d((point_dim+zdim)//4)
        self.bn3 = nn.BatchNorm1d((point_dim+zdim)//8)

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

class Locating(nn.Module):

    def __init__(self, zdim=256):
        super(Locating, self).__init__()

        self.v, self.t = read_4obj('./model/tour_small.4obj')
        # self.v, self.t = read_4obj('./model/tiger.4obj')
        # self.v, self.t = read_4obj('./model/sphere.4obj')
        self.t = torch.from_numpy(self.t).long()
        self.v /= np.max(np.linalg.norm(self.v, axis=1))

        self.v = torch.from_numpy(self.v) 
        print(np.max(np.linalg.norm(self.v, axis=1)))
        self.encoder = PointNet(zdim).float()  
        self.decoder = Map(zdim=zdim).float()
        # self.imgencoder = models.resnet18(pretrained=True)
        # self.linear = nn.Linear(1000, zdim)

    def forward(self, input, type):
        
        batch_size = input.shape[0]
        if type == 'points':
            latent = self.encoder(input)
        else:
            latent = self.imgencoder(input)
            latent = self.linear(latent)
        vertices = self.v.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
        face = self.t.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
        outs = self.decoder(vertices,latent.unsqueeze(2))

        return outs, vertices, face, latent.unsqueeze(2)

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=8, c_dim=128,hidden_size=256, n_blocks=4, 
                 leaky=False, sample_mode='bilinear', padding=0.1, 
                 zdim = 1000):
        super().__init__()
        self.c_dim = c_dim
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks

        self.dim = dim
        
        self.locate = Locating(256)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.BatchNorm = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size,dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        self.c = nn.Parameter(torch.ones(1))
        # self.ini()
        self.sp = nn.Softplus()

    def lip_norm(self, c=1.2):
        for i in range(len(self.blocks)):
            self.fc_c[i].weight.data = self.normalization(self.fc_c[i].weight, c)
            self.blocks[i].fc_0.weight.data = self.normalization(self.blocks[i].fc_0.weight, c)
            self.blocks[i].fc_1.weight.data = self.normalization(self.blocks[i].fc_1.weight, c)
            if self.c_dim != self.hidden_size:
                self.blocks[i].shortcut.weight.data = self.normalization(self.blocks[i].shortcut.weight, c)

        self.fc_p.weight.data = self.normalization(self.fc_p.weight, c)
        self.fc_out.weight.data = self.normalization(self.fc_out.weight, c)
        

    def normalization(self, W, c):
        absrowsum = torch.sum(torch.abs(W),dim=1)
        scale = torch.minimum(torch.ones((1)).cuda(), c/absrowsum)
        return W * scale[:, None]
    
    def ini(self):
        for i in range(self.n_blocks):
           nn.init.zeros_(self.fc_c[i].weight)
           nn.init.zeros_(self.fc_c[i].bias)
        nn.init.zeros_(self.fc_p.weight)
        nn.init.zeros_(self.fc_p.bias)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        
    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c
    
    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, type = 'points'):
    
        Lo, vertices, face, latent = self.locate(p, type)
        

        vertices.requires_grad = True
        if type == 'image':
            c = self.sample_plane_feature(Lo, c_plane[0], plane='xz')

            c += self.sample_plane_feature(Lo, c_plane[1], plane='xy')

            c += self.sample_plane_feature(Lo, c_plane[2], plane='yz')
            
            c /= 3
        else:
            c = self.sample_grid_feature(Lo, c_plane['grid'])
            
        c = c.transpose(1, 2)
        self.lip_norm()
        out = self.deform(vertices, c)
        # out = out +vertices
        # self.test_jacobian(c)

        return out, face

    def test_jacobian(self, c):
        sph  = trimesh.load('./model/mypymeshsph_4.obj')
        v = torch.from_numpy(sph.vertices).cuda().unsqueeze(0).repeat(32,1,1).float()
        f = sph.faces
        points1 = torch.rand((32,3739,4)).cuda()/2
        points2 = torch.rand((32,3739,4)).cuda()/2
        # points[:,0:2562,1:4] = v
        points1.requires_grad = True
        points2.requires_grad = True
        out1 = self.deform(points1, c)
        out1 = out1 + points1
        
        out2 = self.deform(points2, c)
        out2 = out2 + points2
        a = 1
        # for i in range(50):
        #     out = points+((i/50)*out1)
        #     sph = trimesh.Trimesh(
                #vertices=out[0,0:2562,1:4].detach().cpu().numpy(), 
                # faces=f)
        #     sph.export('./interlotation/'+str(i)+'.obj')
        jacobian_eigenvalue, jacobian_det  = jacobian(points1, out1)
        # jacobian_eigenvalue, jacobian_det  = self.jacobian(points, out)
        
    def deform(self, vertices, c):
        
        net = self.fc_p(vertices)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        # return self.fc_out(self.actvn(net))
        return torch.tanh(self.fc_out(self.actvn(net)))

    
def jacobian(vertices, out):
    u, v, w, t = torch.split(out, 1, dim=2)
    
    du = torch.autograd.grad(u, vertices, 
                    grad_outputs = torch.ones_like(u), 
                    retain_graph = True,
                    create_graph = True)[0]
    
    ddu = torch.autograd.grad(du[:,:,0:1], vertices, 
                    grad_outputs = torch.ones_like(u), 
                    retain_graph = True,
                    create_graph = True)[0]
    
    dv = torch.autograd.grad(v, vertices, 
                    grad_outputs = torch.ones_like(v), 
                    retain_graph = True,
                    create_graph = True)[0]
    
    dw = torch.autograd.grad(w, vertices, 
                    grad_outputs = torch.ones_like(w), 
                    retain_graph = True,
                    create_graph = True)[0]
    
    dt = torch.autograd.grad(t, vertices, 
                    grad_outputs = torch.ones_like(t), 
                    retain_graph = True,
                    create_graph = True)[0]
    
    jacobian = torch.cat((du, dv, dw, dt),dim=2).view(-1, vertices.shape[1], 4, 4)
    
    jacobian_det = torch.linalg.det(jacobian)
    jacobian_eigenvalue = torch.linalg.eigvals(jacobian.view(-1,4,4))
    # jacobian_eigensq = jacobian_eigenvalue.real**2 + jacobian_eigenvalue.imag**2
            
    return jacobian_eigenvalue, jacobian_det
    
class mlp_refine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from model.model_atlas_tiger_4d import PointNet, MappingTemplet
        self.encoder1 = PointNet(zdim=512)
        self.encoder2 = PointNet(zdim=512)
        self.decoder = MappingTemplet(zdim=512, point_dim=3)
        
    def forward(self, pointlist, inputs, samples):

        shapelist, points = self.pre_process_pointlist(pointlist)
        latent1 = self.encoder1(samples)
        latent2 = self.encoder2(inputs)
        latent = latent1 + latent2
        res = self.decoder(points,latent.unsqueeze(2))

        points = points + res
        pointlist = self.post_process_pointlist(points, shapelist)
        
        return pointlist
        
    def post_process_pointlist(self, points, shapelist):
        pointlist = []
        batch_size = points.shape[0]
        for i in range(batch_size):
            pointlist.append(points[i:i+1,0:shapelist[i],:])
        return pointlist
            
    def pre_process_pointlist(self, pointlist):
        batchsize = len(pointlist)
        shapelist = []
        for i in range(batchsize):
            shapelist.append(pointlist[i].shape[1])
        max_points = max(shapelist)
        points = torch.zeros(batchsize, max_points, 3).type_as(pointlist[0])
        for i in range(batchsize):
            points[i,0:shapelist[i],:] = pointlist[i]
        
        return shapelist, points
        
    def deform(self, vertices, c):
        
        net = self.fc_p(vertices)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        # return self.fc_out(self.actvn(net))
        return torch.tanh(self.fc_out(self.actvn(net)))

class refine(nn.Module):
    def __init__(self, dim=3, c_dim=32,
                 hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, zdim = 1000):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        from src.encoder.pointnet import LocalPoolPointnet
        self.encoder1 = LocalPoolPointnet(c_dim=32, dim=3, 
                hidden_dim=32, scatter_type='max', 
                unet3d=True, unet3d_kwargs={'num_levels': 3,
                            'f_maps': 32,'in_channels': 32,
                            'out_channels': 32}, 
                grid_resolution=32, plane_type='grid', 
                padding=0.1, n_blocks=5).float()
        self.encoder2 = LocalPoolPointnet(c_dim=32, dim=3, 
                hidden_dim=32, scatter_type='max', 
                unet3d=True, unet3d_kwargs={'num_levels': 3,
                            'f_maps': 32,'in_channels': 32,
                            'out_channels': 32}, 
                grid_resolution=32, plane_type='grid', 
                padding=0.1, n_blocks=5).float()
        self.dim = dim

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size,ini=True) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size,dim)
        # torch.nn.init.zeros_(self.fc_p.weight)
        # torch.nn.init.zeros_(self.fc_p.bias)
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        # self.ini()

    def ini(self):
        for i in range(self.n_blocks):
           nn.init.zeros_(self.fc_c[i].weight)
           nn.init.zeros_(self.fc_c[i].bias)
        nn.init.zeros_(self.fc_p.weight)
        nn.init.zeros_(self.fc_p.bias)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        
    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c
        
    def forward(self, pointlist, inputs, samples):
        c_sample = self.encoder1(samples)
        c_plane = self.encoder2(inputs)
        shapelist, points = self.pre_process_pointlist(pointlist)
        c_plane['grid'] = c_plane['grid']+c_sample['grid']
        c = self.sample_grid_feature(points, c_plane['grid'])
        c = c.transpose(1, 2)

        points = self.deform(points, c)
        points = points
        pointlist = self.post_process_pointlist(points, shapelist)
        
        return pointlist
        
    def post_process_pointlist(self, points, shapelist):
        pointlist = []
        batch_size = points.shape[0]
        for i in range(batch_size):
            pointlist.append(points[i:i+1,0:shapelist[i],:])
        return pointlist
            
    def pre_process_pointlist(self, pointlist):
        batchsize = len(pointlist)
        shapelist = []
        for i in range(batchsize):
            shapelist.append(pointlist[i].shape[1])
        max_points = max(shapelist)
        points = torch.zeros(batchsize, max_points, 3).type_as(pointlist[0])
        for i in range(batchsize):
            points[i,0:shapelist[i],:] = pointlist[i]
        
        return shapelist, points
        
    def deform(self, vertices, c):
        
        net = self.fc_p(vertices)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        # return self.fc_out(self.actvn(net))
        return torch.tanh(self.fc_out(self.actvn(net)))


class HyperNets(nn.Module):
    def __init__(self, c_dim = 32, mapping_layer = 10):
        super(HyperNets, self).__init__()
        
        self.mapping_layers = mapping_layer
        
        self.linears = nn.Sequential(nn.Linear(c_dim, 512),
                                     nn.ReLU(),
                                     nn.Linear(512,512),
                                     nn.ReLU(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     )
        self.linear_weights = nn.ModuleList([nn.Linear(512, 20) for _ in range(mapping_layer)])
        
    def forward(self, fea):
        weightlist = []
        fea = self.linears(fea)
        
        for i in range(self.mapping_layers):
            weightlist.append(self.linear_weights[i](fea))
            
        return weightlist
    
class InjectiveMapping(nn.Module):
    def __init__(self, mapping_layer = 10):
        super(InjectiveMapping, self).__init__()
        
        self.mapping_layers = mapping_layer
        self.act = nn.LeakyReLU(0.2)

    def forward(self, vertices, weightslist):        
        vertices = vertices[:,:,None,:]
        num_p = vertices.shape[1]
        (pointslist, detlist) = ([], [])
        
        for i in range(self.mapping_layers):
            matrix_weights = weightslist[i][:,0:16].view(-1, 1, 4, 4).repeat(1,num_p,1,1)
            detlist.append(torch.linalg.det(matrix_weights)[:,:,None])
            bias_weights = weightslist[i][:,16:20].view(-1,1,1,4).repeat(1,num_p,1,1)
            vertices = self.act((vertices @ matrix_weights) +bias_weights)
            if i < (self.mapping_layers-1) // 2:
                pointslist.append(vertices)
            elif i > self.mapping_layers // 2:
                vertices += pointslist.pop()
            else:
                continue
            
        return vertices.squeeze(2), detlist
# points = torch.rand(32,1000,4)           
# fea = torch.rand(32,1000,32)
# N = HyperNets()
# M = InjectiveMapping()   
# a = N(fea) 
# v = M(points, a)
# a = 1   


class HyperDeform(nn.Module):
    def __init__(self, zdim=1024, mapping_layer = 10):
        super(HyperDeform, self).__init__()

        self.zdim = zdim
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, zdim, 1)
        self.v, self.t = read_4obj('./model/tour_small.4obj')
        self.t = torch.from_numpy(self.t).long()
        # self.v /= np.max(np.linalg.norm(self.v, axis=1))
        self.v = torch.from_numpy(self.v) 

        self.fc1 = nn.Linear(zdim*2, zdim)
        
        self.mapping_layer = mapping_layer
        self.hypernet = HyperNets(c_dim=self.zdim, mapping_layer=self.mapping_layer)
        self.injective_mapping = InjectiveMapping(mapping_layer=self.mapping_layer)


    
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

        vertices = self.v.unsqueeze(0).repeat(x.shape[0],1,1).type_as(x)
        face = self.t.unsqueeze(0).repeat(x.shape[0],1,1).type_as(x)
        
        paramlist = self.hypernet(ms)
        
        vertices, detlist = self.injective_mapping(vertices, paramlist)
        
        return vertices, face, detlist