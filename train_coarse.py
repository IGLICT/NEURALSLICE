def _loss(points, N, pred, tetra, w):
    _, _, _, _, pointlist, facelist = sli(pred, tetra, w)
    meshlist = []
    for i in range(len(pointlist)):
        meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))

    meshlist = join_meshes_as_batch(meshlist)
    L_lap = mesh_laplacian_smoothing(meshes=meshlist, method='uniform')
    # L_nc = mesh_normal_consistency(meshlist)
    try:
        pts, normal = sample_points_from_meshes(meshes=meshlist,num_samples=2562,return_normals=True)
    except:
        print('error slice')

    Lcd, _ = chamfer_distance(x=points,y=pts)
    # Lnor = normal_loss(pts, points,  normal,N)
    return Lcd, L_lap
epslion = 0.0001
def train_AE(experiment, opt):
    # from src import config, data
    print("*********** INITIALIZATION  ************")

    torch.cuda.set_device(0)  
    encoder_type = 'point'   

    lrat = 1e-3
    training_generator = get_dataloader(encoder_type, opt)   
    validation_generator = get_dataloader(encoder_type, opt, split = 'val')   
    

    model = Local_Fea_Deform4(stage=1).to(device)
    print("*********** SETUP TRAINING  ************")
    optimizer = optim.Adam(model.parameters(), lr = lrat)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # load_partial_pretrained(model, './train_models/4DreconEpoch_590')
    # load_partial_pretrained(model,'./train_models/lip_norm_and_lap_smooth')
    
    os.makedirs(opt.model_folder,exist_ok=True)
    print("Models are being saved at :", opt.model_folder)
    if experiment is None: 
        print("Comet_ml logging is disabled, printing on terminal instead")
    print("*********** BEGIN TRAINING  ************")
    (step, epoch, stage) = (0, 0, 1)
    # scheduler.step(10)
    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    for epoch in range(0,opt.num_epochs):
        lloss, lloss2, sstep = (0,0,0)
        if epoch==15:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/1.25)    
        if epoch==20:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/1.5)
        if epoch==35:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/2)
        if epoch==55:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/4)
        if epoch==100:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/10)
        if epoch==125:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/50)
        if epoch==145:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/100)

        for data in training_generator:  
            optimizer.zero_grad()
        
            points = data[0].to(device).float()  
            N = data[1].to(device).float()

            (L1,L2,L3,L4) = (0,0,0,0)
            noise = 0.005 * torch.randn(points.shape[0],points.shape[1],3).to(device)
            pred, tetra, w, _ = model(points+noise) 
            L1, L_lap = _loss(points, N, pred, tetra, (w).cuda())
            loss = L1+L_lap*0.1
            loss = loss.requires_grad_()
            loss.backward()
            lloss += L1.item()
            lloss2 += L_lap.item()
    
            optimizer.step()

            if experiment is not None:
               
                experiment.log_metric("Total_Loss", loss.item(), step=step)
                experiment.log_metric("CD", L1.item(), step=step)
                experiment.log_metric("determinate", L2.item(), step=step)
                # experiment.log_metric("tetra_edge", L2.item(), step=step)

            step+=1
            sstep+=1
            # valid_loss = validate_training_AE(validation_generator,stage, model)

        # scheduler.step()   
        if epoch%10==0:
            valid_loss = validate_training_AE(validation_generator,epoch, model)

            if experiment is not None:
                experiment.log_metric("Valid_Loss", valid_loss, step=step-1)

            
            print("#Epoch:", epoch+1, 
                    "#Step: ", step-1,
                    "Valid Loss: {:.6f}".format(valid_loss))
             
            torch.save(model.state_dict(),opt.model_folder+'lip_epoch_{}'.format(epoch))
            
        print("CD Loss:",lloss/sstep,
              "laplacian Loss:", lloss2/sstep,
            #   " Jacobian Loss:", lloss2/sstep, 
              " Learning Rate:",optimizer.state_dict()['param_groups'][0]['lr'])

    print("Saving models after #Epoch :", epoch+1) 
    torch.save(model.state_dict(),opt.model_folder+'lip_epoch_{}'.format(epoch))
   
def save(folder, epoch, modelfile, vertices, faces):    
    for idx in range(len(vertices)):
        # Extract mesh to CPU using trimesh
        v = vertices[idx].view(-1,3).cpu().numpy()
        f = faces[idx].view(-1,3).cpu().numpy()
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        
        os.makedirs('{}/{}'.format(folder,epoch ), exist_ok=True)
        mesh.export('{}/{}/{}.obj'.format(folder,epoch ,modelfile[idx]));


def validate_training_AE(validation_generator, epoch, model):

    print("Validating and visualization model......")
    # with torch.no_grad():
    total_loss = 0
    items = 0.001
    is_vis = False
    for input, N, cat, modelfile in validation_generator: 

        points = input.to(device).float()
        pred, tetra,w,_ = model(points) 

        _, _, _, _, pointlist, facelist = sli(pred, tetra,(w).cuda())
        meshlist = []
        for i in range(points.shape[0]):
            meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))
            pointlist[i], facelist[i] = (pointlist[i].detach(), facelist[i].detach())
        meshlist = join_meshes_as_batch(meshlist)
        if is_vis == False:
            save("./vis", str(epoch), modelfile, pointlist, facelist)
            is_vis = True
        pts = sample_points_from_meshes(meshes=meshlist,num_samples=2562)
        LossCD2, _ = chamfer_distance(points,pts)
        # pred = pred - torch.mean(pred, axis=1, keepdim=True)
        # loss,_= chamfer_distance(points,pred[:,:,0:3])
        total_loss+=LossCD2.item()
        items+=1
            
    return total_loss/items  


if __name__ == '__main__':
    from config import get_config
    
    experiment, opt = get_config()

    import os
    import torch
    import torch.nn as nn
    from torch import optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import numpy as np
    import trimesh
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
    from pytorch3d.structures import join_meshes_as_batch

    from dataset.dataset import get_dataloader
    from model.model_atlas_tiger_4d import load_partial_pretrained
    from model.localfea import  Local_Fea_Injective_Mapping, Local_Fea_Deform4, Local_Fea_Deform
    from loss import sli, normal_loss, setup_seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        setup_seed()
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')

    
    if opt.train == 'AE':
        train_AE(experiment, opt)
