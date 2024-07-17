from re import L


def train_AE(experiment, opt):
    # from src import config, data
    print("*********** INITIALIZATION  ************")

    torch.cuda.set_device(0)  
    encoder_type = 'point'   

    lrat = 1e-4
    training_generator = get_dataloader(encoder_type, opt)   
    validation_generator = get_dataloader(encoder_type, opt, split = 'val')   
    

    model = LocalSubdivRefine().to(device)
    print("*********** SETUP TRAINING  ************")
    optimizer = optim.Adam(model.refine.parameters(), lr = lrat)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    load_partial_pretrained(model, "./train_models/subdiv tiger")
    # load_partial_pretrained(model.corase_mesh, "./train_models/coarse tiger")
    # load_partial_pretrained(model, "./train_models/subdiv_epoch_50")
    
    os.makedirs(opt.model_folder,exist_ok=True)
    print("Models are being saved at :", opt.model_folder)
    if experiment is None: 
        print("Comet_ml logging is disabled, printing on terminal instead")
    print("*********** BEGIN TRAINING  ************")
    (step, epoch, stage) = (0, 0, 1)
    for epoch in range(0, opt.num_epochs):
        # print(epoch)
        lloss, lloss2, sstep, llap, ledge = (0,0,0,0,0)
        if epoch==20:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/1.5)
        if epoch==35:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/2)
        if epoch==70:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/5)
        if epoch==100:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/10)
        if epoch==125:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/100)
        if epoch==145:
            optimizer = optim.Adam(list(model.parameters()), lr = lrat/1000)
        for data in training_generator:  
            optimizer.zero_grad()
            points = data[0].to(device).float()  
            L_points = data[1].to(device).float()

            (L1,L2,L3,L4) = (0,0,0,0)
            noise = 0.005 * torch.randn(points.shape[0],points.shape[1],3).to(device)
            L_cd1, L_lap1, L_cd2, L_lap2,_,_,L_n= model(points, L_points) 
            loss = L_cd2 + (L_lap2)*0.1
            loss = loss.requires_grad_()
            loss.backward()
            ledge += L_n.item()
            llap += L_lap2.item()
            lloss2 += L_cd2.item()
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
             
            torch.save(model.state_dict(),opt.model_folder+'subdiv_epoch_{}'.format(epoch))
            
        print("CD Loss:",lloss2/sstep,
              " Lap Loss:", llap/sstep, 
              " Norm Loss:", ledge/sstep,
              " Learning Rate:",optimizer.state_dict()['param_groups'][0]['lr'])

    print("Saving models after #Epoch :", epoch+1) 
    torch.save(model.state_dict(),opt.model_folder+'subdiv_epoch_{}'.format(epoch))
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
    with torch.no_grad():
        total_loss = 0
        items = 0.001
        is_vis = False
        for input, N, cat, modelfile in validation_generator: 

            points = input.to(device).float()
            _, _, L_cd2, _, pointlist, facelist,_ = model(points)
            meshlist = []
            for i in range(points.shape[0]):
                meshlist.append(Meshes(verts=pointlist[i],faces=facelist[i]))
                pointlist[i], facelist[i] = (pointlist[i].detach(), facelist[i].detach())

            if is_vis == False:
                save("./vis", str(epoch), modelfile, pointlist, facelist)
                is_vis = True
            # pred = pred - torch.mean(pred, axis=1, keepdim=True)
            # loss,_= chamfer_distance(points,pred[:,:,0:3])
            total_loss+=L_cd2.item()
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
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.structures import join_meshes_as_batch

    import trimesh
    from dataset.dataset import get_dataloader
    from model.model_atlas_tiger_4d import load_partial_pretrained
    from model.localfea import SubdivRefine, LocalSubdivRefine
    from loss import sli, normal_loss, setup_seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    # save_code(opt)
    setup_seed()
    if opt.train == 'AE':
        train_AE(experiment, opt)
