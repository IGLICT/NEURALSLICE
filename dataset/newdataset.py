import torch
from torch.utils import data
import torch.nn as nn
import json
import random
import os
import imageio
import numpy as np

def get_dataloader(type, opt, split = 'train', is_small=False, img_num=0):
    '''
    Helper function let's you choose a dataloader
    type: Choose from 'point' or 'image' for shape completion or SVR tasks.
    opt: options for choosing dataloader
    split: Choose from 'train', 'test' and 'val'
    points_path: PATH to the directory containing Shapenet points dataset
    img_path: PATH to the directory containing ShapeNet renderings from Choy et.al (3dr2n2)
    is_small: Set to True if wish to work with a small dataset of size 100. For demo/debug purpose
    img_num: Choose an image number 00-23 during generation
    '''
    
    # Parameters
    params = {'batch_size': opt.batch_size,
              'shuffle': True,
              'num_workers': opt.num_workers,
              'drop_last' : True}
    
    if split == 'test' or split =='val':
        params['shuffle'] = False
        
    training_set = Dataset(split, opt,  encoder_type = type, is_small=opt.is_small, img_num=img_num)
    training_generator = data.DataLoader(training_set, **params)

    print("Dataloader for {}s with Batch Size : {} and {} workers created for {}ing.".format(type, params['batch_size'], params['num_workers'], split))
    return training_generator


class Dataset(data.Dataset):
    '''
    Main dataset class used for testing/training. Can be used for both SVR and shape completion/AE tasks. 
    '''
    
    def __init__(self, split_type, opt, encoder_type='points', is_small=False, img_num=0):
        '''
        Initialization function.
        split_type: 'train', 'valid', 'test' used to specify the partion of dataset to be used
        opt: options for choosing dataloader
        encoder_type: 'points' or 'image' used to specify the type of input data to be used. Will fetch appropriate image data if 'image' is used.
        is_small: Set to True if wish to work with a small dataset of size 100. For demo/debug purpose
        img_num: Choose an image number 00-23 during generation
        '''
        
        # Load the file containing model splits. Splits are made based on 3dr2n2 by Choy et.al. 
        
        with open('dataset/newsplit.json', 'r') as outfile:  
            split = json.load(outfile)

        self.models = split[split_type]
        self.split = split_type

        random.shuffle(self.models)   # Randomly shuffle the models

        if is_small:            # Work with a very small dataset (used for debug/demo)
            print(" !!! Using small dataset of size 100 !!!  ")
            self.models = self.models[:100]
        
        self.encoder_type = encoder_type
        
        print("total models for {}ing : ".format(self.split), len(self.models))
        
        
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self,index):
        
        modelfile = self.models[index]

        if self.encoder_type == 'point':
            # Fetch dataset for AE 
            
            try:
                X = np.load('{}'.format(modelfile), allow_pickle=True)  # Read point cloud 
                X /= np.max(np.linalg.norm(X, axis=1)) 
                
            except:
                # Incase fail to load the file, simply return the first file of dataset to prevent crashing.
                
                print("Error loading:", modelfile)
                modelfile = self.models[0]
                X = np.load('{}'.format(modelfile), allow_pickle=True)
                X /= np.max(np.linalg.norm(X, axis=1)) 
                X = torch.from_numpy(X)
                X = X.float()

            return X, self.split, modelfile 


