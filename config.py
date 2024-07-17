import argparse

def get_config():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='AE', type=str, help = "Train 'AE' or 'SVR'")
    parser.add_argument("--generate", default='AE', type=str, help = "Generate for 'AE' or 'SVR'")
    parser.add_argument("--noise", default=False, help = "Noise for the pointcloud")
    parser.add_argument("--latent_len", default=1024, type=int, help = "Length of the latent embedding")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help = "Initial learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help = "Batch size used for training")
    parser.add_argument("--num_workers", default=16, type=int, help = "Number of workers used for data loading")
    parser.add_argument("--weight_decay", default=0.98, type=float, help = "Weight decay used during training")
    parser.add_argument("--num_epochs", default=150, type=int, help = "Number of epochs to train")
    parser.add_argument("--is_small", default=False, help = "Set to True if want to work with a small dataset for debug/demo purposes")
    parser.add_argument("--model_folder", default='./train_models/', type=str, help = "PATH to where the models are saved during training ")
    parser.add_argument("--code_folder", default='./train_codes/', type=str, help = "PATH to where the codes are saved during training ")
    parser.add_argument("--points_path", default = './data/NMF_points/', type=str, help = "PATH to the directory containing Shapenet points dataset")
    parser.add_argument("--img_path", default = './data/ShapeNetRendering/', type=str, help = "PATH to the directory containing Shapenet points dataset")
    parser.add_argument("--model_folder_SVR", default = './train_models_svr/', type=str, help = "PATH to where the models are saved during training SVR")
    parser.add_argument("--generate_ae", default = './sub sphere/', type=str, help = "PATH to where meshes for AE are stored")
    parser.add_argument("--generate_svr", default = './demo_out/', type=str, help = "PATH to where meshes for SVR are stored")
    parser.add_argument("--pretrained_svr_weights", default = './train_models_svr/bestnoise', type=str, help = "PATH to pretrained SVR weights")
    parser.add_argument("--pretrained_ae1_weights", default = './train_models/refine', type=str, help = "PATH to pretrained AE weights")
    parser.add_argument("--comet_API", default = '2333', type=str, help = "your API for comet_ml workspace")
    parser.add_argument("--comet_workspace", default = None, type=str, help = "your comet_ml workspace name")
    parser.add_argument("--comet_project_name", default = "NeuralMeshFlow", type=str, help = "Name of this project in comet_ml")
    parser.add_argument("--mapping_layers", default = 50, type=int, help = "number of injective mapping layers")
    
    opt = parser.parse_args()
    
    if opt.comet_API is not None :
        from comet_ml import Experiment

        experiment = None   
        # experiment = Experiment(api_key='CmrZRHjfFmGmIcgCoNRrzfc7T',
        #                 project_name="4D_GAN",workspace='jiang25262')
        
    else:
        experiment = None
        
    return experiment, opt
    
