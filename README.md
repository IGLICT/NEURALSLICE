# NEURALSLICE
The code of "NEURALSLICE: Neural 3D Triangle Mesh Reconstruction via Slicing 4D Tetrahedral Meshes (ICML 2023)"

We release demo code about fitting a deformable animal within single 4D Tetrahedral Mesh. 

Using *python fit_deform.py* to fit a animal.

We thank Neural Mesh Flow and Conv Occ Net's code framework and processed point clould dataset. Please refer to https://github.com/KunalMGupta/NeuralMeshFlow and https://github.com/autonomousvision/convolutional_occupancy_networks.

Using *python train_coarse.py* to train a coarse model and then using *python train_subdifRefine.py* to train a fine model.

