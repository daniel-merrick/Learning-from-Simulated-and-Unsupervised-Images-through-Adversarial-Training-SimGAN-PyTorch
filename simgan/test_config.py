''' 
	config file for testing the SimGAN
'''

import torch

''' ---- Everything below is pretty much hyperparameters to change ----- '''


# Which GPU to use
cuda_num 		= 0

# Not terribly important for testing
batch_size 		= 256

# Directory to load the synthetic images from (path to the folder)
synthetic_folder 	= ''

# Directory to save the refined images to (path to the folder)
save_refined_path 	= ''

# path to load the checkpoint too, the format of the checkpoints should match how it is saved during training. 
#If you want to use different format to load weights, you have to change load_weights in sub_simgan.py
checkpoint_path 	= ''


''' ----- No need to change anything below ----- '''

# prefix of saved discriminator weights
D_path			= 'D_%d.pkl'

# prefix of saved refiner weights
R_path			= 'R_%d.pkl'

# filename of optimizer information, this file 
# is located in the folder pointed to by checkpoint_path
optimizer_path	= 'optimizer_status.pkl'

# check if cuda is available
cuda_use 		= torch.cuda.is_available()

# train = false because we are testing !:)
train = False
