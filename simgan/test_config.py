''' 
	config file for testing the SimGAN
'''

import torch

''' ---- Everything below is pretty much hyperparameters to change ----- '''


data_root 				= '/s/mlsc/dmerric5/'													# path to data root
checkpoint_root			= '/raid/dmerric5/simgan/checkpoints_06_27/'							# simple path to checkpoint root 
cuda_num 				= 0																		# Which NVIDIA GPU to use
batch_size 				= 256																	# Not terribly important for testing, but this is batch size.. Might impact processing speed, I guess?
synthetic_folder 		= data_root + 'unity_eyes/tmp_refiner_test/'							# Directory to load the synthetic images from
save_refined_path 		= data_root + 'unity_eyes/tmp_refiner_results/'							# Directory to save the refined images to
checkpoint_path 		= checkpoint_root + 'checkpoint_lr0001_bs512_deltaP75_kg3_unity/'		# path to load the checkpoint too, the format of the checkpoints should match how it is saved during training. If you want to use different format to load weights, you have to change load_weights in sub_simgan.py



#save_path = root + 'checkpoint_lr001_bs512_deltaP5_kg5_unity/'
#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg5_unreal/' #new
#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg3_unreal/' #new
#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg3_unity/' #new
#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg5_unity/' #new


#save_path = root + 'checkpoint_lr001_bs512_deltaP75_kg5_unity/' #newer
#save_path		= root + 'checkpoint_lr' + str(init_lr)[2:] + '_bs' + str(batch_size) + '_delta' +  str(delta)[1:] + '_kg' + str(k_g) + '_' + fake_data_type


''' ----- No need to change anything below ----- '''
train = False																					# train == false because we are testing !:)
D_path			= 'D_%d.pkl'																	# path to save the discriminator weights
R_path			= 'R_%d.pkl'																	# path to save the refiner weights
optimizer_path	= 'optimizer_status.pkl'														# path to save the optimizer info
cuda_use 		= torch.cuda.is_available()														# check if cuda is available

