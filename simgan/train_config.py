''' 
	Config file for training the refiner / discriminator networks 

'''
import torch

''' ---- Everything below is pretty much hyperparameters to change ----- '''

data_root 		= '/s/mlsc/dmerric5/data/' 															# root of the data directory
checkpoint_root = '/raid/dmerric5/simgan/07_09/'													# root of the checkpoint directory (this must exist)

log = True																							# If log == True, a display of refined images will be shown during training
log_interval 	= 5																					# log_interval == # of steps before changing the display

cuda_num 		= 0																				# what nvidia-gpu to use


delta 			= .5																				# regularization parameter on the reconstruction loss (commonly called lambda)
train_steps		= 200000																			# number of steps to train for
buffer_size		= 12800																				# size of the buffer aka image pool, see paper for more details
momentum 		= .9																				# momentum value to use for optimization (stochastic gradient descent)
batch_size		= 32																				# which batch size to use
lr  			= 0.00001																			# which learning rate to use


#synthetic_path  = data_root + 'synthetic_data/cropped_eyes_/cropped_eyes/'							# path to the synthetic data set
synthetic_path  = data_root + 'unity_eyes/imgs_cropped/'
real_path		= data_root + 'MPIIGaze/cropped_eyes_grayscale_/cropped_eyes_grayscale/'			# path to the real data set

r_pretrain 		= 1000																				# number of steps to pretrain the refiner with
d_pretrain		= 200																				# number of steps to pretrain the discriminator with
k_r				= 2																					# number of updates to the refiner per step
k_d				= 1																					# number of updates to the discriminator per step

print_interval  = 10																				# print training info (loss, accuracy, etc..) every 'print_interval' steps
save_interval	= 25																				# save checkpoint info every 'save_per' steps

checkpoint_path = checkpoint_root + 'checkpoint_lr0001_bs512_deltaP5_unity/'						# path to save the checkpoint too	

#checkpoint_path = checkpoint_root + 'checkpoint_lr0001_bs512_deltaP75_unity/'

#save_path = root + 'checkpoint_lr001_bs512_deltaP5_kg5_unity/'

#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg5_unreal/' #new
#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg3_unreal/' #new
#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg3_unity/' #new
#save_path = root + 'checkpoint_lr0001_bs512_deltaP75_kg5_unity/' #new


#save_path = root + 'checkpoint_lr001_bs512_deltaP75_kg5_unity/' #newer
#save_path		= root + 'checkpoint_lr' + str(init_lr)[2:] + '_bs' + str(batch_size) + '_delta' +  str(delta)[1:] + '_kg' + str(k_g) + '_' + fake_data_type





''' ----- No need to change anything below ----- '''
D_path			= 'D_%d.pkl'																		# name to save discriminator checkpoint as, don't change this or you have to change load functions
R_path			= 'R_%d.pkl'																		# name to save refiner checkpoint as, don't change this or you have to change load functions
optimizer_path	= 'optimizer_status.pkl'															# name to save optimizer checkpoint (current step, etc), don't change this or you have to change load functions

cuda_use 		= torch.cuda.is_available()															# if gpu is available, true.. else false 
train = True																						# train = True because this is the train_config file!...
