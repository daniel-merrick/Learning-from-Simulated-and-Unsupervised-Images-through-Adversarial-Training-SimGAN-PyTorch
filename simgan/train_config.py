''' 
	Config file for training the refiner / discriminator networks 

'''
import torch

''' ---- Everything below is pretty much hyperparameters to change ----- '''

# path to the synthetic data set (a folder with all the synthetic images)
synthetic_path  = ''

# path to the real data set (a folder with all the real images)
real_path	= ''	

# If we want to log the model's generated outputs, set to true.
log 		= True																							# If log == True, a display of refined images will be shown during training

# Save the refiners output every log_interval steps (batches)
log_interval 	= 5																					# log_interval == # of steps before changing the display

# Which cuda device would you like to use?
cuda_num 	= 0																				# what nvidia-gpu to use

# regularization parameter on the reconstruction loss (commonly called lambda)
delta 		= .5	

# number of steps (batches) to train for
train_steps	= 200000

# size of the buffer aka image pool, see paper for more details
buffer_size	= 12800

# momentum value to use for optimization (stochastic gradient descent)
momentum 	= .9

# which batch size to use
batch_size	= 32

# which learning rate to use
lr  		= 0.00001			

# number of steps to pretrain the refiner with
r_pretrain	= 1000

# number of steps to pretrain the discriminator, using only MSE loss
d_pretrain	= 200	

# number of updates to the refiner per step
k_r		= 2	

# number of updates to the discriminator per step
k_d		= 1

# interval to print model status such as loss, measured by step (batch) number
print_interval  = 10	

# interval to save model checkpoints (weights), measured by step (batch) number
save_interval	= 25

# path to save checkpoints to
checkpoint_path = checkpoint_root + 'checkpoint_lr0001_bs512_deltaP5_unity/'



''' ----- No need to change anything below ----- '''
# name to save discriminator checkpoint as, don't change this or you have to change load functions
D_path		= 'D_%d.pkl'		

# name to save refiner checkpoint as, don't change this or you have to change load functions
R_path		= 'R_%d.pkl'

# name to save optimizer checkpoint (current step, etc), don't change this or you have to change load functions
optimizer_path	= 'optimizer_status.pkl'

# if gpu is available, true.. else false 
cuda_use 	= torch.cuda.is_available()		

# train = True because this is the train_config file!...
train 		= True		
