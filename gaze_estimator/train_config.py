'''
	config file for training the gaze estimator
'''

import torch

# This is a reference to all the potential training / testing files I have.
# One of these are assigned to 'trainingFile' below

data_root 				= '/s/mlsc/dmerric5/data/'												# root to data paths

UnityEyes_TrainFile 	= data_root + 'unity_eyes/UnityEyes_Train.txt'
UnityEyes_TestFile 		= data_root + 'unity_eyes/UnityEyes_Test.txt'

MPIIGaze_TrainFile		= data_root + 'MPIIGaze/MPIIGaze_TrainFile.txt'
MPIIGaze_TestFile		= data_root + 'MPIIGaze/MPIIGaze_TestFile.txt'

Unreal_Trainfile		= data_root + 'unrealChars/Unreal_TrainFile.txt'
Unreal_Testfile			= data_root + 'unrealChars/Unreal_TestFile.txt'

RefinedUnity_TrainFile	= data_root + 'unity_eyes/UnityEyes_Refined_Train.txt'
RefinedUnity_TestFile 	= data_root + 'unity_eyes/UnityEyes_Refined_Test.txt'

RefinedUnreal_TrainFile = ''
RefinedUnreal_TestFile	= ''


''' ----- Everything below is adjustable / can be change ----- '''



checkpoint_root 		= '/raid/dmerric5/simgan/07_11/'												# root to checkpoints
cuda_num 				= 3																				# which NVIDIA GPU to use


# Some hyperparameters
lr 						= 0.0005  																		# learning rate	
momentum				= 0.9	  																		# momentum for SGD
train_steps 			= 80000	  																		# numb training steps
batch_size 				= 512	 																		# batch size ...

acceptable_error		= .1																			# how much percent error is acceptable ? * 100
acceptable_degrees 		= 7
save_interval			= 5000																			# save weights every 'save_interval' steps
validate_interval 		= 100

#checkpoint_path 		= checkpoint_root + 'estimator_checkpoint_lr0001_bs512_trainedOnUnity'			# This is where the checkpoints get saved
#checkpoint_path		= checkpoint_root + 'estimator_checkpoint_lr0001_bs512_trainedOnRefinedUnity/'


#old #checkpoint_root 		= '/raid/dmerric5/simgan/07_09/'											# root to checkpoints



checkpoint_path 		= checkpoint_root + 'estimator_checkpoint_lr0005_bs512_trainedOnUnity'
#checkpoint_path 		= checkpoint_root + 'estimator_checkpoint_lr0005_bs512_trainedOnRefinedUnity'

# This is the config file that actually gets used
trainingFile 			= UnityEyes_TrainFile
# trainingFile			= RefinedUnity_TrainFile


# testingFile			= None
# testingFile			= UnityEyes_TestFile
# testingFile 			= RefinedUnity_TestFile
testingFile 			= MPIIGaze_TestFile


#checkpoint_path= 'estimator_checkpoint_lr001_bs256_trainedOnUnityRefined'
#checkpoint_path= 'estimator_checkpoint_lr001_bs256_trainedOnUnreal'
#checkpoint_path= 'estimator_checkpoint_lr001_bs256_trainedOnUnrealRefined'


''' ----- Do not change anything below here, unless you know what happens ----- '''
estimator_path			= 'E_%d.pkl'												# path to save estimator weights
optimizer_path 			= 'optimizer_status.pkl'									# path to save optimizer info
cuda_use 				= torch.cuda.is_available()									# check if cuda available
train 	 				= True														# yes we are training
