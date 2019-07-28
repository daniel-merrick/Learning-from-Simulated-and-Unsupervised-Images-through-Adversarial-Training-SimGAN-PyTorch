'''
	config file for testing
'''
import torch


# This is a reference to all the potential training / testing files I have.
# One of these are assigned to 'testingFile' below

data_root = '/raid/dmerric5/'

UnityEyes_TrainFile 	= data_root + 'unity-eyes/UnityEyes_Train.txt'
UnityEyes_TestFile 		= data_root + 'unity-eyes/UnityEyes_Test.txt'

MPIIGaze_TrainFile		= data_root + 'MPIIGaze/MPIIGaze_TrainFile.txt'
MPIIGaze_TestFile		= data_root + 'MPIIGaze/MPIIGaze_TestFile.txt'

Unreal_Trainfile		= data_root + 'unrealChars/Unreal_TrainFile.txt'
Unreal_Testfile			= data_root + 'unrealChars/Unreal_TestFile.txt'

RefinedUnity_TrainFile	= ''
RefinedUnity_TestFile 	= ''

RefinedUnreal_TrainFile = ''
RefinedUnreal_TestFile	= ''


''' ----- Everything below is adjustable / can be change ----- '''


cuda_num = 0 																		 	# which NVIDIA-GPU to use
acceptable_error= .1 																	# parameter used in testing, % error that is acceptable * 100

checkpoint_root = '/raid/dmerric5/simgan/checkpoint_07_01/'

checkpoint_path = checkpoint_root + 'estimator_checkpoint_lr001_bs256_trainedOnUnity' 	# This is where the checkpoints get saved / loaded

# ignore below
#checkpoint_path= 'estimator_checkpoint_lr001_bs256_trainedOnUnityRefined'
#checkpoint_path= 'estimator_checkpoint_lr001_bs256_trainedOnUnreal'
#checkpoint_path= 'estimator_checkpoint_lr001_bs256_trainedOnUnrealRefined'


testingFile = UnityEyes_TrainFile														# Testing file that is actually used


''' ----- Everything below is adjustable / can be change ----- '''

E_path 			= 'E_%d.pkl'
optimizer_path 	= 'optimizer_status.pkl'
cuda_use = torch.cuda.is_available()
