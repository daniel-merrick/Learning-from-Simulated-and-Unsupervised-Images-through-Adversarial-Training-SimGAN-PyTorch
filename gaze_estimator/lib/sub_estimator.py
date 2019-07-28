
'''
	This file hold a class 'SubEstimator' that has some basic
	functionality we can inherit from when building the
	TrainEstimator and TestEstimator classes.
	
	SubEstimator has functionality to load weights,
	intialize data loaders, and various accuracy metrics.
'''

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from torchvision import transforms

from data_loader import DataLoader
from gaze_estimator_network import GazeEstimatorNetwork

class SubEstimator:
	def __init__(self, cfg):
		# initializing variables to None, used later
		self.estimator = None
		self.estimator_optimizer = None
		self.mse_loss = None
		self.data_loader = None
		self.current_step = None
		self.cfg = cfg
	
		self.pi = math.pi

	# used internally
	# checks for saved weights in the checkpoint path
	# return True if weights are loaded
	# return False if no weights are found
	def load_weights(self):
		
		print("Checking for Saved Weights")

		# If checkpoint path doesn't exist, create it
		if not os.path.isdir(self.cfg.checkpoint_path):
			os.mkdir(self.cfg.checkpoint_path)
		
		# get list of checkpoints from checkpoint path
		checkpoints = os.listdir(self.cfg.checkpoint_path)
		
		# Only load weights that start with 'E_'
		estimator_checkpoints = [ckpt for ckpt in checkpoints if 'E_' == ckpt[:2]]
		
		# Sort the weight files
		estimator_checkpoints.sort(key=lambda x: int(x[2:-4]), reverse=True)

		# return False if there are no previous checkpoints
		if len(estimator_checkpoints) == 0 or not os.path.isfile(
						os.path.join(self.cfg.checkpoint_path, self.cfg.optimizer_path)):
			print("No Previous Weights Found. Building and Initializing new Model")
			self.current_step = 0
			return False
		
		print("Found Saved Weights, Loading...")		

		# load optimizer information / estimator weigths
		optimizer_status = torch.load(os.path.join(self.cfg.checkpoint_path, self.cfg.optimizer_path))
		
		self.estimator_optimizer.load_state_dict(optimizer_status['optE'])
		self.current_step = optimizer_status['step']

		self.estimator.load_state_dict(torch.load(os.path.join(self.cfg.checkpoint_path, estimator_checkpoints[0])))
		
		return True

	# builds and initialized network stuff such as
	# optimizer / loss / weights, etc
	def build_network(self):
		print("Building the Estimator Network")
		
		# init the network and load weights
		self.estimator = GazeEstimatorNetwork()


		if self.cfg.cuda_use:
			self.estimator.cuda(self.cfg.cuda_num)

		self.estimator_optimizer = torch.optim.SGD(self.estimator.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)
		self.mse_loss = nn.MSELoss()
		

		self.load_weights()

		print('Done building')

	# iterator for the data loader..
	# not really important
	def loop_iter(self, dataloader):
		while True:
			for data in iter(dataloader):
				yield data
			
			if not self.cfg.train:
				print('Finished one epoch, done testing')
				self.testing_done = True

	# init data loader stuff
	def get_data_loaders(self):
		self.data_loader1 = DataLoader(self.cfg.trainingFile)
		self.data_loader = Data.DataLoader(self.data_loader1, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=3)
		self.data_loader_iter = self.loop_iter(self.data_loader)	
		
		if not self.cfg.testingFile == None:
			self.data_loader2 = DataLoader(self.cfg.testingFile, mpii=True)
			self.val_data_loader = Data.DataLoader(self.data_loader2, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=1)
			self.val_data_loader_iter = self.loop_iter(self.val_data_loader)
			
			
	# compute some simple accuracy metrics
	def get_accuracy(self, predicted_gaze, gt_gaze):
		errors_ = torch.abs(torch.div(predicted_gaze - gt_gaze, gt_gaze)).cpu()
		where_correct = torch.where(errors_ < torch.tensor(self.cfg.acceptable_error), torch.tensor(True), torch.tensor(False))
		
		num_xyz_correct = torch.sum(where_correct, dim=0, dtype=torch.float32)
		
		vec_correct = torch.sum(where_correct, dim=1, dtype=torch.float32)
		num_3_correct = torch.sum(torch.where(vec_correct == 3, torch.tensor(True), torch.tensor(False)), dtype=torch.float32)
		num_2_correct = torch.sum(torch.where(vec_correct == 2, torch.tensor(True), torch.tensor(False)), dtype=torch.float32)
		num_1_correct = torch.sum(torch.where(vec_correct == 1, torch.tensor(True), torch.tensor(False)), dtype=torch.float32)
		
		return num_xyz_correct, num_3_correct, num_2_correct, num_1_correct
	
	# compute the angle between two batches of vectors in cartesian space
	def get_angle_between(self, predicted_gaze, groundtruth_gaze):
		return torch.acos(F.cosine_similarity(predicted_gaze, groundtruth_gaze, dim=1)) * 180.0 / self.pi

	def how_many_within_degrees(self, angles_in_degrees):
		return torch.sum(torch.where(angles_in_degrees <= torch.tensor(self.cfg.acceptable_degrees, dtype=torch.float32), torch.tensor(True), torch.tensor(False)), dtype=torch.float32)
