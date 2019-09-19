
'''
	This file hold a class 'SubSimGAN' that has some basic
	functionality we can inherit from when building the
	TrainSimGAN or TestSimGAN classes. Most of it isn't 
	terribly important thats why I hide it in this sub class.

	Things such as accuracy metrics, data loaders, 
	weight loading, etc
'''

import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.autograd as autograd
from torchvision import transforms

import torch.nn.functional as F
from simgan_network import Refiner, Discriminator
from data_loader import DataLoader

class SubSimGAN:
	''' 
		SubSimGAN : a class that can be inherited by TestSimGAN or TrainSimGAN.

			Notes*
				1. This class is meant to hide anything that isn't directly used to 
				train or test the simgan network.
				2. Input is the training / testing config file
				3. No output, methods and variables are inherited
	'''
	def __init__(self, cfg):
		
		
		self.cfg = cfg
		
		# initializing variables to None, used later
		self.R = None
		self.D = None
		
		self.refiner = None
		self.discriminator = None
		self.refiner_optimizer = None
		self.discriminator_optimizer = None
		
		self.feature_loss = None #Usually L1 norm or content loss
		self.local_adversarial_loss = None #CrossEntropyLoss
		self.data_loader = None
		self.current_step = None

		self.synthetic_data_loader = None
		self.real_data_loader = None
		self.synthetic_data_iter = None
		self.real_data_iter = None
		self.weights_loaded = None
		self.current_step = 0
	
		if not self.cfg.train:
			self.testing_done = False


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
		
		# Only load weights that start with 'R_' or 'D_'
		refiner_checkpoints = [ckpt for ckpt in checkpoints if 'R_' == ckpt[:2]]
		discriminator_checkpoints = [ckpt for ckpt in checkpoints if 'D_' == ckpt[:2]]

		# Sort the weight files
		refiner_checkpoints.sort(key=lambda x: int(x[2:-4]), reverse=True)
		discriminator_checkpoints.sort(key=lambda x: int(x[2:-4]), reverse=True)

		# return False if there are no previous checkpoints
		if len(refiner_checkpoints) == 0 or len(discriminator_checkpoints) == 0 or not os.path.isfile(
						os.path.join(self.cfg.checkpoint_path, self.cfg.optimizer_path)):
			print("No Previous Weights Found. Building and Initializing new Model")
			self.current_step = 0
			return False
		
		print("Found Saved Weights, Loading...")		

		if self.cfg.train:
			# load optimizer information / estimator weigths
			optimizer_status = torch.load(os.path.join(self.cfg.checkpoint_path, self.cfg.optimizer_path))
			self.refiner_optimizer.load_state_dict(optimizer_status['optR'])
			self.discriminator_optimizer.load_state_dict(optimizer_status['optD'])
			self.current_step = optimizer_status['step']
			self.D.load_state_dict(torch.load(os.path.join(self.cfg.checkpoint_path, discriminator_checkpoints[0])))
		
		self.R.load_state_dict(torch.load(os.path.join(self.cfg.checkpoint_path, refiner_checkpoints[0])))

		
		return True

	def build_network(self):
		print("Building SimGAN Network")
		
		# init the network and load weights
		self.R = Refiner()
		self.D = Discriminator()

		
		# If we are using cuda, place the models on the GPU
		if self.cfg.cuda_use:
			self.R.cuda(self.cfg.cuda_num)
			self.D.cuda(self.cfg.cuda_num)

		if self.cfg.train:
			# Set optimizers
			self.refiner_optimizer = torch.optim.SGD(self.R.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)
			self.discriminator_optimizer = torch.optim.SGD(self.D.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)
		

		self.weights_loaded = self.load_weights()

		# Set loss functions
		self.feature_loss = nn.L1Loss()
		self.local_adversarial_loss = nn.CrossEntropyLoss()
	
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
		#synthetic_folder = torchvision.datasets.ImageFolder(root=self.cfg.synthetic_path, transform=self.transform)
		#real_folder = torchvision.datasets.ImageFolder(root=self.cfg.real_path, transform=self.transform)
		
		synthetic_data = DataLoader(self.cfg.synthetic_path)
		self.synthetic_data_loader = Data.DataLoader(synthetic_data, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=3)
		print('num synthetic_data : {}'.format(synthetic_data.data_len))
		
		if self.cfg.train:
			
			self.synthetic_data_iter = self.loop_iter(self.synthetic_data_loader)
			
			real_data 				 = DataLoader(self.cfg.real_path)
			self.real_data_loader 	 = Data.DataLoader(real_data, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=3)
			self.real_data_iter      = self.loop_iter(self.real_data_loader)
		

	''' simple function to return accuracy '''
	def get_accuracy(self, output, type='real'):
		assert type in ['real', 'refine']
		score = F.softmax(output, dim=1)
		class1 = score.cpu().data.numpy()[:, 0]
		class2 = score.cpu().data.numpy()[:, 1]
	
		if type == 'real':
			return (class1 < class2).mean()
		else:
			return (class1 > class2).mean()

	''' function to print some info on 
		the refiner during training '''
	def print_refiner_info(self, step, pretrain=False):

		if not pretrain:
			print('Step: {}'.format(step))
			print('Refiner... Reconstruction Loss: %.4f, Adversarial Loss: %.4f\n' % (self.recon_loss.item(), self.adv_loss.item()))
		else:
			print('Step: {} / {}'.format(step, self.cfg.r_pretrain))
			print('Refiner... Reconstruction Loss: %.4f\n' % (self.recon_loss.item()))
	
	''' function to print some info on
		the discriminator during training '''
	def print_discriminator_info(self, step, pretrain=False):
		if not pretrain:
			print('Step: {}'.format(step))
		else:
			print('step: {} / {}'.format(step, self.cfg.d_pretrain))
		
		print('Discriminator... real loss: %.4f, refined loss: %.4f accuracy_real: %.2f, accuracy_refined: %.2f\n' % (self.loss_real.item(), self.loss_refined.item(), self.accuracy_real, self.accuracy_refined))
		
