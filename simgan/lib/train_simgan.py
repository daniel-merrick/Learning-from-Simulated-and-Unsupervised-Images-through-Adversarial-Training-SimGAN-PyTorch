'''
	This file includes the class to train SimGAN
	
	TrainSimGAN inherits the functions from 
	SubSimGAN. 

	SubSimGAN has the functions for 
	weight loading, initializing 
	data loaders, and accuracy metrics
'''

import os

import torch
import torch.nn as nn
import cv2

from sub_simgan import SubSimGAN
from show_images import *
from image_pool import ImagePool
from data_loader import DataLoader

class TrainSimGAN(SubSimGAN):
	def __init__(self, cfg):
		SubSimGAN.__init__(self, cfg)
		
		self.recon_loss = None
		self.refiner_loss = None
		self.adv_loss = None
		self.loss_real = None
		self.loss_refined = None

		self.cfg = cfg

		self.c = 0.01
			
	# Used to update the generator
	def update_refiner(self, pretrain=False):
		
		''' Get batch of synthetic images '''
		synthetic_images, _ = next(self.synthetic_data_iter)
		synthetic_images = synthetic_images.cuda(self.cfg.cuda_num)
		
		''' Refine synthetic images '''
		refined_images = self.R(synthetic_images)
		
		''' Get feature loss between 
			synthetic and refined images '''
		self.recon_loss = torch.mul(self.feature_loss(refined_images, synthetic_images), self.cfg.delta)
			
		if not pretrain:
			''' Get Discriminators predictions
				on the refined images '''
			refined_predictions = self.D(refined_images).view(-1, 2)

			''' Get a batch of real image labels '''
			real_labels = torch.ones(refined_predictions.size(0)).type(torch.LongTensor).cuda(self.cfg.cuda_num)
			
			''' Get the adversarial loss for generator
				... Trying to fool discriminator ... '''
			self.adv_loss = self.local_adversarial_loss(refined_predictions, real_labels)

			''' Combine losses '''
			self.refiner_loss = self.recon_loss + self.adv_loss
		else:
			self.refiner_loss = self.recon_loss
		
		self.refiner_optimizer.zero_grad()
		self.refiner_loss.backward()
		self.refiner_optimizer.step()

	
	# Used to update the discriminator
	def update_discriminator(self, pretrain=False):
		''' get batch of real images '''
		real_images, _ = next(self.real_data_iter)
		real_images = real_images.cuda(self.cfg.cuda_num)
		real_predictions = self.D(real_images)
		real_labels = torch.ones(real_predictions.size(0)).type(torch.LongTensor).cuda(self.cfg.cuda_num)

		self.accuracy_real = self.get_accuracy(real_predictions, 'real')
		self.loss_real = self.local_adversarial_loss(real_predictions, real_labels)

		''' get batch of synthetic images '''
		synthetic_images, _ = next(self.synthetic_data_iter)
		synthetic_images = synthetic_images.cuda(self.cfg.cuda_num)
		
		''' get batch of refined images and labels '''
		refined_images = self.image_pool.query(self.R(synthetic_images).detach().cpu())
		refined_images = refined_images.cuda(self.cfg.cuda_num)

		refined_predictions = self.D(refined_images)
		refined_labels = torch.zeros(refined_predictions.size(0)).type(torch.LongTensor).cuda(self.cfg.cuda_num)
		
		self.accuracy_refined = self.get_accuracy(real_predictions, 'refine')
		self.loss_refined = self.local_adversarial_loss(refined_predictions, refined_labels)
	

		# gradient_penalty = self.calc_gradient_penalty(real_images, refined_images)
	
		''' this is the loss used to backprop '''
		discriminator_loss = (self.loss_refined + self.loss_real) / 2 #+ gradient_penalty
	
		''' Update params '''
		self.discriminator_optimizer.zero_grad()
		discriminator_loss.backward()
		self.discriminator_optimizer.step()
		

	''' Used to pretrain the refiner if no previous
		weights are found '''
	def pretrain_refiner(self):
		# This method pretrains the generator if called
		print('Pre-training the refiner network {} times'.format(self.cfg.r_pretrain))

		''' Set the refiner gradients parameters to True 
			Set the discriminators gradients params to False'''
		self.R.train_mode(True)
		self.D.train_mode(False)
		
		''' Begin pre-training the refiner '''
		for step in range(self.cfg.r_pretrain):
			self.update_refiner(pretrain=True)
			
			if step % self.cfg.print_interval == 0 or (step == self.cfg.r_pretrain-1):
				self.print_refiner_info(step, pretrain=True)
				
		print('Done pre-training the refiner')
	
	''' Used to pretrain the discriminator if no previous
		weights are found '''
	def pretrain_discriminator(self):
		print('Pre-training the discriminator network {} times'.format(self.cfg.d_pretrain))

		''' Set the Discriminators gradient parameters to True
			Set the Refiners gradient parameters to False '''
		self.D.train_mode(True)
		self.R.train_mode(False)
		self.R.eval()

		''' Begin pretraining the discriminator '''
		for step in range(self.cfg.d_pretrain):

			''' update discriminator and return some important info for printing '''
			self.update_discriminator(pretrain=True)

			if step % self.cfg.print_interval == 0 or (step == self.cfg.d_pretrain - 1):
				self.print_discriminator_info(step, pretrain=True)

		print('Done pre-training the discriminator')
	

	''' main function called externally
		used to train the entire network '''
	def train(self):
		self.build_network()	
		self.get_data_loaders()

		''' If no saved weights are found,
			pretrain the refiner / discriminator '''
		if not self.weights_loaded:
			self.pretrain_refiner()
			self.pretrain_discriminator()
		
		''' Initialize the image buffer '''
		self.image_pool = ImagePool(self.cfg.buffer_size)
		
		''' Check if step is valid '''
		assert self.current_step < self.cfg.train_steps, 'Target step is smaller than current step'


		for step in range((self.current_step + 1), self.cfg.train_steps):
			
			self.current_step = step
			
			''' Train Refiner ''' 
			self.D.eval()
			self.D.train_mode(False)
			
			self.R.train()
			self.R.train_mode(True)
			
			for idx in range(self.cfg.k_r):
				''' update refiner and return some important info for printing '''
				self.update_refiner(pretrain=False)

			

			''' Train Discriminator '''
			self.R.eval()
			self.R.train_mode(False)

			self.D.train()
			self.D.train_mode(True)

			for idx in range(self.cfg.k_d):
				''' update discriminator and return some important info for printing '''
				self.update_discriminator(pretrain=False)
			
			"""
			''' Train Discriminator '''
			self.R.eval()
			self.R.train_mode(False)

			self.D.train()
			self.D.train_mode(True)

			for idx in range(self.cfg.k_d):
				''' update discriminator and return some important info for printing '''
				self.update_discriminator(pretrain=False)
				
				#''' Weight Clipping -- Include this for more stable training
				#	something about lipshitz variables... aka WGAN '''
				#for p in self.D.parameters():
				#	p.data.clamp_(-self.c, self.c)


		
			''' Train Refiner '''
			self.D.eval()
			self.D.train_mode(False)
			
			self.R.train()
			self.R.train_mode(True)

			for idx in range(self.cfg.k_r):
				''' update refiner and return some important info for printing '''
				self.update_refiner(pretrain=False)
			"""

			if step % self.cfg.print_interval == 0 and step > 0:
				self.print_refiner_info(step, pretrain=False)
				self.print_discriminator_info(step, pretrain=False)
			
			
			if self.cfg.log == True and (step % self.cfg.log_interval == 0 or step == 0):
				synthetic_images, _ = next(self.synthetic_data_iter)
				synthetic_images = synthetic_images.cuda(self.cfg.cuda_num)
				refined_images = self.R(synthetic_images)

				figure = np.stack([
					var_to_np(synthetic_images[:32]),
					var_to_np(refined_images[:32]),
					], axis=1)

				figure = figure.transpose((0, 1, 3, 4, 2))
				figure = figure.reshape((4, 8) + figure.shape[1:])
				figure = stack_images(figure)
				figure = np.squeeze(figure, axis=2)
				figure = np.clip(figure*255, 0, 255).astype('uint8')

				cv2.imwrite(self.cfg.checkpoint_path + 'images/' + 'eyes_' + str(step) + '_.jpg', figure)

			if step % self.cfg.save_interval == 0:
				print('Saving checkpoints, Step : {}'.format(step))	
				torch.save(self.R.state_dict(), os.path.join(self.cfg.checkpoint_path, self.cfg.R_path % step))
				torch.save(self.D.state_dict(), os.path.join(self.cfg.checkpoint_path, self.cfg.D_path % step))

				state = {
					'step': step,
					'optD' : self.discriminator_optimizer.state_dict(),
					'optR' : self.refiner_optimizer.state_dict()
				}
				
				torch.save(state, os.path.join(self.cfg.checkpoint_path, self.cfg.optimizer_path))

