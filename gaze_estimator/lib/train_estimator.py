'''
	This file contains a class to train the GazeEstimatorNetwork
	
	TrainEstimator inherits the SubEstimator class. This is 
	meant to be fairly high-level.
	
	SubEstimator class has utility functions for things
	such as weight loading, data loader initializers, and
	accuracy calculations.
'''

import os

import torch
import torch.nn as nn
from torchvision import transforms

from data_loader import DataLoader
from gaze_estimator_network import GazeEstimatorNetwork
from sub_estimator import SubEstimator

import numpy as np
class TrainEstimator(SubEstimator):
	def __init__(self, cfg):
		SubEstimator.__init__(self, cfg)	
		self.cfg = cfg

	# Occasionally check out how we 
	# are doing on the test set
	def validate(self):

		print('-----------------------------------------------------------------')
		print('Evaluating estimator on 10 random batches of testing data')
		angles_between = []
		for test_batch in range(10):
			# Get batch of data
			images, gt_gazes, imgPaths = next(self.val_data_loader_iter)
			images, gt_gazes = images.cuda(self.cfg.cuda_num), gt_gazes.cuda(self.cfg.cuda_num)
			
			# Get predicted gaze vectors
			predicted_gaze = self.estimator(images)
				
			# Compute Mean Squared Error (MSE) loss between
			# predicted_gaze, and gt_gaze
			loss = self.mse_loss(predicted_gaze, gt_gazes)
					
			# Get some basic accuracy metrics
			xyz_correct, num_3_correct, num_2_correct, num_1_correct = self.get_accuracy(predicted_gaze, gt_gazes)
			angle_between = self.get_angle_between(predicted_gaze, gt_gazes).cpu()
			num_within_degrees = self.how_many_within_degrees(angle_between)
			mean_angle_between = torch.mean(angle_between).item()
			angles_between.append(mean_angle_between)

			print('Test Batch: %d, loss: %.4f\n percent X correct: %.2f, percent Y correct: %.2f, percent Z correct %.2f\n percent 3 correct: %.2f, percent 2 correct: %.2f, percent 1 correct %.2f\n batch_size: %d, lr: %.5f' % (test_batch, loss.item(), (xyz_correct[0] / self.cfg.batch_size) * 100.0, (xyz_correct[1] / self.cfg.batch_size) * 100.0, (xyz_correct[2] / self.cfg.batch_size) * 100.0, (num_3_correct / self.cfg.batch_size) * 100.0, (num_2_correct / self.cfg.batch_size) * 100.0,(num_1_correct / self.cfg.batch_size) * 100.0, self.cfg.batch_size, self.cfg.lr))
			print('percent within degree difference : {}, mean angle between: {}\n'.format((num_within_degrees / self.cfg.batch_size) * 100.0, mean_angle_between))
		
		angles_between = np.array(angles_between)
		mean_angles_between = np.mean(angles_between)
		print('Average angles_between : {}'.format(mean_angles_between))
		print('Finished testing. Returning to training.')
		print('-----------------------------------------------------------------')
	
	def train(self):
		# Normal init stuff
		self.build_network()
		self.get_data_loaders()
		
		self.estimator.train_mode()

		print('Begin Training .... :)')
		assert self.current_step < self.cfg.train_steps, 'Target step is smaller than current step! ... :('

		for step in range(self.current_step + 1, self.cfg.train_steps):
			# Get batch of data
			images, gt_gazes, imgPaths = next(self.data_loader_iter)
			images, gt_gazes = images.cuda(self.cfg.cuda_num), gt_gazes.cuda(self.cfg.cuda_num)

			# Get predicted gaze vectors
			predicted_gaze = self.estimator(images)
		
			# Compute Mean Squared Error (MSE) loss between
			# predicted_gaze, and gt_gaze
			loss = self.mse_loss(predicted_gaze, gt_gazes)
			
			# Get some basic accuracy metrics
			xyz_correct, num_3_correct, num_2_correct, num_1_correct = self.get_accuracy(predicted_gaze, gt_gazes)
			angle_between = self.get_angle_between(predicted_gaze, gt_gazes).cpu()
			num_within_degrees = self.how_many_within_degrees(angle_between)
			mean_angle_between = torch.mean(angle_between)

			# Update Params
			self.estimator_optimizer.zero_grad()
			loss.backward()
			self.estimator_optimizer.step()

			print('step: %d, loss: %.4f\n percent X correct: %.2f, percent Y correct: %.2f, percent Z correct %.2f\n percent 3 correct: %.2f, percent 2 correct: %.2f, percent 1 correct %.2f\n batch_size: %d, lr: %.5f' % (step, loss.item(), (xyz_correct[0] / self.cfg.batch_size) * 100.0, (xyz_correct[1] / self.cfg.batch_size) * 100.0, (xyz_correct[2] / self.cfg.batch_size) * 100.0, (num_3_correct / self.cfg.batch_size) * 100.0, (num_2_correct / self.cfg.batch_size) * 100.0,(num_1_correct / self.cfg.batch_size) * 100.0, self.cfg.batch_size, self.cfg.lr))

			print('percent within degree difference : {}, mean angle between: {}\n'.format((num_within_degrees / self.cfg.batch_size) * 100.0, mean_angle_between))

			
			# Save progress
			if step % self.cfg.save_interval == 0 and step > 0:
				print("Saving Model Dictionaries ... ")

				torch.save(self.estimator.state_dict(), os.path.join(self.cfg.checkpoint_path, self.cfg.estimator_path % step))
				state = {
					'step': step,
					'optE': self.estimator_optimizer.state_dict()
				}
				torch.save(state, os.path.join(self.cfg.checkpoint_path, self.cfg.optimizer_path))
			
			# If we run validation
			if not self.cfg.testingFile == None and step % self.cfg.validate_interval == 0:
				self.validate()

