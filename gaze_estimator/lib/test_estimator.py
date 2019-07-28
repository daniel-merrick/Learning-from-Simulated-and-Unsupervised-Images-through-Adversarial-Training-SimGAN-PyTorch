'''
	This file contains a class to test the GazeEstimatorNetwork
	
	TestEstimator inherits the SubEstimator class. This is 
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

class TestEstimator(SubEstimator):
	def __init__(self, cfg):
		SubEstimator.__init__(self, cfg)	
		self.cfg = cfg
		
	def test(self):
		self.build_network()
		self.get_data_loaders()

		print('Begin Testing .... :)')
		assert self.current_step < cfg.train_steps, 'Target step is smaller than current step! ... :('

		for step in range(self.current_step, cfg.train_steps):
			images, gt_gazes, imgPaths = self.data_loader.next_batch()
			images, gt_gazes = images.cuda(cfg.cuda_num), gt_gazes.cuda(cfg.cuda_num)

			predicted_gaze = self.estimator(images).cpu()
		
			loss = self.mse_loss(predicted_gaze, gt_gazes)
				
			xyz_correct, num_3_correct, num_2_correct, num_1_correct = self.get_accuracy(predicted_gaze, gt_gazes)
			
			mean_error = self.get_mean_error(predicted_gaze, gt_gazes).item()
			if mean_error == float("Inf"):
				continue
			results.append(mean_error)
			torch.cuda.empty_cache()

			print('step: %d, loss: %.2f\n percent X correct: %.2f, percent Y correct: %.2f, percent Z correct %.2f\n percent 3 correct: %.2f, percent 2 correct: %.2f, percent 1 correct %.2f\n batch_size: %d, lr: %.3f mean_error: %.2f\n' % (step, loss.item(), (xyz_correct[0] / cfg.batch_size) * 100.0, (xyz_correct[1] / cfg.batch_size) * 100.0, (xyz_correct[2] / cfg.batch_size) * 100.0, (num_3_correct / cfg.batch_size) * 100.0, (num_2_correct / cfg.batch_size) * 100.0,(num_1_correct / cfg.batch_size) * 100.0, cfg.batch_size, cfg.lr, mean_error))
			
		results = np.array(results)
		print('average gaze vector MSE : {}'.format(np.mean(results)))
