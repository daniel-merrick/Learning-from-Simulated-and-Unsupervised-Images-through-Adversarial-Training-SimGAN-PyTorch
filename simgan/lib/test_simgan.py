'''
	This file includes the class to train SimGAN
'''

import torch
import torch.nn as nn
import torchvision


from sub_simgan import SubSimGAN

class TestSimGAN(SubSimGAN):
	def __init__(self, cfg):
		SubSimGAN.__init__(self, cfg)
		
		self.feature_loss = None
		self.adv_loss = None
		self.loss_real = None
		self.loss_refined = None

		self.cfg = cfg
	
	def save_images(self, refined_images, imageFiles):
		image_tuple = torch.unbind(refined_images)
		
		for image, imageFile in zip(image_tuple, imageFiles):
			torchvision.utils.save_image(image, self.cfg.save_refined_path + imageFile[:-4] + '_refined.png')

		
		
	# Used to update the generator
	def next_refine(self, images):
		
		''' Get batch of synthetic images '''
		synthetic_images = images
		synthetic_images = synthetic_images.cuda(self.cfg.cuda_num)
		
		''' Refine synthetic images '''
		refined_images = self.R(synthetic_images)
		return refined_images

	''' main function called externally
		used to test images '''

	def refine(self):
		self.build_network()	
		self.get_data_loaders()

		''' If no saved weights are found,
			pretrain the refiner / discriminator '''
		if not self.weights_loaded:
			print('Returning because no weights were found!!')
			return

		for images, imageFiles in self.synthetic_data_loader:
			self.D.eval()
			self.D.train_mode(False)
						
			self.R.eval()
			self.R.train_mode(False)
			
					
			refined_images = self.next_refine(images)
			
			self.save_images(refined_images, imageFiles)
