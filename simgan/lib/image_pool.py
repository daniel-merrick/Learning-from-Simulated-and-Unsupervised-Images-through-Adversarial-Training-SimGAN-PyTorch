'''
	This file holds the class 'ImagePool'
	ImagePool holds a history of generated
	images. For more info, see the paper in 
	the README

	This is also taken from the repo posted 
	at the bottom of the README
'''

import random
import numpy as np
import torch
from torch.autograd import Variable

class ImagePool():
	def __init__(self, pool_size):
		self.pool_size = pool_size
		if self.pool_size > 0:
			self.num_imgs = 0
			self.images = None

	def query(self, img_var):
		img_npy = img_var.data.numpy()
		if self.pool_size == 0 or self.num_imgs < img_var.size(0):
			self.images = img_npy
			self.num_imgs = img_var.size(0)
			return img_var

		history_imgs = self.images[:int(img_npy.shape[0] / 2)]
		return_imgs  = np.concatenate((history_imgs, img_npy[:int(img_npy.shape[0] / 2)]), axis=0)
		self.images  = np.concatenate((self.images, img_npy), axis=0)
		np.random.shuffle(self.images)
		
		if self.images.shape[0] > self.pool_size:
			self.images = self.images[:self.pool_size]

		return Variable(torch.from_numpy(return_imgs))
